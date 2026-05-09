from __future__ import annotations

import os

from .train_loop_concept_helper_cpu2 import *

def configure_dataloader_worker_process(max_threads: int = 1) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(max_threads))
    except Exception:
        pass

    try:
        torch.set_num_threads(int(max_threads))
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(int(max_threads))
    except Exception:
        pass


class ThreadLimitedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that limits worker thread usage on first access."""

    def __init__(self, base_dataset: torch.utils.data.Dataset) -> None:
        self.base_dataset = base_dataset
        self._configured_pid: Optional[int] = None

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Any:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            pid = os.getpid()
            if self._configured_pid != pid:
                configure_dataloader_worker_process(max_threads=1)
                self._configured_pid = pid
        return self.base_dataset[index]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)


def assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"Non-finite detected in {name}: {value.detach().cpu().item()}")


def log_tensor_stats(name: str, value: torch.Tensor) -> None:
    value_fp32 = value.detach().float()
    finite = torch.isfinite(value_fp32)
    finite_ratio = float(finite.float().mean().cpu().item())
    min_value = float(value_fp32[finite].min().cpu().item()) if finite.any() else float("nan")
    max_value = float(value_fp32[finite].max().cpu().item()) if finite.any() else float("nan")
    print(f"{name}: finite_ratio={finite_ratio:.4f} min={min_value:.4e} max={max_value:.4e}")


def train(
    config: TrainConfig,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder: ReportPromptBuilder,
    loaders: Dict[str, DataLoader],
    concept_class_weights: Optional[torch.Tensor],
    balanced_train_loader: Optional[DataLoader],
    device: torch.device,
    autocast_dtype: torch.dtype,
    logger: MetricsLogger,
) -> TrainSummary:
    model.to(device)
    if device.type == "cuda":
        model.to(dtype=autocast_dtype)

    model = maybe_compile_model(model=model, config=config)

    trainable, total = count_trainable_params(unwrap_compiled_module(model))
    print(f"trainable_params={trainable:,} total_params={total:,}")
    logger.set_summary({"params/trainable": int(trainable), "params/total": int(total)})

    base_lr = float(config.lm_lr) if float(config.lm_lr) > 0 else float(config.lr)
    mm_projector_lr = float(config.adapter_lr) if float(config.adapter_lr) > 0 else None
    vision_tower_lr = None

    use_fused = str(config.optimizer_type).strip().lower() == "adamw_fused"
    optimizer = create_qwen3_vl_finetuning_optimizer(
        model=unwrap_compiled_module(model),
        base_lr=float(base_lr),
        weight_decay=float(config.weight_decay),
        mm_projector_lr=mm_projector_lr,
        vision_tower_lr=vision_tower_lr,
        adam_beta1=float(config.adam_beta1),
        adam_beta2=float(config.adam_beta2),
        adam_eps=float(config.adam_eps),
        use_fused=bool(use_fused),
    )

    effective_train_loader = balanced_train_loader if balanced_train_loader is not None else loaders["train"]
    steps_per_epoch = math.ceil(
        len(effective_train_loader) / max(1, int(config.grad_accum_steps))
    )
    total_steps = int(config.num_epochs) * int(steps_per_epoch) #### change it here !!!!!!!!!
    #total_steps = int(50) * int(steps_per_epoch) 
    warmup_steps = int(float(config.warmup_ratio) * float(total_steps))

    scheduler = create_warmup_decay_scheduler(
        optimizer=optimizer,
        num_training_steps=int(total_steps),
        num_warmup_steps=int(warmup_steps),
        scheduler_type=str(config.lr_scheduler_type),
        min_lr=float(config.min_lr),
    )

    use_scaler = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    best_val_report_acc = -1.0
    best_val_loss = float("inf")
    best_step = -1
    global_step = 0

    interval_total_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    interval_gen_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    interval_concept_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    interval_contrastive_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    interval_margin_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    interval_total_loss_count = 0

    interval_correct = torch.zeros((), device=device, dtype=torch.long)
    interval_total = torch.zeros((), device=device, dtype=torch.long)

    model.train()

    trainable_params = [p for p in unwrap_compiled_module(model).parameters() if p.requires_grad]
    test_metrics_computer = ReportMetricsComputer(
        #bleurt_checkpoint=BLEURT_CHECKPOINT,
        #bleurt_device=torch.device("cpu"),
        #bleurt_batch_size=16,
        #bleurt_max_length=128,
    )

    eval_loss_weights = EvalLossWeights(
        generation_loss_weight=float(0.6),
        concept_loss_weight=float(config.concept_loss_weight),
        contrastive_loss_weight=float(config.contrastive_loss_weight),
        contrastive_margin_weight=float(config.contrastive_margin_weight),
        contrastive_temperature=float(config.contrastive_temperature),
        contrastive_margin=float(config.contrastive_margin),
    )

    
    for epoch in range(int(config.num_epochs)):
        use_balanced_loader = (
            balanced_train_loader is not None
            and bool(config.use_concept_balanced_sampler)
            and int(epoch) >= int(config.concept_sampler_start_epoch)
        )
        epoch_loader = balanced_train_loader if use_balanced_loader else loaders["train"]
        num_batches = len(epoch_loader)

        drw_alpha = compute_drw_blend_factor(
            epoch=int(epoch),
            start_epoch=int(config.concept_drw_start_epoch),
            ramp_epochs=int(config.concept_drw_ramp_epochs),
        )

        epoch_class_weights: Optional[torch.Tensor] = None
        strategy_name = str(config.concept_imbalance_strategy).strip().lower()
        if strategy_name not in {"", "none"} and concept_class_weights is not None:
            blended_weights = blend_concept_class_weights(
                full_weights=concept_class_weights,
                blend_factor=float(drw_alpha),
            )
            if blended_weights is not None:
                epoch_class_weights = blended_weights.to(device=device, dtype=torch.float32)

        for step, batch in enumerate(epoch_loader):
            video_features = batch["video_features"].to(device=device, dtype=autocast_dtype, non_blocking=True)
            video_mask = batch["video_mask"].to(device=device, non_blocking=True)
            input_ids = batch["input_ids"].to(device=device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device=device, non_blocking=True)
            labels = batch["labels"].to(device=device, non_blocking=True)
            concept_targets = batch["concept_targets"].to(device=device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=(device.type == "cuda"),
            ):
                out = model(
                    video_features=video_features,
                    video_mask=video_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    concept_targets=concept_targets,
                    study_ids=batch.get("exam_id"),
                )

                gen_loss = out.loss

                concept_loss = torch.zeros((), device=gen_loss.device, dtype=gen_loss.dtype)
                concept_logits = getattr(out, "concept_logits", None)
                function_logits = getattr(out, "function_logits", None)
                '''
                if concept_logits is not None:
                    log_tensor_stats("concept_logits", concept_logits)

                if epoch_class_weights is not None:
                    log_tensor_stats("epoch_class_weights", epoch_class_weights) '''
                if concept_logits is not None:
                    '''concept_ce = masked_weighted_cross_entropy_mean_by_concept(
                        logits=concept_logits,
                        targets=concept_targets,
                        class_weight_by_concept=epoch_class_weights,
                        ignore_index=int(IGNORE_INDEX),
                        label_smoothing=float(config.concept_loss_label_smoothing),
                        balance_across_concepts=bool(config.concept_balance_across_concepts),
                    )'''
                    with torch.autocast(device_type=device.type, enabled=False):
                        concept_ce = masked_weighted_cross_entropy_mean_by_concept(
                            logits=concept_logits.float(),
                            targets=concept_targets,
                            class_weight_by_concept=None if epoch_class_weights is None else epoch_class_weights.float(),
                            ignore_index=int(IGNORE_INDEX),
                            label_smoothing=float(config.concept_loss_label_smoothing),
                            balance_across_concepts=bool(config.concept_balance_across_concepts),
                        )


                    if function_logits is not None:
                        function_ce = masked_weighted_cross_entropy_mean_by_concept(
                            logits=function_logits,
                            targets=concept_targets,
                            class_weight_by_concept=epoch_class_weights,
                            ignore_index=int(IGNORE_INDEX),
                            label_smoothing=float(config.concept_loss_label_smoothing),
                            balance_across_concepts=bool(config.concept_balance_across_concepts),
                        )
                        concept_loss = 0.5 * concept_ce + 0.5 * function_ce
                    else:
                        concept_loss = concept_ce

                contrastive_loss = torch.zeros((), device=gen_loss.device, dtype=gen_loss.dtype)
                margin_loss = torch.zeros((), device=gen_loss.device, dtype=gen_loss.dtype)

                if float(config.contrastive_loss_weight) > 0.0:
                    video_repr = f.normalize(out.video_repr, dim=-1)
                    text_repr = f.normalize(out.text_repr, dim=-1)
                    contrastive_loss, margin_loss = compute_contrastive_losses(
                        video_repr=video_repr,
                        text_repr=text_repr,
                        temperature=float(config.contrastive_temperature),
                        margin=float(config.contrastive_margin),
                    )

                total_loss = (
                    float(0.6) * gen_loss
                    + float(config.concept_loss_weight) * concept_loss
                    + float(config.contrastive_loss_weight) * contrastive_loss
                    + float(config.contrastive_margin_weight) * margin_loss
                )

                loss = total_loss / float(max(1, int(config.grad_accum_steps)))

            interval_total_loss_sum = interval_total_loss_sum + total_loss.detach().to(dtype=torch.float32)
            interval_gen_loss_sum = interval_gen_loss_sum + gen_loss.detach().to(dtype=torch.float32)
            interval_concept_loss_sum = interval_concept_loss_sum + concept_loss.detach().to(dtype=torch.float32)
            interval_contrastive_loss_sum = interval_contrastive_loss_sum + contrastive_loss.detach().to(dtype=torch.float32)
            interval_margin_loss_sum = interval_margin_loss_sum + margin_loss.detach().to(dtype=torch.float32)
            interval_total_loss_count += 1

            interval_correct = interval_correct + out.token_correct.detach()
            interval_total = interval_total + out.token_total.detach()

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_accum_step = (step + 1) % int(config.grad_accum_steps) == 0
            is_last_batch = (step + 1) == int(num_batches)
            should_step = bool(is_accum_step or is_last_batch)

            if not should_step:
                continue

            if float(config.max_grad_norm) > 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=float(config.max_grad_norm),
                )

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if global_step % int(config.log_every_steps) == 0:
                avg_total_loss = (interval_total_loss_sum / max(1, interval_total_loss_count)).item()
                avg_gen_loss = (interval_gen_loss_sum / max(1, interval_total_loss_count)).item()
                avg_concept_loss = (interval_concept_loss_sum / max(1, interval_total_loss_count)).item()
                avg_contrastive_loss = (interval_contrastive_loss_sum / max(1, interval_total_loss_count)).item()
                avg_margin_loss = (interval_margin_loss_sum / max(1, interval_total_loss_count)).item()

                token_acc = (
                    float(interval_correct.float().div(interval_total.clamp(min=1)).item())
                    if int(interval_total.item()) > 0
                    else 0.0
                )
                lr_value = float(scheduler.get_last_lr()[0])

                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={avg_total_loss:.4f} gen={avg_gen_loss:.4f} "
                    f"concept={avg_concept_loss:.4f} cont={avg_contrastive_loss:.4f} "
                    f"margin={avg_margin_loss:.4f} token_acc={token_acc:.4f} lr={lr_value:.6e}"
                )

                debug_metrics: Dict[str, Any] = {}
                base_model = unwrap_compiled_module(model)
                if isinstance(base_model, EchoReportVlm) and isinstance(base_model.adapter.last_projected_feature_stats, dict):
                    debug_metrics = {f"debug/projected/{k}": v for k, v in base_model.adapter.last_projected_feature_stats.items()}

                logger.log(
                    {
                        "train/loss": float(avg_total_loss),
                        "train/gen_loss": float(avg_gen_loss),
                        "train/concept_loss": float(avg_concept_loss),
                        "train/contrastive_loss": float(avg_contrastive_loss),
                        "train/contrastive_margin_loss": float(avg_margin_loss),
                        "train/token_acc": float(token_acc),
                        "train/lr": float(lr_value),
                        "train/epoch": int(epoch),
                        "train/concept_drw_alpha": float(drw_alpha),
                        "train/use_concept_balanced_sampler": int(use_balanced_loader),
                        **debug_metrics,
                    },
                    step=int(global_step),
                )

                interval_total_loss_sum.zero_()
                interval_gen_loss_sum.zero_()
                interval_concept_loss_sum.zero_()
                interval_contrastive_loss_sum.zero_()
                interval_margin_loss_sum.zero_()
                interval_total_loss_count = 0
                interval_correct.zero_()
                interval_total.zero_()
                
            if global_step % int(config.save_every_steps) == 0:
                save_checkpoint(
                    output_dir=config.output_dir,
                    step=global_step,
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                )
            '''
            if global_step % int(config.eval_every_steps) == 0:
                val_metrics = run_eval_metrics(
                    model=model,
                    loader=loaders["val"],
                    device=device,
                    autocast_dtype=autocast_dtype,
                    ignore_index=int(IGNORE_INDEX),
                    concept_names=CONCEPT_NAMES,
                    concept_specs=CONCEPT_SPECS,
                    region_to_indices=REGION_TO_CONCEPT_INDICES,
                    loss_weights=eval_loss_weights,
                )
                val_report_acc = run_report_accuracy(
                    model=unwrap_compiled_module(model),
                    tokenizer=tokenizer,
                    prompt_builder=prompt_builder,
                    loader=loaders["val"],
                    device=device,
                    autocast_dtype=autocast_dtype,
                    max_prompt_tokens=config.max_prompt_tokens,
                    gen_max_new_tokens=config.gen_max_new_tokens,
                )

                print(
                    f"epoch={epoch} step={global_step} "
                    f"val_loss={val_metrics['loss']:.4f} val_gen_loss={val_metrics['gen_loss']:.4f} "
                    f"val_concept_loss={val_metrics['concept_loss']:.4f} "
                    f"val_cont_loss={val_metrics['contrastive_loss']:.4f} "
                    f"val_margin_loss={val_metrics['contrastive_margin_loss']:.4f} "
                    f"val_token_acc={val_metrics['token_acc']:.4f} "
                    f"val_concept_acc={val_metrics['concept_acc_overall']:.4f} "
                    f"val_report_acc={val_report_acc:.4f}"
                )

                logger.log(
                    {
                        "val/loss": float(val_metrics["loss"]),
                        "val/gen_loss": float(val_metrics["gen_loss"]),
                        "val/token_acc": float(val_metrics["token_acc"]),
                        "val/concept_loss": float(val_metrics["concept_loss"]),
                        "val/contrastive_loss": float(val_metrics["contrastive_loss"]),
                        "val/contrastive_margin_loss": float(val_metrics["contrastive_margin_loss"]),
                        "val/concept_acc_overall": float(val_metrics["concept_acc_overall"]),
                        "val/report_word_acc": float(val_report_acc),
                        "val/epoch": int(epoch),
                        **{f"val/{k}": float(v) for k, v in val_metrics.items() if k.startswith("concept_acc/")},
                        **{f"val/{k}": float(v) for k, v in val_metrics.items() if k.startswith("concept_acc_region/")},
                        **{f"val/{k}": float(v) for k, v in val_metrics.items() if k.startswith("concept_count/")},
                        **{f"val/{k}": float(v) for k, v in val_metrics.items() if k.startswith("concept_auroc/")},
                    },
                    step=int(global_step),
                )

                if val_report_acc > best_val_report_acc:
                    best_val_report_acc = float(val_report_acc)
                    best_val_loss = float(val_metrics["loss"])
                    best_step = int(global_step)

                    save_best_checkpoint(
                        output_dir=config.output_dir,
                        step=global_step,
                        model=model,
                        tokenizer=tokenizer,
                        val_loss=best_val_loss,
                        val_report_acc=best_val_report_acc,
                        config=config,
                    )

                    logger.log(
                        {
                            "best/val_loss": float(best_val_loss),
                            "best/val_report_word_acc": float(best_val_report_acc),
                            "best/step": int(best_step),
                        },
                        step=int(global_step),
                    )
                    logger.set_summary(
                        {
                            "best_step": int(best_step),
                            "best_val_loss": float(best_val_loss),
                            "best_val_report_word_acc": float(best_val_report_acc),
                        }
                    )

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if global_step % int(config.save_every_steps) == 0:
                save_checkpoint(
                    output_dir=config.output_dir,
                    step=global_step,
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                )

        val_metrics = run_eval_metrics(
            model=model,
            loader=loaders["val"],
            device=device,
            autocast_dtype=autocast_dtype,
            ignore_index=int(IGNORE_INDEX),
            concept_names=CONCEPT_NAMES,
            concept_specs=CONCEPT_SPECS,
            region_to_indices=REGION_TO_CONCEPT_INDICES,
            loss_weights=eval_loss_weights,
        )
        val_report_acc = run_report_accuracy(
            model=unwrap_compiled_module(model),
            tokenizer=tokenizer,
            prompt_builder=prompt_builder,
            loader=loaders["val"],
            device=device,
            autocast_dtype=autocast_dtype,
            max_prompt_tokens=config.max_prompt_tokens,
            gen_max_new_tokens=config.gen_max_new_tokens,
        )
        test_eval_metrics = run_eval_metrics(
            model=model,
            loader=loaders["test"],
            device=device,
            autocast_dtype=autocast_dtype,
            ignore_index=int(IGNORE_INDEX),
            concept_names=CONCEPT_NAMES,
            concept_specs=CONCEPT_SPECS,
            region_to_indices=REGION_TO_CONCEPT_INDICES,
            loss_weights=eval_loss_weights,
        )

        print(
            f"epoch={epoch} "
            f"test_loss={test_eval_metrics['loss']:.4f} test_gen_loss={test_eval_metrics['gen_loss']:.4f} "
            f"test_concept_loss={test_eval_metrics['concept_loss']:.4f} "
            f"test_cont_loss={test_eval_metrics['contrastive_loss']:.4f} "
            f"test_margin_loss={test_eval_metrics['contrastive_margin_loss']:.4f} "
            f"test_token_acc={test_eval_metrics['token_acc']:.4f} "
            f"test_concept_acc={test_eval_metrics['concept_acc_overall']:.4f}"
        )

        logger.log(
            {
                "test/loss_epoch": float(test_eval_metrics["loss"]),
                "test/gen_loss_epoch": float(test_eval_metrics["gen_loss"]),
                "test/token_acc_epoch": float(test_eval_metrics["token_acc"]),
                "test/concept_loss_epoch": float(test_eval_metrics["concept_loss"]),
                "test/contrastive_loss_epoch": float(test_eval_metrics["contrastive_loss"]),
                "test/contrastive_margin_loss_epoch": float(test_eval_metrics["contrastive_margin_loss"]),
                "test/concept_acc_overall_epoch": float(test_eval_metrics["concept_acc_overall"]),
                "test/concept_balanced_acc_overall": float(test_eval_metrics["concept_balanced_acc_overall"]),
                "test/concept_f1_overall": float(test_eval_metrics["concept_f1_overall"]),
                "test/epoch": int(epoch),
                **{f"test/{k}": float(v) for k, v in test_eval_metrics.items() if k.startswith("concept_acc/")},
                **{f"test/{k}": float(v) for k, v in test_eval_metrics.items() if k.startswith("concept_acc_region/")},
                **{f"test/{k}": float(v) for k, v in test_eval_metrics.items() if k.startswith("concept_auroc/")},
                **{f"test/{k}": float(v) for k, v in test_eval_metrics.items() if k.startswith("concept_balanced_acc/")},
                **{f"test/{k}": float(v) for k, v in test_eval_metrics.items() if k.startswith("concept_f1/")},
            },
            step=int(global_step),
        )


        logger.log(
            {
                "val/loss_epoch": float(val_metrics["loss"]),
                "val/gen_loss_epoch": float(val_metrics["gen_loss"]),
                "val/token_acc_epoch": float(val_metrics["token_acc"]),
                "val/concept_loss_epoch": float(val_metrics["concept_loss"]),
                "val/contrastive_loss_epoch": float(val_metrics["contrastive_loss"]),
                "val/contrastive_margin_loss_epoch": float(val_metrics["contrastive_margin_loss"]),
                "val/concept_acc_overall_epoch": float(val_metrics["concept_acc_overall"]),
                "val/concept_balanced_acc_overall": float(val_metrics["concept_balanced_acc_overall"]),
                "val/concept_f1_overall": float(val_metrics["concept_f1_overall"]),
                "val/report_word_acc_epoch": float(val_report_acc),
                "val/epoch": int(epoch),
                **{f"val/{k}": float(v) for k, v in val_metrics.items() if k.startswith("concept_auroc/")},
                **{f"val/{k}": float(v) for k, v in val_metrics.items() if k.startswith("concept_balanced_acc/")},
                **{f"val/{k}": float(v) for k, v in val_metrics.items() if k.startswith("concept_f1/")},
            },
            step=int(global_step),
        )

        if val_report_acc > best_val_report_acc:
            best_val_report_acc = float(val_report_acc)
            best_val_loss = float(val_metrics["loss"])
            best_step = int(global_step)
            save_best_checkpoint(
                output_dir=config.output_dir,
                step=global_step,
                model=model,
                tokenizer=tokenizer,
                val_loss=best_val_loss,
                val_report_acc=best_val_report_acc,
                config=config,
            )

        test_summary = run_test_generation_and_save_metrics(
            model=unwrap_compiled_module(model),
            tokenizer=tokenizer,
            prompt_builder=prompt_builder,
            loader=loaders["test"],
            device=device,
            autocast_dtype=autocast_dtype,
            max_prompt_tokens=config.max_prompt_tokens,
            gen_max_new_tokens=config.gen_max_new_tokens,
            metrics_computer=test_metrics_computer,
            output_csv_path=GENERATED_ALL_TEST_CSV_PATH,
            concept_names=CONCEPT_NAMES,
            concept_specs=CONCEPT_SPECS,
            ignore_index=int(IGNORE_INDEX),
        )

        print(
            "epoch="
            f"{epoch} test_bleu1={test_summary['bleu_1']:.4f} "
            f"test_bleu2={test_summary['bleu_2']:.4f} "
            f"test_bleu3={test_summary['bleu_3']:.4f} "
            f"test_bleu4={test_summary['bleu_4']:.4f} "
            f"test_rougeL={test_summary['rouge_l']:.4f} "
            f"test_meteor={test_summary['meteor']:.4f} "
            f"test_cider={test_summary['cider']:.4f} "
            #f"test_bleurt={test_summary['bleurt']:.4f} "
            f"test_ce_p={test_summary['ce_precision']:.4f} "
            f"test_ce_r={test_summary['ce_recall']:.4f} "
            f"test_ce_f1={test_summary['ce_f1']:.4f} "
            f"csv={str(GENERATED_ALL_TEST_CSV_PATH)}"
        )

        logger.log(
            {
                "test/bleu_1": float(test_summary["bleu_1"]),
                "test/bleu_2": float(test_summary["bleu_2"]),
                "test/bleu_3": float(test_summary["bleu_3"]),
                "test/bleu_4": float(test_summary["bleu_4"]),
                "test/rouge_l": float(test_summary["rouge_l"]),
                "test/meteor": float(test_summary["meteor"]),
                "test/cider": float(test_summary["cider"]),
                #"test/bleurt": float(test_summary["bleurt"]),
                "test/ce_precision": float(test_summary["ce_precision"]),
                "test/ce_recall": float(test_summary["ce_recall"]),
                "test/ce_f1": float(test_summary["ce_f1"]),
                "test/num_examples": float(test_summary["num_examples"]),
                "test/epoch": int(epoch),
            },
            step=int(global_step),
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()'''

    save_checkpoint(
        output_dir=config.output_dir,
        step=global_step,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    logger.set_summary(
        {
            "final_step": int(global_step),
            "best_step": int(best_step),
            "best_val_loss": float(best_val_loss),
            "best_val_report_word_acc": float(best_val_report_acc),
        }
    )

    return TrainSummary(
        best_step=int(best_step),
        best_val_loss=float(best_val_loss),
        best_val_report_acc=float(best_val_report_acc),
        final_step=int(global_step),
    )

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lm_lr", type=float, default=-1.0)
    parser.add_argument("--adapter_lr", type=float, default=-1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "constant"])
    parser.add_argument("--min_lr", type=float, default=0.0)

    parser.add_argument("--optimizer_type", type=str, default="adamw_torch", choices=["adamw_torch", "adamw_fused"])
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)

    parser.add_argument("--max_prompt_tokens", type=int, default=512)
    parser.add_argument("--max_target_tokens", type=int, default=768)

    parser.add_argument("--lm_lora_r", type=int, default=4)
    parser.add_argument("--lm_lora_alpha", type=int, default=32)
    parser.add_argument("--lm_lora_dropout", type=float, default=0.1)
    # parser.add_argument(
    #     "--lm_target_modules",
    #     type=str,
    #     default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", 
    # )  # include the linear 
    # drop linear  
    parser.add_argument(
        "--lm_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
    )

    parser.add_argument("--num_visual_tokens", type=int, default=256)

    parser.add_argument("--projector_layers", type=int, default=2)
    parser.add_argument("--projector_hidden_ratio", type=float, default=2.0)
    parser.add_argument("--projector_dropout", type=float, default=0.2)

    parser.add_argument("--adapter_layers", type=int, default=4)
    parser.add_argument("--adapter_heads", type=int, default=8)
    parser.add_argument("--adapter_attn_dropout", type=float, default=0.1)
    parser.add_argument("--adapter_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--adapter_mlp_dropout", type=float, default=0.1)

    parser.add_argument("--disable_projected_feature_check", action="store_true")
    parser.add_argument("--projected_feature_cosine_threshold", type=float, default=0.9995)
    parser.add_argument("--projected_feature_max_pairs_to_log", type=int, default=8)

    parser.add_argument("--use_mask_template_prompt", action="store_true")

    parser.add_argument("--concept_csv_path", type=str, default=str(DEFAULT_CONCEPT_CSV_PATH))
    parser.add_argument("--concept_loss_weight", type=float, default=0.2)


    parser.add_argument(
        "--concept_imbalance_strategy",
        type=str,
        default="class_balanced",
        choices=["none", "weighted_ce", "inverse_freq", "class_balanced"],
    )
    parser.add_argument("--concept_loss_label_smoothing", type=float, default=0.02)
    parser.add_argument("--concept_cb_beta", type=float, default=0.9999)
    parser.add_argument("--concept_class_weight_power", type=float, default=0.5) # possible some classes labels are non
    parser.add_argument("--concept_max_class_weight", type=float, default=5)
    parser.add_argument("--concept_drw_start_epoch", type=int, default=10)
    parser.add_argument("--concept_drw_ramp_epochs", type=int, default=10)
    parser.add_argument(
        "--concept_balance_across_concepts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--use_concept_balanced_sampler",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--concept_sampler_start_epoch", type=int, default=1)
    parser.add_argument(
        "--concept_sampler_reduction",
        type=str,
        default="mean",
        choices=["max", "mean"],
    )
    parser.add_argument("--concept_sampler_weight_power", type=float, default=1.0)
    parser.add_argument("--concept_sampler_max_weight", type=float, default=20.0)

    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1)
    parser.add_argument("--contrastive_margin_weight", type=float, default=0.05)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--contrastive_margin", type=float, default=0.2)

    parser.add_argument("--index_range", type=int, default=10000)
    parser.add_argument("--max_videos_per_study", type=int, default=MAX_VIDEOS_PER_STUDY)

    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--eval_every_steps", type=int, default=100)
    parser.add_argument("--save_every_steps", type=int, default=200)

    parser.add_argument("--gen_max_new_tokens", type=int, default=512)

    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lm_head_chunk_size", type=int, default=64)
    parser.add_argument("--enable_tf32", action="store_true")

    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--torch_compile_dynamic", action="store_true")
    parser.add_argument("--torch_compile_fullgraph", action="store_true")

    args = parser.parse_args()

    return TrainConfig(
        model_name_or_path=str(args.model_name_or_path),
        output_dir=Path(args.output_dir),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        lr=float(args.lr),
        lm_lr=float(args.lm_lr),
        adapter_lr=float(args.adapter_lr),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        num_epochs=int(args.num_epochs),
        grad_accum_steps=int(args.grad_accum_steps),
        max_grad_norm=float(args.max_grad_norm),
        lr_scheduler_type=str(args.lr_scheduler_type),
        min_lr=float(args.min_lr),
        optimizer_type=str(args.optimizer_type),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_eps=float(args.adam_eps),
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_target_tokens=int(args.max_target_tokens),
        lm_lora_r=int(args.lm_lora_r),
        lm_lora_alpha=int(args.lm_lora_alpha),
        lm_lora_dropout=float(args.lm_lora_dropout),
        lm_target_modules=str(args.lm_target_modules),
        num_visual_tokens=int(args.num_visual_tokens),
        projector_layers=int(args.projector_layers),
        projector_hidden_ratio=float(args.projector_hidden_ratio),
        projector_dropout=float(args.projector_dropout),
        adapter_layers=int(args.adapter_layers),
        adapter_heads=int(args.adapter_heads),
        adapter_attn_dropout=float(args.adapter_attn_dropout),
        adapter_mlp_ratio=float(args.adapter_mlp_ratio),
        adapter_mlp_dropout=float(args.adapter_mlp_dropout),
        projected_feature_check=not bool(args.disable_projected_feature_check),
        projected_feature_cosine_threshold=float(args.projected_feature_cosine_threshold),
        projected_feature_max_pairs_to_log=int(args.projected_feature_max_pairs_to_log),
        use_mask_template_prompt=bool(args.use_mask_template_prompt),
        concept_csv_path=Path(str(args.concept_csv_path)),
        concept_loss_weight=float(args.concept_loss_weight),
        concept_loss_label_smoothing=float(args.concept_loss_label_smoothing),
        concept_imbalance_strategy=str(args.concept_imbalance_strategy),
        concept_cb_beta=float(args.concept_cb_beta),
        concept_class_weight_power=float(args.concept_class_weight_power),
        concept_max_class_weight=float(args.concept_max_class_weight),
        concept_drw_start_epoch=int(args.concept_drw_start_epoch),
        concept_drw_ramp_epochs=int(args.concept_drw_ramp_epochs),
        concept_balance_across_concepts=bool(args.concept_balance_across_concepts),

        use_concept_balanced_sampler=bool(args.use_concept_balanced_sampler),
        concept_sampler_start_epoch=int(args.concept_sampler_start_epoch),
        concept_sampler_reduction=str(args.concept_sampler_reduction),
        concept_sampler_weight_power=float(args.concept_sampler_weight_power),
        concept_sampler_max_weight=float(args.concept_sampler_max_weight),

        contrastive_loss_weight=float(args.contrastive_loss_weight),
        contrastive_margin_weight=float(args.contrastive_margin_weight),
        contrastive_temperature=float(args.contrastive_temperature),
        contrastive_margin=float(args.contrastive_margin),
        index_range=int(args.index_range),
        max_videos_per_study=int(args.max_videos_per_study),
        log_every_steps=int(args.log_every_steps),
        eval_every_steps=int(args.eval_every_steps),
        save_every_steps=int(args.save_every_steps),
        gen_max_new_tokens=int(args.gen_max_new_tokens),
        wandb_project=str(args.wandb_project),
        wandb_entity=str(args.wandb_entity),
        wandb_run_name=str(args.wandb_run_name),
        wandb_tags=str(args.wandb_tags),
        wandb_mode=str(args.wandb_mode),
        attn_implementation=str(args.attn_implementation),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        lm_head_chunk_size=int(args.lm_head_chunk_size),
        enable_tf32=bool(args.enable_tf32),
        torch_compile=bool(args.torch_compile),
        torch_compile_mode=str(args.torch_compile_mode),
        torch_compile_dynamic=bool(args.torch_compile_dynamic),
        torch_compile_fullgraph=bool(args.torch_compile_fullgraph),
    )



def main() -> None:
    replace_qwen2_vl_attention_class()

    config = parse_args()
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    configure_torch_backends(enable_tf32=config.enable_tf32)

    logger_obj = create_logger(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = select_torch_dtype()

    concept_table = ConceptLabelTable.from_csv(
        csv_path=config.concept_csv_path,
        concept_specs=CONCEPT_SPECS,
        ignore_index=int(IGNORE_INDEX),
    )

    masked_findings_by_exam_id = load_findings_by_exam_id(MASK_JSON_PATH)
    gt_findings_by_exam_id = load_findings_by_exam_id(GT_JSON_PATH)

    items = process_dicoms(
        masked_findings_by_exam_id=masked_findings_by_exam_id,
        gt_findings_by_exam_id=gt_findings_by_exam_id,
        index_range=config.index_range,
        embedding_dir=PATCH_FEATURES_DIR,
        max_videos_per_study=config.max_videos_per_study,
    )

    dataset = EchoPrimeReportDataset(items)
    dataset = ThreadLimitedDataset(base_dataset=dataset)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    frozen_vocab_size = int(len(tokenizer))

    #mask_token = "<MASK>"
    special_tokens = build_concept_and_label_special_tokens(CONCEPT_SPECS)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    concept_token_ids = build_concept_token_ids(tokenizer=tokenizer, concept_names=CONCEPT_NAMES)
    concept_label_token_ids, concept_label_token_mask = build_concept_label_token_id_tensor(
        tokenizer=tokenizer,
        concept_specs=CONCEPT_SPECS,
    )

    prompt_builder = ReportPromptBuilder(use_mask_template_prompt=config.use_mask_template_prompt)
    base_collator = VlmDataCollator(
        tokenizer=tokenizer,
        prompt_builder=prompt_builder,
        max_prompt_tokens=config.max_prompt_tokens,
        max_target_tokens=config.max_target_tokens,
    )
    collator = ConceptAwareVlmDataCollator(
        base_collator=base_collator,
        concept_table=concept_table,
    )

    loaders = create_data_loaders(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        collate_fn=collator,
    )


    train_exam_ids = extract_exam_ids_from_dataset(loaders["train"].dataset)
    train_class_counts = compute_concept_class_counts(
        exam_ids=train_exam_ids,
        concept_table=concept_table,
        concept_specs=CONCEPT_SPECS,
        ignore_index=int(IGNORE_INDEX),
    )

    full_concept_class_weights = build_concept_class_weight_tensor(
        concept_specs=CONCEPT_SPECS,
        class_counts_by_concept=train_class_counts,
        strategy=str(config.concept_imbalance_strategy),
        cb_beta=float(config.concept_cb_beta),
        weight_power=float(config.concept_class_weight_power),
        max_weight=float(config.concept_max_class_weight),
    )

    balanced_train_loader: Optional[DataLoader] = None
    if bool(config.use_concept_balanced_sampler):
        sample_weights = compute_sample_weights_for_concept_balanced_sampler(
            exam_ids=train_exam_ids,
            concept_table=concept_table,
            class_weight_by_concept=full_concept_class_weights,
            ignore_index=int(IGNORE_INDEX),
            reduction=str(config.concept_sampler_reduction),
            weight_power=float(config.concept_sampler_weight_power),
            max_weight=float(config.concept_sampler_max_weight),
        )

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=int(len(sample_weights)),
            replacement=True,
        )

        base_train_loader = loaders["train"]
        balanced_train_loader = DataLoader(
            dataset=base_train_loader.dataset,
            batch_size=int(config.batch_size),
            sampler=sampler,
            num_workers=int(config.num_workers),
            collate_fn=base_train_loader.collate_fn,
            drop_last=bool(getattr(base_train_loader, "drop_last", False)),
            pin_memory=bool(getattr(base_train_loader, "pin_memory", False)),
        )

    lm = build_lm_with_lora(
        model_name_or_path=config.model_name_or_path,
        tokenizer=tokenizer,
        lora_r=config.lm_lora_r,
        lora_alpha=config.lm_lora_alpha,
        lora_dropout=config.lm_lora_dropout,
        torch_dtype=autocast_dtype,
        target_modules_spec=config.lm_target_modules,
        attn_implementation=config.attn_implementation,
    )

    enable_training_on_new_tokens(model=lm, frozen_vocab_size=int(frozen_vocab_size))

    if config.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(lm)
        if hasattr(lm, "config"):
            lm.config.use_cache = False

    hidden_size = get_lm_hidden_size(lm)
    concept_query_init = lm.get_input_embeddings()(concept_token_ids).detach().to(dtype=torch.float32)

    adapter = VideoFeatureAdapter(
        input_dim=768,
        lm_hidden_size=hidden_size,
        num_report_tokens=int(config.num_visual_tokens),
        num_concept_tokens=int(len(CONCEPT_NAMES)),
        concept_query_init=concept_query_init,
        projector_layers=config.projector_layers,
        projector_hidden_ratio=config.projector_hidden_ratio,
        projector_dropout=config.projector_dropout,
        num_layers=config.adapter_layers,
        num_heads=config.adapter_heads,
        attn_dropout=config.adapter_attn_dropout,
        mlp_ratio=config.adapter_mlp_ratio,
        mlp_dropout=config.adapter_mlp_dropout,
        enable_projected_feature_check=config.projected_feature_check,
        projected_feature_cosine_threshold=config.projected_feature_cosine_threshold,
        projected_feature_max_pairs_to_log=config.projected_feature_max_pairs_to_log,
        concept_names=CONCEPT_NAMES,
    )

    model = EchoReportVlm(
        lm=lm,
        adapter=adapter,
        num_report_visual_tokens=int(config.num_visual_tokens),
        num_concept_tokens=int(len(CONCEPT_NAMES)),
        concept_label_token_ids=concept_label_token_ids,
        concept_label_token_mask=concept_label_token_mask,
        lm_head_chunk_size=config.lm_head_chunk_size,
    )

    summary = train(
        config=config,
        model=model,
        tokenizer=tokenizer,
        prompt_builder=prompt_builder,
        loaders=loaders,
        concept_class_weights=full_concept_class_weights,
        balanced_train_loader=balanced_train_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        logger=logger_obj,
    )

    best_dir = config.output_dir / "checkpoint-best"
    if not best_dir.exists():
        print("!!!!!!!!!!!!!!!!!!!!! wrong checkpoint-best not found; skipping best-model evaluation")
        logger_obj.finish()
        return None

    best_checkpoint_dir = resolve_checkpoint_dir(best_dir)

    best_tokenizer = AutoTokenizer.from_pretrained(
        best_checkpoint_dir / "tokenizer",
        trust_remote_code=True,
        use_fast=True,
    )
    if best_tokenizer.pad_token_id is None:
        best_tokenizer.pad_token = best_tokenizer.eos_token
    best_tokenizer.padding_side = "right"

    best_concept_token_ids = build_concept_token_ids(best_tokenizer, CONCEPT_NAMES)
    best_concept_label_token_ids, best_concept_label_token_mask = build_concept_label_token_id_tensor(
        tokenizer=best_tokenizer,
        concept_specs=CONCEPT_SPECS,
    )

    best_model, best_tokenizer = load_best_model_from_checkpoint(
        checkpoint_dir=best_checkpoint_dir,
        base_model_name_or_path=config.model_name_or_path,
        config=config,
        torch_dtype=autocast_dtype,
        concept_label_token_ids=best_concept_label_token_ids,
        concept_label_token_mask=best_concept_label_token_mask,
        concept_token_ids=best_concept_token_ids,
    )


    
    best_model.to(device)
    if device.type == "cuda":
        best_model.to(dtype=autocast_dtype)

    eval_loss_weights = EvalLossWeights(
        generation_loss_weight=float(0.6),
        concept_loss_weight=float(config.concept_loss_weight),
        contrastive_loss_weight=float(config.contrastive_loss_weight),
        contrastive_margin_weight=float(config.contrastive_margin_weight),
        contrastive_temperature=float(config.contrastive_temperature),
        contrastive_margin=float(config.contrastive_margin),
    )

    best_val_metrics = run_eval_metrics(
        model=best_model,
        loader=loaders["val"],
        device=device,
        autocast_dtype=autocast_dtype,
        ignore_index=int(IGNORE_INDEX),
        concept_names=CONCEPT_NAMES,
        concept_specs=CONCEPT_SPECS,
        region_to_indices=REGION_TO_CONCEPT_INDICES,
        loss_weights=eval_loss_weights,
    )
    best_val_report_acc = run_report_accuracy(
        model=best_model,
        tokenizer=best_tokenizer,
        prompt_builder=prompt_builder,
        loader=loaders["val"],
        device=device,
        autocast_dtype=autocast_dtype,
        max_prompt_tokens=config.max_prompt_tokens,
        gen_max_new_tokens=config.gen_max_new_tokens,
    )
    print(
        f"best_checkpoint val_loss={best_val_metrics['loss']:.4f} "
        f"val_gen_loss={best_val_metrics['gen_loss']:.4f} "
        f"val_contrastive_loss={best_val_metrics['contrastive_loss']:.4f} "
        f"val_margin_loss={best_val_metrics['contrastive_margin_loss']:.4f} "
        f"val_token_acc={best_val_metrics['token_acc']:.4f} "
        f"val_concept_loss={best_val_metrics['concept_loss']:.4f} "
        f"val_concept_acc={best_val_metrics['concept_acc_overall']:.4f} "
        f"val_report_acc={best_val_report_acc:.4f}"
    )

    test_metrics = run_eval_metrics(
        model=best_model,
        loader=loaders["test"],
        device=device,
        autocast_dtype=autocast_dtype,
        ignore_index=int(IGNORE_INDEX),
        concept_names=CONCEPT_NAMES,
        concept_specs=CONCEPT_SPECS,
        region_to_indices=REGION_TO_CONCEPT_INDICES,
        loss_weights=eval_loss_weights,
    )
    test_report_acc = run_report_accuracy(
        model=best_model,
        tokenizer=best_tokenizer,
        prompt_builder=prompt_builder,
        loader=loaders["test"],
        device=device,
        autocast_dtype=autocast_dtype,
        max_prompt_tokens=config.max_prompt_tokens,
        gen_max_new_tokens=config.gen_max_new_tokens,
    )
    print(
        f"best_checkpoint test_loss={test_metrics['loss']:.4f} "
        f"test_gen_loss={test_metrics['gen_loss']:.4f} "
        f"test_contrastive_loss={test_metrics['contrastive_loss']:.4f} "
        f"test_margin_loss={test_metrics['contrastive_margin_loss']:.4f} "
        f"test_token_acc={test_metrics['token_acc']:.4f} "
        f"test_concept_loss={test_metrics['concept_loss']:.4f} "
        f"test_concept_acc={test_metrics['concept_acc_overall']:.4f} "
        f"test_report_acc={test_report_acc:.4f}"
    )

    logger_obj.log(
        {
            "best_checkpoint/val_loss": float(best_val_metrics["loss"]),
            "best_checkpoint/val_gen_loss": float(best_val_metrics["gen_loss"]),
            "best_checkpoint/val_contrastive_loss": float(best_val_metrics["contrastive_loss"]),
            "best_checkpoint/val_contrastive_margin_loss": float(best_val_metrics["contrastive_margin_loss"]),
            "best_checkpoint/val_token_acc": float(best_val_metrics["token_acc"]),
            "best_checkpoint/val_concept_loss": float(best_val_metrics["concept_loss"]),
            "best_checkpoint/val_concept_acc_overall": float(best_val_metrics["concept_acc_overall"]),
            "best_checkpoint/val_report_word_acc": float(best_val_report_acc),
            "best_checkpoint/test_loss": float(test_metrics["loss"]),
            "best_checkpoint/test_gen_loss": float(test_metrics["gen_loss"]),
            "best_checkpoint/test_contrastive_loss": float(test_metrics["contrastive_loss"]),
            "best_checkpoint/test_contrastive_margin_loss": float(test_metrics["contrastive_margin_loss"]),
            "best_checkpoint/test_token_acc": float(test_metrics["token_acc"]),
            "best_checkpoint/test_concept_loss": float(test_metrics["concept_loss"]),
            "best_checkpoint/test_concept_acc_overall": float(test_metrics["concept_acc_overall"]),
            "best_checkpoint/test_report_word_acc": float(test_report_acc),
            **{f"best_checkpoint/val_{k}": float(v) for k, v in best_val_metrics.items() if k.startswith("concept_acc/")},
            **{f"best_checkpoint/val_{k}": float(v) for k, v in best_val_metrics.items() if k.startswith("concept_acc_region/")},
            **{f"best_checkpoint/val_{k}": float(v) for k, v in best_val_metrics.items() if k.startswith("concept_auroc/")},
            **{f"best_checkpoint/test_{k}": float(v) for k, v in test_metrics.items() if k.startswith("concept_acc/")},
            **{f"best_checkpoint/test_{k}": float(v) for k, v in test_metrics.items() if k.startswith("concept_acc_region/")},
            **{f"best_checkpoint/test_{k}": float(v) for k, v in test_metrics.items() if k.startswith("concept_auroc/")},
        },
        step=int(summary.final_step),
    )

    num_saved = save_first_n_test_generations_to_csv(
        model=best_model,
        tokenizer=best_tokenizer,
        prompt_builder=prompt_builder,
        loader=loaders["test"],
        device=device,
        autocast_dtype=autocast_dtype,
        max_prompt_tokens=config.max_prompt_tokens,
        gen_max_new_tokens=config.gen_max_new_tokens,
        output_csv_path=GENERATED_CSV_PATH,
        num_examples=100,
    )
    print(f"saved_test_generations={num_saved} csv_path={GENERATED_CSV_PATH}")

    logger_obj.finish()
    return None


if __name__ == "__main__":
    main()
