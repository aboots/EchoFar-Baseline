from s3torchconnector.s3map_dataset import S3MapDataset, get_objects_from_uris, identity
from scipy.io import loadmat
from rclstream.config import create_mysql_engine
import pandas as pd
from io import BytesIO
from functools import partial
import numpy as np
from typing import Any, Dict, List, Optional
import pydicom


def get_metadata_columns() -> List[str]:
    """Gets the columns of the `cardiac_file` table in the `echo_inventory` database.

    Returns:
        A list of column names for the `cardiac_file` table.
    """
    engine = create_mysql_engine(database="echo_inventory")
    query = (
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_NAME = 'cardiac_file';"
    )
    column_frame = pd.read_sql(query, engine)
    return column_frame["COLUMN_NAME"].tolist()


def get_patient_metadata_columns() -> List[str]:
    """Gets the columns of the `cardiac_exam` table in the `echo_inventory` database.

    Returns:
        A list of column names for the `cardiac_exam` table.
    """
    engine = create_mysql_engine(database="echo_inventory")
    query = (
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_NAME = 'cardiac_exam';"
    )
    column_frame = pd.read_sql(query, engine)
    return column_frame["COLUMN_NAME"].tolist()


def get_metadata(
    columns: Optional[List[str]] = None,
    exam_id_max_exclusive: Optional[int] = None,
) -> pd.DataFrame:
    """Fetches metadata columns from the `cardiac_file` table in the `echo_inventory` database.

    Args:
        columns: List of column names to retrieve in addition to 'stream_id'.
        exam_id_max_exclusive: If provided, only rows with `exam_id < exam_id_max_exclusive`
            are returned.

    Returns:
        A DataFrame indexed by 'stream_id', containing the specified metadata columns.
    """
    engine = create_mysql_engine(database="echo_inventory")
    selected_columns = ["stream_id"] + (columns or [])
    columns_str = ", ".join(selected_columns)

    where_clauses = ["stream_id IS NOT NULL"]
    if exam_id_max_exclusive is not None:
        where_clauses.append(f"exam_id < {int(exam_id_max_exclusive)}")

    where_str = " AND ".join(where_clauses)
    query = f"SELECT {columns_str} FROM cardiac_file WHERE {where_str};"

    metadata_frame = pd.read_sql(query, engine, index_col="stream_id")
    metadata_frame.index.name = "stream_id"
    metadata_frame.index = metadata_frame.index.astype(int)
    metadata_frame.sort_index(inplace=True)
    return metadata_frame


def get_raw_metadata(
    columns: Optional[List[str]] = None,
    exam_id_max_exclusive: Optional[int] = None,
) -> pd.DataFrame:
    """Fetches raw metadata from the `cardiac_file` table in the `echo_inventory` database.

    Args:
        columns: List of column names to retrieve in addition to 'id' and 'stream_id'.
        exam_id_max_exclusive: If provided, only rows with `exam_id < exam_id_max_exclusive`
            are returned.

    Returns:
        A DataFrame indexed by 'id', containing the specified raw metadata columns.
    """
    engine = create_mysql_engine(database="echo_inventory")
    selected_columns = ["id", "stream_id"] + (columns or [])
    columns_str = ", ".join(selected_columns)

    where_str = ""
    if exam_id_max_exclusive is not None:
        where_str = f" WHERE exam_id < {int(exam_id_max_exclusive)}"

    query = f"SELECT {columns_str} FROM cardiac_file{where_str};"

    metadata_frame = pd.read_sql(query, engine, index_col="id")
    metadata_frame.index.name = "raw_id"
    metadata_frame.sort_index(inplace=True)
    return metadata_frame


def get_patient_metadata(
    columns: Optional[List[str]] = None,
    exam_id_max_exclusive: Optional[int] = None,
) -> pd.DataFrame:
    """Fetches patient metadata from the `cardiac_exam` table in the `echo_inventory` database.

    Args:
        columns: List of column names to retrieve in addition to 'id'.
        exam_id_max_exclusive: If provided, only rows with `id < exam_id_max_exclusive`
            are returned.

    Returns:
        A DataFrame indexed by 'id' (renamed to 'exam_id'), containing the specified
        patient metadata columns.
    """
    engine = create_mysql_engine(database="echo_inventory")
    selected_columns = ["id"] + (columns or [])
    columns_str = ", ".join(selected_columns)

    where_str = ""
    if exam_id_max_exclusive is not None:
        where_str = f" WHERE id < {int(exam_id_max_exclusive)}"

    query = f"SELECT {columns_str} FROM cardiac_exam{where_str};"

    patient_frame = pd.read_sql(query, engine, index_col="id")
    patient_frame.index.name = "exam_id"
    patient_frame.sort_index(inplace=True)
    return patient_frame


class EchoDataset(S3MapDataset):
    """Dataset for accessing and processing echocardiogram files stored in S3."""

    def __init__(self):
        """Initializes the EchoDataset.

        Args:
            exam_id_max_exclusive: If provided, only files with `exam_id < exam_id_max_exclusive`
                are included (reduces metadata load and speeds up initialization).
        """
        exam_id_max_exclusive = 15000
        metadata = get_metadata(
            ["processed_file_address", "exam_id"],
            exam_id_max_exclusive=exam_id_max_exclusive,
        )
        self.metadata = metadata.drop(columns=["processed_file_address"])
        self.object_uris = [
            f"s3://ra-puranga-1/{uri.lstrip('/')}"
            for uri in metadata.processed_file_address.tolist()
        ]
        self.stream_ids = metadata.index.astype(int).tolist()
        self.stream_id_to_dataset_index = {
            stream_id: dataset_index
            for dataset_index, stream_id in enumerate(self.stream_ids)
        }

        super().__init__(
            "us-east-1",
            partial(get_objects_from_uris, self.object_uris),
            endpoint="https://chinook.arc.ubc.ca",
            transform=identity,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieves and processes an item from the dataset by positional index.

        Args:
            idx: Positional dataset index.

        Returns:
            A dict containing the loaded and processed .mat file data.
        """
        item = super().__getitem__(idx)
        file_bytes = BytesIO(item.read())
        mat_data = loadmat(file_bytes, simplify_cells=True)

        mat_data["stream_id"] = int(self.stream_ids[idx])
        mat_data["processed_filepath"] = self.object_uris[idx]
        mat_data["raw_filepath"] = mat_data["processed_filepath"].replace(
            "processed", "raw"
        )
        mat_data["video"] = mat_data["cropped"].astype(np.uint8).transpose(2, 0, 1)

        del (
            mat_data["cropped"],
            mat_data["__header__"],
            mat_data["__version__"],
            mat_data["__globals__"],
        )

        return mat_data


class EchoRawDataset(S3MapDataset):
    """Dataset for accessing and processing raw echocardiogram DICOM files stored in S3."""

    def __init__(self):
        """Initializes the EchoRawDataset.

        Args:
            exam_id_max_exclusive: If provided, only files with `exam_id < exam_id_max_exclusive`
                are included.
        """
        exam_id_max_exclusive = 15000
        metadata = get_raw_metadata(
            columns=["raw_file_address", "exam_id"],
            exam_id_max_exclusive=exam_id_max_exclusive,
        )
        self.metadata = metadata.drop(columns=["raw_file_address"])
        self.object_uris = [
            f"s3://ra-puranga-1/{uri[1:]}" for uri in metadata.raw_file_address.tolist()
        ]
        super().__init__(
            "us-east-1",
            partial(get_objects_from_uris, self.object_uris),
            endpoint="https://chinook.arc.ubc.ca",
            transform=identity,
        )

    def __getitem__(self, idx: int) -> pydicom.dataset.FileDataset:
        """Retrieves and decodes a raw DICOM file from S3 by index.

        Args:
            idx: Positional dataset index.

        Returns:
            The decoded DICOM file as a pydicom object.
        """
        item = super().__getitem__(idx)
        file_bytes = BytesIO(item.read())
        return pydicom.dcmread(file_bytes)


class EchoPatientDataset(EchoDataset):
    """Patient-level dataset (grouped by exam_id).

    This dataset only loads exams with `exam_id < exam_id_max_exclusive` to reduce processing time.
    """

    def __init__(self):
        exam_id_max_exclusive = 15000
        super().__init__()

        patient_metadata = get_patient_metadata(
            columns=[
                "report_page1",
                "report_page2",
            ],
            exam_id_max_exclusive=exam_id_max_exclusive,
        )

        self.patient_metadata = (
            pd.merge(
                patient_metadata,
                self.metadata.reset_index(),
                on="exam_id",
                how="left",
            )
            .sort_values(by=["exam_id", "stream_id"])
            .dropna(subset="stream_id")
            .astype({"stream_id": int})
            .groupby("exam_id")
            .agg(
                {
                    "stream_id": list,
                    "report_page1": "first",
                    "report_page2": "first",
                }
            )
            .reset_index()
        )

    def __len__(self) -> int:
        return len(self.patient_metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieves a patient (exam) sample by patient-level index."""
        sample: Dict[str, Any] = {}
        sample.update(self.patient_metadata.iloc[idx])

        videos = []
        for stream_id in sample["stream_id"]:
            if stream_id> 8256628:
                break
            dataset_index = self.stream_id_to_dataset_index[int(stream_id)]
            videos.append(super().__getitem__(dataset_index)["video"])

        sample["videos"] = videos
        return sample
