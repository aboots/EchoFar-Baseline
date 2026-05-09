import re

def normalize_vlm_prediction(text: str) -> str:
    # --- Format 1: Section name / Content ---
    pattern1 = r"Section name:\s*(.*?)\n[Cc]ontent:\s*"
    text = re.sub(pattern1, r"\1: ", text)
    
    # --- Format 2: Markdown with numbered/bold sections ---
    # Strip preamble (### ...)
    text = re.sub(r"(?m)^###.*?\n", "", text)
    # Strip patient information section completely
    text = re.sub(r"\*\*Patient Information:\*\*.*?(?=\*\*\d+\.|\*\*Left|$)", "", text, flags=re.DOTALL)
    
    # Handle bold headers (numbered or not): **1. Left Ventricle:** or **Left Ventricle:**
    # We look for a line starting with bold header
    text = re.sub(r"(?m)^\s*\*\*(?:\d+[\.]?\s*)?(.*?):\*\*\s*", r"\1: ", text)
    
    # Handle sub-bullets headers: "- **Size:**" -> " " (strip header)
    text = re.sub(r"-\s*\*\*(.*?):\*\*\s*", " ", text)
    
    # Remove any remaining bolding stars
    text = text.replace("**", "")
    
    # Remove separators
    text = text.replace("---", "")
    
    # Remove patient info placeholders
    text = re.sub(r"\[.*?\]", "", text)
    
    # Clean up whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    
    return text.strip()

test_inputs = [
    """### Echocardiography Findings Report

**Patient Information:**
[Patient's Name, Age, Gender, Date of Examination, etc.]

---

**1. Left Ventricle:**
- **Size:** Normal, based on indexed linear dimension.
- **Systolic Function:** Normal, with an estimated ejection fraction of 60%.""",
    """ **Left Ventricle:**
The left ventricle appears to have normal wall thickness and contractility, with no signs of dilation or hypokinesis."""
]

for i, inp in enumerate(test_inputs):
    print(f"\n--- Test {i} ---")
    print("Original:")
    print(inp)
    print("\nNormalized:")
    print(normalize_vlm_prediction(inp))
