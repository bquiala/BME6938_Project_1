"""
data_loader.py
==============
Utilities for loading CKD datasets in ARFF (OpenML) format.

Supports both file paths (str / pathlib.Path) and file-like objects such as
Streamlit's UploadedFile, making it usable both from the CLI pipeline and the
web interface.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.io import arff

from src.config import DATA_DIR, get_logger

logger = get_logger(__name__)


def _strip_data_whitespace(content: str) -> str:
    """Normalise every data line in the @DATA section before parsing.

    Some ARFF files exported from OpenML contain embedded tab characters or
    single-quote wrappers (e.g. ``'\\tno'``, ``' yes'``) and values that span
    two physical lines.  scipy's ARFF parser does strict nominal-value
    matching and cannot handle any of these forms.

    This preprocessor:
    1. Joins lines that were wrapped mid-value (a ``'`` at end of a line
       followed by continuation on the next).
    2. Removes single-quote characters and embedded tab characters from
       each field.
    3. Strips leading/trailing whitespace from each field.
    """
    lines = content.splitlines()
    in_data = False
    result = []
    buf: str | None = None  # accumulates a wrapped data row
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "@data":
            in_data = True
            result.append(line)
            continue
        if not in_data:
            result.append(line)
            continue
        if not stripped or stripped.startswith("%"):
            result.append(line)
            continue
        # Accumulate wrapped lines (a data value split across lines)
        if buf is not None:
            buf = buf + " " + stripped
        else:
            buf = stripped
        # A complete data row has a fixed number of commas; keep joining until
        # single-quote count is even (all opened quotes are closed).
        if buf.count("'") % 2 != 0:
            continue  # still inside a quoted value that spans lines
        # Clean up each field: strip single-quotes, literal \t escape sequences,
        # actual tab characters, and surrounding whitespace
        fields = [
            f.replace("'", "").replace("\\t", "").replace("\t", "").strip()
            for f in buf.split(",")
        ]
        result.append(",".join(fields))
        buf = None
    if buf is not None:
        # Flush any remaining partial row
        fields = [
            f.replace("'", "").replace("\\t", "").replace("\t", "").strip()
            for f in buf.split(",")
        ]
        result.append(",".join(fields))
    return "\n".join(result)


def _convert_string_to_nominal(content: str) -> tuple[str, list[str]]:
    """Convert ``STRING``-typed attributes to nominal ``{v1,v2,...}`` declarations.

    scipy's ARFF parser raises ``NotImplementedError`` for ``@ATTRIBUTE ... STRING``
    declarations.  Dropping them loses data (including the target column when it is
    declared as STRING).  This function instead:

    1.  Identifies every ``@ATTRIBUTE <name> STRING`` line (case-insensitive,
        handles bare names, single-quoted names, and double-quoted names).
    2.  Scans the ``@DATA`` section to collect all unique non-missing values for
        each such column.
    3.  Rewrites the declaration as ``@ATTRIBUTE <name> {val1,val2,...}`` so that
        scipy treats it as a nominal attribute and preserves the column.

    If a STRING column contains only ``?`` (all missing), it is converted to a
    single-value nominal ``{?}`` so the file remains structurally valid.

    Parameters
    ----------
    content : str
        Pre-whitespace-stripped ARFF file text.

    Returns
    -------
    tuple[str, list[str]]
        Patched ARFF text and a list of STRING attribute names that were converted.
    """
    _attr_line_re   = re.compile(r"@attribute\b",              re.IGNORECASE)
    _string_type_re = re.compile(r"\bSTRING\b\s*(%.*)?$",      re.IGNORECASE)
    _attr_name_re   = re.compile(
        r"@attribute\s+(?:'(?P<sq>[^']*)'|\"(?P<dq>[^\"]*)\"|(?P<bare>\S+))",
        re.IGNORECASE,
    )

    lines = content.splitlines()

    # ── Pass 1: locate STRING attribute indices and original header lines ─────
    # string_attrs maps col_index -> (attr_name, original_line_index_in_lines)
    string_attrs: dict[int, tuple[str, int]] = {}
    attr_col_index = 0
    in_data = False
    data_start_line = None                     # line index where @DATA appears

    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower() == "@data":
            in_data = True
            data_start_line = line_idx
            break
        if _attr_line_re.match(stripped):
            if _string_type_re.search(stripped):
                nm = _attr_name_re.match(stripped)
                attr_name = (
                    nm.group("sq") or nm.group("dq") or nm.group("bare")
                    if nm else f"col_{attr_col_index}"
                )
                string_attrs[attr_col_index] = (attr_name, line_idx)
                logger.warning(
                    "STRING attribute '%s' (column %d) will be converted to nominal.",
                    attr_name, attr_col_index,
                )
            attr_col_index += 1

    if not string_attrs or data_start_line is None:
        return content, []

    # ── Pass 2: collect unique values per STRING column from the data section ─
    unique_vals: dict[int, set[str]] = {idx: set() for idx in string_attrs}

    for line in lines[data_start_line + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        fields = stripped.split(",")
        for col_idx in string_attrs:
            if col_idx < len(fields):
                val = fields[col_idx].strip()
                if val and val != "?":
                    unique_vals[col_idx].add(val)

    # ── Pass 3: rewrite the STRING header lines as nominal declarations ───────
    patched_lines = list(lines)
    converted_names: list[str] = []

    for col_idx, (attr_name, line_idx) in string_attrs.items():
        vals = sorted(unique_vals[col_idx])
        if not vals:
            vals = ["?"]                      # degenerate: all missing
        nominal_values = ",".join(vals)
        # Preserve original quoting style for the attribute name
        original = lines[line_idx]
        nm = _attr_name_re.match(original.strip())
        if nm and nm.group("sq") is not None:
            name_token = f"'{attr_name}'"
        elif nm and nm.group("dq") is not None:
            name_token = f'"{attr_name}"'
        else:
            name_token = attr_name
        patched_lines[line_idx] = f"@ATTRIBUTE {name_token} {{{nominal_values}}}"
        converted_names.append(attr_name)
        logger.info(
            "Converted STRING attribute '%s' → nominal with %d unique value(s): %s",
            attr_name, len(vals), nominal_values,
        )

    return "\n".join(patched_lines), converted_names


def load_arff(filepath: Union[str, Path, io.BytesIO]) -> pd.DataFrame:
    """
    Load an ARFF dataset and convert it to a pandas DataFrame.

    Byte-string columns produced by the scipy ARFF parser are automatically
    decoded to plain Python strings, and leading/trailing whitespace is
    stripped.

    Parameters
    ----------
    filepath : str | Path | file-like object
        Path to a ``.arff`` file on disk, or a file-like object (e.g. a
        Streamlit ``UploadedFile`` or ``io.BytesIO``).

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns intact.

    Raises
    ------
    FileNotFoundError
        If *filepath* is a path string/Path that does not exist.
    Exception
        Propagates any parsing errors from the scipy ARFF parser.
    """
    logger.info("Loading ARFF dataset from: %s", filepath)
    try:
        if isinstance(filepath, (str, Path)):
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"ARFF file not found: {path}")
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            # File-like object (Streamlit UploadedFile / BytesIO)
            content = filepath.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")

        # Strip whitespace around comma-separated values in data lines so that
        # nominal attributes like ' yes' match their declared values ('yes').
        content = _strip_data_whitespace(content)

        # Convert STRING-typed attributes to nominal so scipy can parse them
        # without losing any columns (including the target column).
        content, converted = _convert_string_to_nominal(content)
        if converted:
            logger.info(
                "Converted %d STRING attribute(s) to nominal: %s",
                len(converted), converted,
            )

        try:
            data, _meta = arff.loadarff(io.StringIO(content))
        except NotImplementedError as nie:
            # Last-resort: if scipy still raises NotImplementedError, attempt
            # conversion again (catches unusual quoting or relational types).
            if "string" not in str(nie).lower():
                raise
            logger.warning(
                "scipy raised NotImplementedError (%s) after conversion pass; "
                "retrying conversion.",
                nie,
            )
            content, extra = _convert_string_to_nominal(content)
            if not extra:
                raise
            data, _meta = arff.loadarff(io.StringIO(content))

        df = pd.DataFrame(data)

        # Decode byte strings emitted by the scipy ARFF parser
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8").strip() if isinstance(x, bytes) else x
            )

        logger.info("Dataset loaded successfully — shape: %s", df.shape)
        return df

    except Exception as exc:
        logger.error("Failed to load ARFF file: %s", exc)
        raise


def load_sample_data() -> pd.DataFrame:
    """
    Load the first ``.arff`` file found in the project ``data/`` directory.

    This is a convenience helper for running the pipeline without explicitly
    specifying a file path.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        When ``data/`` contains no ``.arff`` files.
    """
    arff_files = sorted(DATA_DIR.glob("*.arff"))
    if not arff_files:
        raise FileNotFoundError(
            f"No .arff files found in '{DATA_DIR}'.\n"
            "Please place your CKD dataset (e.g. chronic_kidney_disease.arff) "
            "in the data/ directory, or pass the path explicitly via --data."
        )
    logger.info("Found ARFF file: %s", arff_files[0])
    return load_arff(arff_files[0])
