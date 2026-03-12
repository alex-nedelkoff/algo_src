"""loader.py — tbparse wrapper for loading TensorBoard event files."""

from __future__ import annotations

import pandas as pd
from tbparse import SummaryReader


def load_run(path: str, last_n: int | None = None) -> pd.DataFrame:
    """Load scalar metrics from a TensorBoard run directory.

    Uses tbparse.SummaryReader to recursively load all event files found under
    *path* (SB3 creates a new event file on each resume, so multiple files are
    expected). Returns a DataFrame with columns: step, tag, value.

    Args:
        path: Path to a TensorBoard run directory (or a single event file).
        last_n: When provided, filter the DataFrame to rows where
            ``step >= max_step - last_n``.  Applied immediately after loading
            to keep memory usage bounded.

    Returns:
        DataFrame with columns ``step`` (int), ``tag`` (str), ``value`` (float).
        Returns an empty DataFrame with those columns if no scalar data is found.
    """
    try:
        reader = SummaryReader(path, pivot=False)
        df: pd.DataFrame = reader.scalars
    except ValueError:
        return pd.DataFrame(columns=["step", "tag", "value"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["step", "tag", "value"])

    # Normalise to the expected column schema.  tbparse may emit extra columns
    # (e.g. "dir_name", "file_name") — keep only what downstream consumers need.
    df = df[["step", "tag", "value"]].reset_index(drop=True)

    if last_n is not None:
        max_step = int(df["step"].max())
        df = df[df["step"] >= max_step - last_n].reset_index(drop=True)

    return df
