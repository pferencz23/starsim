import pandas as pd
import numpy as np

def check_contact_time_resolution(df: pd.DataFrame, time_col: str = "time") -> dict:
    """
    Inspect the resolution of a contact time column.

    Works with either:
      - raw numeric Unix timestamps
      - datetime64 columns

    Returns a report with:
      - number of unique times
      - smallest positive gap between consecutive unique timestamps
      - a human-readable guess for the resolution
      - how many timestamps fall exactly on day / hour / minute / second boundaries
    """
    s = df[time_col].dropna()

    if s.empty:
        return {"error": f"Column '{time_col}' has no non-null values."}

    # Normalize to a comparable numeric representation
    if pd.api.types.is_datetime64_any_dtype(s):
        x = s.sort_values().drop_duplicates().astype("int64")  # nanoseconds
        unit_name = "ns"
    else:
        # Assume numeric or coercible to numeric
        x = pd.to_numeric(s, errors="coerce").dropna().sort_values().drop_duplicates()
        if x.empty:
            return {"error": f"Column '{time_col}' could not be interpreted as numeric or datetime."}
        unit_name = "raw units"

    # Compute positive gaps between unique timestamps
    diffs = np.diff(x.to_numpy())
    positive_diffs = diffs[diffs > 0]

    if len(positive_diffs) == 0:
        smallest_gap = None
    else:
        smallest_gap = positive_diffs.min()

    # Helper to format a gap if datetime-based
    def _format_gap_ns(ns: int) -> str:
        td = pd.to_timedelta(ns, unit="ns")
        return str(td)

    # Guess resolution
    resolution_guess = "unknown"
    if smallest_gap is not None:
        if unit_name == "ns":
            if smallest_gap % 1_000_000_000 == 0:
                sec = smallest_gap // 1_000_000_000
                if sec % 86400 == 0:
                    resolution_guess = f"{sec // 86400} day(s)"
                elif sec % 3600 == 0:
                    resolution_guess = f"{sec // 3600} hour(s)"
                elif sec % 60 == 0:
                    resolution_guess = f"{sec // 60} minute(s)"
                else:
                    resolution_guess = f"{sec} second(s)"
            else:
                resolution_guess = f"{smallest_gap} nanosecond(s)"
        else:
            resolution_guess = f"{smallest_gap} {unit_name}"

    report = {
        "n_rows": int(len(s)),
        "n_unique_times": int(x.shape[0]),
        "smallest_positive_gap": int(smallest_gap) if smallest_gap is not None else None,
        "smallest_positive_gap_human": _format_gap_ns(int(smallest_gap)) if (smallest_gap is not None and unit_name == "ns") else None,
        "resolution_guess": resolution_guess,
        "time_dtype": str(df[time_col].dtype),
    }

    # Extra boundary checks if the data are datetime-like or can be parsed as such
    dt = None
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = s
    else:
        # Try a best-effort parse for human-friendly boundary checks
        try:
            dt = pd.to_datetime(s, errors="coerce", utc=True)
            if dt.notna().sum() == 0:
                dt = None
        except Exception:
            dt = None

    if dt is not None:
        dt = dt.dropna()
        report.update({
            "pct_on_day_boundary": float((dt.dt.hour.eq(0) & dt.dt.minute.eq(0) & dt.dt.second.eq(0) & dt.dt.microsecond.eq(0)).mean()),
            "pct_on_hour_boundary": float((dt.dt.minute.eq(0) & dt.dt.second.eq(0) & dt.dt.microsecond.eq(0)).mean()),
            "pct_on_minute_boundary": float((dt.dt.second.eq(0) & dt.dt.microsecond.eq(0)).mean()),
            "pct_on_second_boundary": float((dt.dt.microsecond.eq(0)).mean()),
        })

    return report