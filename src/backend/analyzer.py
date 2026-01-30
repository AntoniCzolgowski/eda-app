"""
Data analysis logic - column detection and statistics computation
Enhanced with better date and ID detection
Updated: No row limit - returns full dataset
"""
import pandas as pd
import numpy as np
from typing import Any
from models import ColumnType, QualityFlag
import math
import re
from dateutil import parser as date_parser
from dateutil.parser import ParserError


def clean_value(val):
    """Clean value for JSON serialization - handle NaN, inf, etc."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 4)
    if isinstance(val, (np.integer, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float64)):
        if np.isnan(val) or np.isinf(val):
            return None
        return round(float(val), 4)
    return val


# ============================
# Date Detection
# ============================

# Common date patterns to try
DATE_PATTERNS = [
    r'^\d{4}-\d{2}-\d{2}$',           # YYYY-MM-DD
    r'^\d{2}-\d{2}-\d{4}$',           # DD-MM-YYYY or MM-DD-YYYY
    r'^\d{2}/\d{2}/\d{4}$',           # DD/MM/YYYY or MM/DD/YYYY
    r'^\d{4}/\d{2}/\d{2}$',           # YYYY/MM/DD
    r'^\d{2}\.\d{2}\.\d{4}$',         # DD.MM.YYYY
    r'^\d{8}$',                        # YYYYMMDD
    r'^\d{4}-\d{2}-\d{2}T',           # ISO datetime
    r'^\d{4}-\d{2}-\d{2} \d{2}:',     # YYYY-MM-DD HH:
    r'^[A-Za-z]{3}\s+\d{1,2},?\s+\d{4}$',  # Jan 15, 2024
    r'^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$',    # 15 Jan 2024
    r'^\d{1,2}/\d{1,2}/\d{2}$',       # M/D/YY or D/M/YY
]


def looks_like_date(value: str) -> bool:
    """Check if a string value looks like a date"""
    if not isinstance(value, str) or len(value) < 6:
        return False
    
    value = value.strip()
    
    # Check against common patterns
    for pattern in DATE_PATTERNS:
        if re.match(pattern, value):
            return True
    
    return False


def try_parse_dates(series: pd.Series, sample_size: int = 100) -> tuple[bool, pd.Series]:
    """
    Try to parse a series as dates.
    Returns (success, parsed_series)
    """
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return False, series
    
    # Sample for efficiency
    sample = non_null.head(sample_size)
    
    # Check if values look like dates
    date_like_count = sum(1 for v in sample.astype(str) if looks_like_date(str(v)))
    
    # If more than 70% look like dates, try parsing
    if date_like_count / len(sample) < 0.7:
        return False, series
    
    # Try pandas date parsing with various options
    try:
        # Try with infer_datetime_format
        parsed = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        
        # Check success rate
        success_rate = parsed.notna().sum() / non_null.notna().sum()
        
        if success_rate >= 0.8:
            return True, parsed
    except Exception:
        pass
    
    # Try dateutil parser on sample
    try:
        parsed_count = 0
        for val in sample.astype(str):
            try:
                date_parser.parse(str(val))
                parsed_count += 1
            except (ParserError, ValueError):
                pass
        
        if parsed_count / len(sample) >= 0.7:
            # Full parsing
            def safe_parse(x):
                try:
                    return date_parser.parse(str(x))
                except:
                    return pd.NaT
            
            parsed = series.apply(safe_parse)
            return True, pd.to_datetime(parsed, errors='coerce')
    except Exception:
        pass
    
    return False, series


# ============================
# ID Detection
# ============================

# Patterns that suggest ID columns
ID_NAME_PATTERNS = [
    r'(?i)^id$',
    r'(?i)_id$',
    r'(?i)^id_',
    r'(?i)_id_',
    r'(?i)identifier',
    r'(?i)^key$',
    r'(?i)_key$',
    r'(?i)^code$',
    r'(?i)_code$',
    r'(?i)^index$',
    r'(?i)^row',
    r'(?i)^record',
    r'(?i)serial',
    r'(?i)^no$',
    r'(?i)^num$',
    r'(?i)^number$',
]

# Patterns for alphanumeric IDs like ABC-001, ABC_002
ALPHANUMERIC_ID_PATTERN = r'^[A-Za-z]{1,5}[-_]?\d+$'


def is_id_column(series: pd.Series, column_name: str) -> bool:
    """
    Detect if a column is an identifier column.
    Checks:
    1. Column name contains 'id', 'key', 'code', etc.
    2. Values are unique or nearly unique
    3. Values are sequential integers or follow ID patterns
    """
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return False
    
    # Check column name
    name_matches_id = any(re.search(pattern, column_name) for pattern in ID_NAME_PATTERNS)
    
    # Check uniqueness (>95% unique)
    unique_ratio = non_null.nunique() / len(non_null)
    is_highly_unique = unique_ratio > 0.95
    
    # Check if numeric and sequential
    is_sequential = False
    if pd.api.types.is_numeric_dtype(non_null):
        try:
            sorted_vals = non_null.sort_values().reset_index(drop=True)
            diffs = sorted_vals.diff().dropna()
            # Check if mostly incrementing by 1
            if len(diffs) > 0:
                is_sequential = (diffs == 1).mean() > 0.9
        except Exception:
            pass
    
    # Check for alphanumeric ID patterns (ABC-001, etc.)
    is_alphanumeric_id = False
    sample = non_null.head(50).astype(str)
    alphanumeric_matches = sum(1 for v in sample if re.match(ALPHANUMERIC_ID_PATTERN, str(v)))
    if alphanumeric_matches / len(sample) > 0.8:
        is_alphanumeric_id = True
    
    # Decision: ID if name matches AND (unique OR sequential OR alphanumeric pattern)
    if name_matches_id and (is_highly_unique or is_sequential or is_alphanumeric_id):
        return True
    
    # Or if very unique + sequential (even without ID in name)
    if is_highly_unique and is_sequential and len(non_null) > 50:
        return True
    
    # Or if alphanumeric ID pattern regardless of name
    if is_alphanumeric_id and is_highly_unique:
        return True
    
    return False


# ============================
# Column Type Detection
# ============================

def detect_column_type(series: pd.Series, column_name: str) -> tuple[ColumnType, dict]:
    """
    Detect the type of a column based on its values.
    Returns (type, extra_info)
    """
    non_null = series.dropna()
    extra_info = {}
    
    if len(non_null) == 0:
        return ColumnType.UNKNOWN, extra_info
    
    # Check for ID first
    if is_id_column(series, column_name):
        return ColumnType.IDENTIFIER, extra_info
    
    # Check for datetime
    is_date, parsed_dates = try_parse_dates(series)
    if is_date:
        extra_info['parsed_dates'] = parsed_dates
        return ColumnType.DATETIME, extra_info
    
    # Check if numeric
    if pd.api.types.is_integer_dtype(series):
        # Check low cardinality (treat as categorical)
        if non_null.nunique() <= 5:
            return ColumnType.CATEGORICAL, extra_info
        return ColumnType.NUMERIC_INT, extra_info
    
    if pd.api.types.is_float_dtype(series):
        return ColumnType.NUMERIC_FLOAT, extra_info
    
    # Try to parse as numeric
    try:
        numeric = pd.to_numeric(non_null, errors='raise')
        if (numeric == numeric.astype(int)).all():
            if non_null.nunique() <= 5:
                return ColumnType.CATEGORICAL, extra_info
            return ColumnType.NUMERIC_INT, extra_info
        return ColumnType.NUMERIC_FLOAT, extra_info
    except (ValueError, TypeError):
        pass
    
    # String analysis
    unique_count = non_null.nunique()
    total_count = len(non_null)
    
    # High cardinality + looks like ID (already checked above, but double-check)
    if unique_count / total_count > 0.9 and unique_count > 100:
        return ColumnType.IDENTIFIER, extra_info
    
    # Categorical: less than 20 unique values or less than 5% of total
    if unique_count <= 20 or unique_count / total_count < 0.05:
        return ColumnType.CATEGORICAL, extra_info
    
    # Otherwise it's text
    return ColumnType.TEXT, extra_info


# ============================
# Statistics Computation
# ============================

def detect_quality_flags(df: pd.DataFrame, column: str) -> list[QualityFlag]:
    """Detect quality issues with a column"""
    flags = []
    series = df[column]
    
    if series.nunique() <= 1:
        flags.append(QualityFlag.SINGLE_VALUE)
    
    missing_pct = series.isna().sum() / len(series)
    if missing_pct > 0.5:
        flags.append(QualityFlag.HIGH_MISSING)
    
    # Don't flag as potential_id here - handled by type detection
    
    for other_col in df.columns:
        if other_col != column:
            if df[column].equals(df[other_col]):
                flags.append(QualityFlag.DUPLICATE_COLUMN)
                break
    
    return flags


def compute_numeric_stats(series: pd.Series) -> dict[str, Any]:
    """Compute statistics for numeric columns"""
    non_null = pd.to_numeric(series, errors='coerce').dropna()
    
    if len(non_null) == 0:
        return {}
    
    q1 = float(non_null.quantile(0.25))
    q3 = float(non_null.quantile(0.75))
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mode_result = non_null.mode()
    mode_val = float(mode_result.iloc[0]) if len(mode_result) > 0 else None
    
    return {
        "min": clean_value(non_null.min()),
        "max": clean_value(non_null.max()),
        "mean": clean_value(non_null.mean()),
        "median": clean_value(non_null.median()),
        "mode": clean_value(mode_val),
        "std": clean_value(non_null.std()),
        "q1": clean_value(q1),
        "q3": clean_value(q3),
        "iqr": clean_value(iqr),
        "outliers_low": int((non_null < lower_bound).sum()),
        "outliers_high": int((non_null > upper_bound).sum()),
    }


def compute_id_stats(series: pd.Series) -> dict[str, Any]:
    """Compute statistics for ID columns - only min, max, count"""
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return {}
    
    stats = {
        "unique_count": int(non_null.nunique()),
        "is_identifier": True,
        "message": "This column appears to be an identifier/key column"
    }
    
    # Try to get min/max for numeric IDs
    if pd.api.types.is_numeric_dtype(non_null):
        stats["min"] = clean_value(non_null.min())
        stats["max"] = clean_value(non_null.max())
    else:
        # For string IDs, show sample
        stats["sample_values"] = non_null.head(3).tolist()
    
    return stats


def compute_datetime_stats(series: pd.Series, parsed_dates: pd.Series = None) -> dict[str, Any]:
    """Compute statistics for datetime columns - no mean/median/std"""
    try:
        if parsed_dates is not None:
            dates = parsed_dates.dropna()
        else:
            dates = pd.to_datetime(series, errors='coerce').dropna()
        
        if len(dates) == 0:
            return {}
        
        min_date = dates.min()
        max_date = dates.max()
        date_range = (max_date - min_date).days
        
        return {
            "min_date": str(min_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(min_date, 'strftime') else min_date),
            "max_date": str(max_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(max_date, 'strftime') else max_date),
            "date_range_days": int(date_range),
            "unique_count": int(dates.nunique()),
            "is_datetime": True
        }
    except Exception as e:
        print(f"Datetime stats error: {e}")
        return {}


def compute_categorical_stats(series: pd.Series) -> dict[str, Any]:
    """Compute statistics for categorical columns"""
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return {"unique_count": 0, "top_values": []}
    
    value_counts = non_null.value_counts()
    total = len(non_null)
    
    top_values = [
        {
            "value": str(val),
            "count": int(count),
            "percent": round(count / total * 100, 2)
        }
        for val, count in value_counts.head(10).items()
    ]
    
    return {
        "unique_count": int(non_null.nunique()),
        "top_values": top_values
    }


def compute_text_stats(series: pd.Series, column_name: str) -> dict[str, Any]:
    """Compute statistics for text columns - collect samples for LLM summary"""
    non_null = series.dropna().astype(str)
    
    if len(non_null) == 0:
        return {"unique_count": 0, "avg_length": 0, "sample_values": []}
    
    # Get diverse sample values for LLM analysis
    sample_size = min(15, len(non_null))
    if len(non_null) <= sample_size:
        sample_values = non_null.tolist()
    else:
        sample_values = non_null.sample(sample_size, random_state=42).tolist()
    
    return {
        "unique_count": int(non_null.nunique()),
        "avg_length": clean_value(non_null.str.len().mean()),
        "sample_values": sample_values
    }


def analyze_column(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Complete analysis of a single column"""
    series = df[column]
    col_type, extra_info = detect_column_type(series, column)
    quality_flags = detect_quality_flags(df, column)
    
    # Compute type-specific stats
    if col_type == ColumnType.IDENTIFIER:
        stats = compute_id_stats(series)
        quality_flags.append(QualityFlag.POTENTIAL_ID)
    elif col_type == ColumnType.DATETIME:
        parsed_dates = extra_info.get('parsed_dates')
        stats = compute_datetime_stats(series, parsed_dates)
    elif col_type in [ColumnType.NUMERIC_INT, ColumnType.NUMERIC_FLOAT]:
        stats = compute_numeric_stats(series)
    elif col_type == ColumnType.CATEGORICAL:
        stats = compute_categorical_stats(series)
    elif col_type == ColumnType.TEXT:
        stats = compute_text_stats(series, column)
    else:
        stats = {"unique_count": int(series.nunique())}
    
    return {
        "name": column,
        "dtype": col_type.value if hasattr(col_type, 'value') else str(col_type),
        "missing": int(series.isna().sum()),
        "missing_pct": round(series.isna().sum() / len(series) * 100, 2),
        "quality_flags": [f.value if hasattr(f, 'value') else str(f) for f in quality_flags],
        "stats": stats
    }


def clean_preview_data(data: list) -> list:
    """Clean preview data for JSON serialization"""
    cleaned = []
    for row in data:
        cleaned_row = []
        for val in row:
            if pd.isna(val):
                cleaned_row.append(None)
            elif isinstance(val, (np.floating, float)):
                if np.isnan(val) or np.isinf(val):
                    cleaned_row.append(None)
                else:
                    cleaned_row.append(round(float(val), 4))
            elif isinstance(val, (np.integer,)):
                cleaned_row.append(int(val))
            elif isinstance(val, (pd.Timestamp, np.datetime64)):
                cleaned_row.append(str(val))
            else:
                cleaned_row.append(val)
        cleaned.append(cleaned_row)
    return cleaned


def analyze_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Complete analysis of the entire dataset"""
    columns_analysis = []
    
    for column in df.columns:
        col_analysis = analyze_column(df, column)
        columns_analysis.append(col_analysis)
    
    # Return ALL data, no limit
    preview_data = clean_preview_data(df.values.tolist())
    
    return {
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "columns": columns_analysis,
        "data_preview": preview_data
    }