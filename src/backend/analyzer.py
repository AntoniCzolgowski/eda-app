"""
Data analysis logic - column detection and statistics computation
"""
import pandas as pd
import numpy as np
from typing import Any
from models import ColumnType, QualityFlag
import math


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


def detect_column_type(series: pd.Series) -> ColumnType:
    """Detect the type of a column based on its values"""
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return ColumnType.UNKNOWN
    
    # Check if numeric
    if pd.api.types.is_integer_dtype(series):
        return ColumnType.NUMERIC_INT
    
    if pd.api.types.is_float_dtype(series):
        return ColumnType.NUMERIC_FLOAT
    
    # Try to parse as numeric
    try:
        numeric = pd.to_numeric(non_null, errors='raise')
        if (numeric == numeric.astype(int)).all():
            return ColumnType.NUMERIC_INT
        return ColumnType.NUMERIC_FLOAT
    except (ValueError, TypeError):
        pass
    
    # Try to parse as datetime
    try:
        pd.to_datetime(non_null, errors='raise')
        return ColumnType.DATETIME
    except (ValueError, TypeError):
        pass
    
    # String analysis
    unique_count = non_null.nunique()
    total_count = len(non_null)
    
    # High cardinality + looks like ID
    if unique_count / total_count > 0.9 and unique_count > 100:
        return ColumnType.IDENTIFIER
    
    # Categorical: less than 20 unique values or less than 5% of total
    if unique_count <= 20 or unique_count / total_count < 0.05:
        return ColumnType.CATEGORICAL
    
    # Otherwise it's text
    return ColumnType.TEXT


def detect_quality_flags(df: pd.DataFrame, column: str) -> list[QualityFlag]:
    """Detect quality issues with a column"""
    flags = []
    series = df[column]
    
    # Single value check
    if series.nunique() <= 1:
        flags.append(QualityFlag.SINGLE_VALUE)
    
    # High missing check (>50%)
    missing_pct = series.isna().sum() / len(series)
    if missing_pct > 0.5:
        flags.append(QualityFlag.HIGH_MISSING)
    
    # Potential ID check
    non_null = series.dropna()
    if len(non_null) > 0:
        unique_ratio = non_null.nunique() / len(non_null)
        if unique_ratio > 0.95 and len(non_null) > 50:
            flags.append(QualityFlag.POTENTIAL_ID)
    
    # Duplicate column check
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
    
    # Outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Mode - handle multiple modes
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


def compute_text_stats(series: pd.Series) -> dict[str, Any]:
    """Compute statistics for text columns"""
    non_null = series.dropna().astype(str)
    
    if len(non_null) == 0:
        return {"unique_count": 0, "avg_length": 0, "sample_values": []}
    
    return {
        "unique_count": int(non_null.nunique()),
        "avg_length": clean_value(non_null.str.len().mean()),
        "sample_values": non_null.head(3).tolist()
    }


def compute_datetime_stats(series: pd.Series) -> dict[str, Any]:
    """Compute statistics for datetime columns"""
    try:
        dates = pd.to_datetime(series, errors='coerce').dropna()
        
        if len(dates) == 0:
            return {}
        
        return {
            "min_date": str(dates.min()),
            "max_date": str(dates.max()),
            "date_range_days": int((dates.max() - dates.min()).days),
            "unique_count": int(dates.nunique())
        }
    except Exception:
        return {}


def analyze_column(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Complete analysis of a single column"""
    series = df[column]
    col_type = detect_column_type(series)
    quality_flags = detect_quality_flags(df, column)
    
    # Compute type-specific stats
    if col_type in [ColumnType.NUMERIC_INT, ColumnType.NUMERIC_FLOAT]:
        stats = compute_numeric_stats(series)
    elif col_type == ColumnType.CATEGORICAL:
        stats = compute_categorical_stats(series)
    elif col_type == ColumnType.TEXT:
        stats = compute_text_stats(series)
    elif col_type == ColumnType.DATETIME:
        stats = compute_datetime_stats(series)
    else:
        stats = {"unique_count": int(series.nunique())}
    
    return {
        "name": column,
        "dtype": col_type,
        "missing": int(series.isna().sum()),
        "missing_pct": round(series.isna().sum() / len(series) * 100, 2),
        "quality_flags": quality_flags,
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
    
    # Clean preview data
    preview_data = clean_preview_data(df.head(100).values.tolist())
    
    return {
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "columns": columns_analysis,
        "data_preview": preview_data
    }