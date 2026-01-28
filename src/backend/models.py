"""
Pydantic models and enums for EDA application
"""
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


class ColumnType(str, Enum):
    """Types of columns detected in a dataset"""
    NUMERIC_INT = "numeric_int"
    NUMERIC_FLOAT = "numeric_float"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    IDENTIFIER = "identifier"
    UNKNOWN = "unknown"


class QualityFlag(str, Enum):
    """Quality issues that can be detected in columns"""
    DUPLICATE_COLUMN = "duplicate_column"
    SINGLE_VALUE = "single_value"
    HIGH_MISSING = "high_missing"
    POTENTIAL_ID = "potential_id"


class JobStatus(str, Enum):
    """Status of an analysis job"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisJob:
    """Tracks the state of an analysis job"""
    
    def __init__(
        self,
        job_id: str,
        filename: str,
        status: JobStatus = JobStatus.PENDING,
        progress: int = 0,
        error: Optional[str] = None,
        results: Optional[dict] = None
    ):
        self.job_id = job_id
        self.filename = filename
        self.status = status
        self.progress = progress
        self.error = error
        self.results = results


# Pydantic models for API requests/responses

class UploadResponse(BaseModel):
    job_id: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int
    error: Optional[str] = None


class PlotSuggestionRequest(BaseModel):
    job_id: str
    columns: list[str]


class PlotSuggestionResponse(BaseModel):
    plot_type: str
    x_column: str
    y_column: Optional[str] = None
    reasoning: str


class GeneratePlotRequest(BaseModel):
    job_id: str
    plot_type: str
    x_column: str
    y_column: Optional[str] = None
    options: Optional[dict] = None