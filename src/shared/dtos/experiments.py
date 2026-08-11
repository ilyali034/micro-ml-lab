
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field

class CreateExperimentDto(BaseModel): 
    model_id: UUID
    run_name: str 
    params: str 
    metrics: str 

class ExperimentDto(BaseModel): 
    id: UUID
    model_id: UUID
    run_name: str 
    params: str 
    metrics: str 

class PaginationExperimentsDto(BaseModel):
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=100, description="Items per page")
    model_id: Optional[UUID] = Field(default=None, description="Filter by model ID")
    

class PaginatedResponseDto(BaseModel):
    items: List[ExperimentDto]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool