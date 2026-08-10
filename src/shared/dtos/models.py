from pydantic import BaseModel 
from uuid import UUID 

class ModelDto(BaseModel): 
    id: UUID
    name: str 
    model_file_path: str 

class CreateModelDto(BaseModel): 
    name: str 
    model_file_path: str