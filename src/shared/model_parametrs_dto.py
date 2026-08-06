
from typing import Any

from pydantic import BaseModel 

class ModelParamsDto(BaseModel): 
    total_parameters: Any
    trainable_parameters: Any
    non_trainable_parameters: Any
    model_type: str
    model_name: str

class ModelParamsInputDto(BaseModel): 
    model_name: str 
    model: Any 