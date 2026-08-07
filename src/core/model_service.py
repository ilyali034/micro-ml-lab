from .database.repositories.protocols import ModelRepository
from .exceptions.models import ModelAlreadyExistsError, ModelNotFoundError

class ModelService(object): 
    def __init__(
        self, 
        model_repository: ModelRepository
    ):
        self.model_repository = model_repository

    async def add_model(
        self,
        model_name: str,
        model: str
    ):
        existing = await self.model_repository.get_model_by_name(name=model_name) 
        if existing is not None: 
            raise ModelAlreadyExistsError(f"Model '{model_name}' already exists (id={existing['id']})") 
        return await self.model_repository.create_model(name=model_name, model_file_path=model) 

    async def delete_model(
        self, 
        model_id: int
    ) -> bool:
        existing = await self.model_repository.get_model_by_id(model_id=model_id)
        if existing is None:
            raise ModelNotFoundError(f"Model {model_id} not found")

        return await self.model_repository.delete_model(model_id) 

    