from uuid import UUID

from .database.repositories.protocols import ModelRepository
from .exceptions.models import ModelAlreadyExistsError, ModelNotFoundError

import logging

class ModelService(object):
    def __init__(
        self,
        model_repository: ModelRepository
    ):
        self.model_repository = model_repository
        self.logger = logging.getLogger(__name__)

    async def add_model_async(
        self,
        model_name: str,
        model: str
    ):
        self.logger.info("Adding model '%s' (file_path=%s)", model_name, model)

        existing = await self.model_repository.get_model_by_name(name=model_name)
        if existing is not None:
            self.logger.warning("Model '%s' already exists (id=%s)", model_name, existing["id"])
            raise ModelAlreadyExistsError(f"Model '{model_name}' already exists (id={existing['id']})")

        model_id = await self.model_repository.create_model(name=model_name, model_file_path=model)
        self.logger.info("Model '%s' created successfully (id=%s)", model_name, model_id)
        return model_id

    async def delete_model_async(
        self,
        model_id: UUID
    ) -> bool:
        self.logger.info("Deleting model id=%s", model_id)

        existing = await self.model_repository.get_model_by_id(model_id=model_id)
        if existing is None:
            self.logger.warning("Model id=%s not found, cannot delete", model_id)
            raise ModelNotFoundError(f"Model {model_id} not found")

        result = await self.model_repository.delete_model(model_id)
        if result:
            self.logger.info("Model id=%s deleted successfully", model_id)
        else:
            self.logger.error("Model id=%s existed but delete_model returned False", model_id)
        return result

    async def get_model_by_id_async(
        self, 
        model_id: UUID
    ): 

        return await self.model_repository.get_model_by_id(model_id) 


    async def get_model_by_name_async(
        self,
        model_name: str 
    ): 
        return await self.model_repository.get_model_by_name(model_name) 
    

    