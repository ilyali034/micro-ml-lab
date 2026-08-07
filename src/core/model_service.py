from .database.repositories.protocols import ModelRepository
from .exceptions.models import ModelAlreadyExistsError, ModelNotFoundError

import logging

logger = logging.getLogger(__name__)


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
        logger.info("Adding model '%s' (file_path=%s)", model_name, model)

        existing = await self.model_repository.get_model_by_name(name=model_name)
        if existing is not None:
            logger.warning("Model '%s' already exists (id=%s)", model_name, existing["id"])
            raise ModelAlreadyExistsError(f"Model '{model_name}' already exists (id={existing['id']})")

        model_id = await self.model_repository.create_model(name=model_name, model_file_path=model)
        logger.info("Model '%s' created successfully (id=%s)", model_name, model_id)
        return model_id

    async def delete_model(
        self,
        model_id: int
    ) -> bool:
        logger.info("Deleting model id=%s", model_id)

        existing = await self.model_repository.get_model_by_id(model_id=model_id)
        if existing is None:
            logger.warning("Model id=%s not found, cannot delete", model_id)
            raise ModelNotFoundError(f"Model {model_id} not found")

        result = await self.model_repository.delete_model(model_id)
        if result:
            logger.info("Model id=%s deleted successfully", model_id)
        else:
            logger.error("Model id=%s existed but delete_model returned False", model_id)
        return result