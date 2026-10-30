from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, status
from typing import Annotated
import logging

from shared.dtos.models import ModelDto, CreateModelDto
from core.model_service import ModelService
from core.database.repositories.protocols import ModelRepository

async def get_model_repository(request: Request) -> ModelRepository:
    db_context = getattr(request.app.state, "db_context", None)
    if db_context is None:
        raise RuntimeError("Database not initialized")
    return db_context.models


router = APIRouter(prefix="/models", tags=["models"])
logger = logging.getLogger(__name__)

async def get_model_service(
    repo: Annotated[ModelRepository, Depends(get_model_repository)]
) -> ModelService:
    return ModelService(model_repository=repo)

@router.post(
    "/add",
    response_model=dict[str, UUID],
    status_code=status.HTTP_201_CREATED
)
async def add_model_route_async(
    model: CreateModelDto,
    model_service: Annotated[ModelService, Depends(get_model_service)]
):
    logger.info(f"Received request to create model: {model.name}")
    
    try: 
        result = await model_service.add_model_async(model.name, model.model_file_path)
        logger.info(f"Successfully created model with ID: {result}")
        return {"id": result}  
    
    except Exception as e: 
        logger.error(f"Failed to create model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while creating the model"
        )

@router.delete(
    "/delete/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_model_route_async(
    model_id: UUID,
    model_service: Annotated[ModelService, Depends(get_model_service)]
):
    try: 
        result = await model_service.delete_model_async(model_id) 
        if result: 
            logger.info(f"Successfully deleted model with id {model_id}")
            return 
    except HTTPException as e: 
        raise 
    except Exception as e: 
        logger.error(f"Failed to delete model with id: {model_id} error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while deleting the model"
        )

@router.get(
    "/get/{model_id}",
    response_model=ModelDto
)
async def get_model_by_id_route_async(
    model_id: UUID,
    model_service: Annotated[ModelService, Depends(get_model_service)] 
) -> ModelDto:
    try: 
        result = await model_service.get_model_by_id_async(model_id) 

        if result is None: 
            logger.warning(f"Model with ID {model_id} not found in database") 
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"Model with ID {model_id} does not exist"
            )
        
        logger.info(f"Successfully retrieved model with ID: {result['id']}")
        return result 

    except HTTPException as e: 
        raise 
    except Exception as e: 
        logger.error(f"Failed to get model with id: {model_id} error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while fetching the model"
        )

@router.get(
    "/get/name/{model_name}"  
)
async def get_model_by_name_route_async(
    model_name: str, 
    model_service: Annotated[ModelService, Depends(get_model_service)]
) -> ModelDto:
    try: 
        model = await model_service.get_model_by_name_async(model_name)

        if model is None: 
            logger.warning(f"Model with name {model_name} not found in database") 
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"Model with name {model_name} does not exist"
            ) 

        return model 

    except HTTPException as e: 
        raise 
    except Exception as e: 
        logger.error(f"Failed to get model with name: {model_name} error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while fetching the model"
        )