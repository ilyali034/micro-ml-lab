
from uuid import UUID

from fastapi import Depends, status, HTTPException 
from typing import Annotated

import logging

from ...shared.dtos.models import ModelDto, CreateModelDto
from ...core.model_service import ModelService

from app import app 

logger = logging.Logger(__name__) 

_model_service_instance = ModelService()

def get_model_service() -> ModelService: 
    return _model_service_instance

@app.post(
    "/models/add", 
    response_model=dict[str, UUID],
    status_code=status.HTTP_201_CREATED
)
async def add_model_route_async(
    model: bool,
    model_service: Annotated[ModelService, Depends(get_model_service)]
): 

    logger.info(f"Received request to create model: {model.name}")
    
    try: 
        result = await model_service.add_model(model.name, model.model_file_path)
        logger.info(f"Successfully created model with ID: {result.id}")
        return result
    
    except Exception as e: 
        logger.error(f"Failed to create model: {str(e)}", exc_info=True)
        raise 


@app.delete(
    "/models/delete/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_model_route_async(
    model_id: UUID,
    model_service: Annotated[ModelService, Depends(get_model_service)]
) -> bool: 

    try: 
        model = await model_service.delete_model(model) 
        if model is True: 
            logger.info(f"Successfully deleted model with id {model_id}")
            return True 

    except HTTPException as e: 
        raise 
    
    except Exception as e: 
        logger.error(f"Failed deleted model with id: {model_id} error: {str(e)}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while fetching the model"
        )


@app.get(
    "/models/get/{model_id}",
    response_model=ModelDto
)
async def get_model_route_async(
    model_id: UUID,
    model_service: Annotated[ModelService, Depends(get_model_service)] 
) -> ModelDto: 

    try: 
        result = await model_service.get_model_by_id(model_id) 

        if result is None: 
            logger.warning(f"Model with ID {model_id} not found in database") 
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"Model with ID {model_id} does not exists"
            )
        
        logger.info(f"Successfully getted model with ID: {result.id}")
        return result 

    except HTTPException as e: 
        raise 

    except Exception as e: 
        logger.error(f"Failed get model with id: {model_id} error: {str(e)}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while fetching the model"
        )