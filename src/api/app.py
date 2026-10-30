from fastapi import FastAPI
from contextlib import asynccontextmanager  
import logging

from .routes.model_routes import router as model_router
from .routes.runtime_routes import router as runtime_router
from core.inference.runtime import LocalInferenceEngine
from core.database.db import DatabaseContext, init_db
from core.database.repositories.protocols import ModelRepository, ExperimentRepository

logger = logging.getLogger(__name__)  

_db_context: DatabaseContext | None = None
_inference_engine: LocalInferenceEngine | None = None 

@asynccontextmanager  
async def lifespan(app): 
    global _db_context, _inference_engine 

    logger.info('Initializing database...') 
    _db_context = await init_db(strategy="aiosqlite")
    app.state.db_context = _db_context
    logger.info('Database initialized successfully')

    _inference_engine = LocalInferenceEngine()
    app.state.inference_engine = _inference_engine
    logger.info('Local inference runtime initialized') 

    yield 

    if _inference_engine:
        logger.info('Unloading active ML model...')
        _inference_engine.unload()
        _inference_engine = None

    if _db_context: 
        logger.info("Closing database connections...")
        await _db_context.models.close()
        await _db_context.experiments.close() 
        logger.info("Database connections closed")

app = FastAPI(lifespan=lifespan) 
app.include_router(model_router)
app.include_router(runtime_router)

async def get_model_repository() -> ModelRepository: 
    if _db_context is None: 
        raise RuntimeError("Database not initialized")
    return _db_context.models

async def get_experiment_repository() -> ExperimentRepository: 
    if _db_context is None: 
        raise RuntimeError("Database not initialized")
    return _db_context.experiments

__all__ = ['app', 'get_model_repository', 'get_experiment_repository']