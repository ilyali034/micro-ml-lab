import os
from typing import Literal, NamedTuple, Union
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import sqlite

from .tables import Base, ModelTable, ExperimentTable
from .repositories.model_repository import AioSqliteModelRepository, SqlAlchemyModelRepository
from .repositories.experiment_repository import AioSqliteExperimentRepository, SqlAlchemyExperimentRepository
from .repositories.protocols import ModelRepository, ExperimentRepository 

class DatabaseContext(NamedTuple):
    models: ModelRepository
    experiments: ExperimentRepository

async def init_db(strategy: Literal["aiosqlite", "sqlalchemy"] | None = None) -> DatabaseContext:
    """
    Centralized database initialization. Creates database files and 
    generates all tables in one clean, isolated place.
    """
    if strategy is None:
        strategy = os.getenv("DB_STRATEGY", "aiosqlite")

    print(f"Selected database engine: {strategy}")

    if strategy == "aiosqlite":
        db_path = "ml_lab.db"
        model_repo = AioSqliteModelRepository(db_path=db_path)
        experiment_repo = AioSqliteExperimentRepository(db_path=db_path)
        
        await model_repo.connect()
        await experiment_repo.connect()
        
        print(f"Creating aiosqlite schema elements...")
        model_stmt = str(CreateTable(ModelTable.__table__).compile(dialect=sqlite.dialect()))
        exp_stmt = str(CreateTable(ExperimentTable.__table__).compile(dialect=sqlite.dialect()))
        
        async with model_repo.db.execute(model_stmt): pass
        async with model_repo.db.execute(exp_stmt): pass
        await model_repo.db.commit()

    elif strategy == "sqlalchemy":
        database_url = "sqlite+aiosqlite:///ml_lab.db"
        model_repo = SqlAlchemyModelRepository(database_url=database_url)
        experiment_repo = SqlAlchemyExperimentRepository(database_url=database_url)
        
        print(f"Creating SQLAlchemy schema elements...")
        async with model_repo.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
    else:
        raise ValueError(f"❌ Unknown database strategy: {strategy}")

    print("✅ Database components successfully initialized!")
    return DatabaseContext(models=model_repo, experiments=experiment_repo)
