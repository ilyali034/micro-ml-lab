import os
from typing import Literal, NamedTuple, Union
from .tables.experiments import metadata
from .repositories.model_repository import AioSqliteModelRepository, SqlAlchemyModelRepository
from .repositories.experiment_repository import AioSqliteExperimentRepository, SqlAlchemyExperimentRepository

class DatabaseContext(NamedTuple):
    models: Union[AioSqliteModelRepository, SqlAlchemyModelRepository]
    experiments: Union[AioSqliteExperimentRepository, SqlAlchemyExperimentRepository]

async def init_db(strategy: Literal["aiosqlite", "sqlalchemy"] | None = None) -> DatabaseContext:
    """
    Initializes the tracking database components using a selected driver framework strategy.
    Guarantees correct cross-table sequence generation and returns paired storage repositories.
    """
    if strategy is None:
        strategy = os.getenv("DB_STRATEGY", "aiosqlite")

    print(f"🔄 Selected database engine: {strategy}")

    if strategy == "aiosqlite":
        db_path = "ml_lab.db"
        model_repo = AioSqliteModelRepository(db_path=db_path)
        experiment_repo = AioSqliteExperimentRepository(db_path=db_path)
        
        print(f"Running schema initialization for {strategy}...")
        await model_repo.connect()
        await experiment_repo.connect()

    elif strategy == "sqlalchemy":
        database_url = "sqlite+aiosqlite:///ml_lab.db"
        model_repo = SqlAlchemyModelRepository(database_url=database_url)
        experiment_repo = SqlAlchemyExperimentRepository(database_url=database_url)
        
        print(f"Running schema initialization for {strategy}...")
        await model_repo.connect() 
    else:
        raise ValueError(f"Unknown database strategy: {strategy}")

    print("Database components successfully initialized!")
    
    return DatabaseContext(models=model_repo, experiments=experiment_repo)
