import os
from typing import Literal
from .tables.experiment import metadata, experiments_table
from repositories import AbstractMLRepository, AioSqliteRepository, SqlAlchemyRepository

async def init_db(strategy: Literal["aiosqlite", "sqlalchemy"] | None = None) -> AbstractMLRepository:
    """
    Initializes the database based on the provided strategy or an environment variable.
    Automatically creates database files and sets up the required tables.
    """
    if strategy is None:
        strategy = os.getenv("DB_STRATEGY", "aiosqlite")

    print(f"Selected database engine: {strategy}")

    if strategy == "aiosqlite":
        repo = AioSqliteRepository(db_path="ml_lab.db")
    elif strategy == "sqlalchemy":
        repo = SqlAlchemyRepository(database_url="sqlite+aiosqlite:///ml_lab.db")
    else:
        raise ValueError(f"❌ Unknown database strategy: {strategy}")

    print(f"Running schema initialization for {strategy}...")
    await repo.connect()
    print("Database successfully initialized!")
    
    return repo
