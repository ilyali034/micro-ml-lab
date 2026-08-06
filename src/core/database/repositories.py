import json
import aiosqlite
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from sqlalchemy import select
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.schema import CreateTable
from .tables.experiment import metadata, experiments_table

class AbstractMLRepository(ABC):
    @abstractmethod
    async def connect(self) -> None: pass
    
    @abstractmethod
    async def log_experiment(self, name: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> int: pass
    
    @abstractmethod
    async def get_all_experiments(self) -> List[Dict[str, Any]]: pass
    
    @abstractmethod
    async def close(self) -> None: pass


class AioSqliteRepository(AbstractMLRepository):
    def __init__(self, db_path: str = "ml_lab.db"):
        self.db_path = db_path
        self.db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db = await aiosqlite.connect(self.db_path)
        await self.db.execute("PRAGMA journal_mode=WAL;")
        create_stmt = str(CreateTable(experiments_table).compile(dialect=sqlite.dialect()))
        await self.db.execute(create_stmt)
        await self.db.commit()

    async def log_experiment(self, name: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> int:
        stmt = experiments_table.insert().values(
            name=name, params=json.dumps(params), metrics=json.dumps(metrics)
        )
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params) as cursor:
            await self.db.commit()
            return cursor.lastrowid

    async def get_all_experiments(self) -> List[Dict[str, Any]]:
        stmt = experiments_table.select()
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params) as cursor:
            rows = await cursor.fetchall()
            return [
                {"id": r[0], "name": r[1], "params": json.loads(r[2]), "metrics": json.loads(r[3])}
                for r in rows
            ]

    async def close(self) -> None:
        if self.db:
            await self.db.close()


class SqlAlchemyRepository(AbstractMLRepository):
    def __init__(self, database_url: str = "sqlite+aiosqlite:///ml_lab.db"):
        self.engine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def connect(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def log_experiment(self, name: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> int:
        async with self.session_factory() as session:
            stmt = experiments_table.insert().values(
                name=name, params=json.dumps(params), metrics=json.dumps(metrics)
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.inserted_primary_key[0]

    async def get_all_experiments(self) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            stmt = select(experiments_table)
            result = await session.execute(stmt)
            rows = result.fetchall()
            return [
                {"id": r[0], "name": r[1], "params": json.loads(r[2]), "metrics": json.loads(r[3])}
                for r in rows
            ]

    async def close(self) -> None:
        await self.engine.dispose()
