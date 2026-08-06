import json
import aiosqlite
from typing import Any, Dict, List
from sqlalchemy import select
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.schema import CreateTable

from ..tables.experiments import metadata, experiments_table

class AioSqliteExperimentRepository(object):
    def __init__(self, db_path: str = "ml_lab.db"):
        self.db_path = db_path
        self.db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db = await aiosqlite.connect(self.db_path)
        await self.db.execute("PRAGMA journal_mode=WAL;")
        await self.db.execute("PRAGMA foreign_keys=ON;")
        create_stmt = str(CreateTable(experiments_table).compile(dialect=sqlite.dialect()))
        await self.db.execute(create_stmt)
        await self.db.commit()

    async def log_experiment(self, model_id: int, name: str, params: Dict[str, Any], metrics: Dict[str, Any], model_file_path: str | None = None) -> int:
        stmt = experiments_table.insert().values(
            model_id=model_id,
            name=name, 
            params=json.dumps(params), 
            metrics=json.dumps(metrics),
            model_file_path=model_file_path 
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
                {
                    "id": r[0], 
                    "model_id": r[1],
                    "name": r[2], 
                    "params": json.loads(r[3]), 
                    "metrics": json.loads(r[4]),
                    "model_file_path": r[5] 
                }
                for r in rows
            ]

    async def close(self) -> None:
        if self.db:
            await self.db.close()


class SqlAlchemyExperimentRepository(object):
    def __init__(self, database_url: str = "sqlite+aiosqlite:///ml_lab.db"):
        self.engine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def connect(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def log_experiment(self, model_id: int, name: str, params: Dict[str, Any], metrics: Dict[str, Any], model_file_path: str | None = None) -> int:
        async with self.session_factory() as session:
            stmt = experiments_table.insert().values(
                model_id=model_id,
                name=name, 
                params=json.dumps(params), 
                metrics=json.dumps(metrics),
                model_file_path=model_file_path 
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
                {
                    "id": r[0], 
                    "model_id": r[1],
                    "name": r[2], 
                    "params": json.loads(r[3]), 
                    "metrics": json.loads(r[4]),
                    "model_file_path": r[5] 
                }
                for r in rows
            ]

    async def close(self) -> None:
        await self.engine.dispose()
