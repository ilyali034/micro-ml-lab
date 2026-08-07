import json
import aiosqlite
from typing import Any, Dict, List
from sqlalchemy import select
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.schema import CreateTable

from ..tables.experiments import ExperimentTable

class AioSqliteExperimentRepository(object):
    def __init__(self, db_path: str = "ml_lab.db"):
        self.db_path = db_path
        self.db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db = await aiosqlite.connect(self.db_path)
        await self.db.execute("PRAGMA journal_mode=WAL;")
        await self.db.execute("PRAGMA foreign_keys=ON;")
        create_stmt = str(CreateTable(ExperimentTable.__table__).compile(dialect=sqlite.dialect()))
        await self.db.execute(create_stmt)
        await self.db.commit()

    async def log_experiment(self, model_id: int, name: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> int:
        stmt = ExperimentTable.__table__.insert().values(
            model_id=model_id,
            run_name=name,  
            params=json.dumps(params), 
            metrics=json.dumps(metrics),
        )
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params) as cursor:
            await self.db.commit()
            return cursor.lastrowid

    async def get_all_experiments(self) -> List[Dict[str, Any]]:
        stmt = ExperimentTable.__table__.select()
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0], 
                    "model_id": r[1],
                    "run_name": r[2], 
                    "params": json.loads(r[3]) if r[3] else {}, 
                    "metrics": json.loads(r[4]) if r[4] else {},
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
        pass  

    async def log_experiment(self, model_id: int, name: str, params: Dict[str, Any], metrics: Dict[str, Any], model_file_path: str | None = None) -> int:
        async with self.session_factory() as session:
            stmt = ExperimentTable.__table__.insert().values(
                model_id=model_id,
                run_name=name, 
                params=json.dumps(params), 
                metrics=json.dumps(metrics),
                model_file_path=model_file_path
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.inserted_primary_key[0]

    async def get_all_experiments(self) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            stmt = select(ExperimentTable)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "id": r.id, 
                    "model_id": r.model_id,
                    "run_name": r.run_name, 
                    "params": json.loads(r.params) if r.params else {}, 
                    "metrics": json.loads(r.metrics) if r.metrics else {},
                    "model_file_path": r.model_file_path
                }
                for r in rows
            ]

    async def close(self) -> None:
        await self.engine.dispose()
