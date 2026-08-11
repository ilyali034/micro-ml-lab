import json
from uuid import UUID
import aiosqlite
from typing import Any, Dict, List, Optional
from sqlalchemy import select, func
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

    async def log_experiment(self, model_id: UUID, name: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> UUID:
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
        
    async def get_all_experiments(
        self,
        limit: int = 10,
        offset: int = 0,
        model_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM experiments"
        params = []
        
        if model_id:
            query += " WHERE model_id = ?"
            params.append(str(model_id))
        
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [
                {
                    "id": row[columns.index("id")],
                    "model_id": row[columns.index("model_id")],
                    "run_name": row[columns.index("run_name")],
                    "params": json.loads(row[columns.index("params")]) if row[columns.index("params")] else {},
                    "metrics": json.loads(row[columns.index("metrics")]) if row[columns.index("metrics")] else {},
                    "model_file_path": row[columns.index("model_file_path")]
                }
                for row in rows
            ]

    async def get_experiments_count(
        self,
        model_id: Optional[UUID] = None
    ) -> int:
        query = "SELECT COUNT(*) FROM experiments"
        params = []
        
        if model_id:
            query += " WHERE model_id = ?"
            params.append(str(model_id))
        
        async with self.db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def get_experiment_by_id(
        self,
        experiment_id: int
    ) -> Optional[Dict[str, Any]]:

        query = "SELECT * FROM experiments WHERE id = ?"
        async with self.db.execute(query, (experiment_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            columns = [description[0] for description in cursor.description]
            result = {
                "id": row[columns.index("id")],
                "model_id": row[columns.index("model_id")],
                "run_name": row[columns.index("run_name")],
                "params": json.loads(row[columns.index("params")]) if row[columns.index("params")] else {},
                "metrics": json.loads(row[columns.index("metrics")]) if row[columns.index("metrics")] else {},
                "model_file_path": row[columns.index("model_file_path")]
            }
            return result

    async def close(self) -> None:
        if self.db:
            await self.db.close()


class SqlAlchemyExperimentRepository(object):
    def __init__(self, database_url: str = "sqlite+aiosqlite:///ml_lab.db"):
        self.engine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def connect(self) -> None:
        pass  

    async def log_experiment(self, model_id: UUID, name: str, params: Dict[str, Any], metrics: Dict[str, Any], model_file_path: str | None = None) -> UUID:
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

    async def get_all_experiments(
        self,
        limit: int = 10,
        offset: int = 0,
        model_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            stmt = select(ExperimentTable)
            
            if model_id:
                stmt = stmt.where(ExperimentTable.model_id == model_id)
            
            stmt = stmt.order_by(ExperimentTable.id.desc()).limit(limit).offset(offset)
            
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

    async def get_experiments_count(
        self,
        model_id: Optional[UUID] = None
    ) -> int:
        async with self.session_factory() as session:
            stmt = select(func.count()).select_from(ExperimentTable)
            
            if model_id:
                stmt = stmt.where(ExperimentTable.model_id == model_id)
            
            result = await session.execute(stmt)
            return result.scalar() or 0

    async def get_experiment_by_id(
        self,
        experiment_id: int
    ) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            stmt = select(ExperimentTable).where(ExperimentTable.id == experiment_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            
            if not row:
                return None
                
            return {
                "id": row.id,
                "model_id": row.model_id,
                "run_name": row.run_name,
                "params": json.loads(row.params) if row.params else {},
                "metrics": json.loads(row.metrics) if row.metrics else {},
                "model_file_path": row.model_file_path
            }

    async def close(self) -> None:
        await self.engine.dispose()
