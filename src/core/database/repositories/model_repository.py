import json
import aiosqlite
from typing import Any, Dict, List
from sqlalchemy import select, func
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.schema import CreateTable

from ..tables.experiments import metadata, models_table, experiments_table

class AioSqliteModelRepository(object):
    def __init__(self, db_path: str = "ml_lab.db"):
        self.db_path = db_path
        self.db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db = await aiosqlite.connect(self.db_path)
        # Enable write-ahead logging and enforce Foreign Key checks in SQLite
        await self.db.execute("PRAGMA journal_mode=WAL;")
        await self.db.execute("PRAGMA foreign_keys=ON;")
        
        # Compile and create the models table safely
        create_stmt = str(CreateTable(models_table).compile(dialect=sqlite.dialect()))
        await self.db.execute(create_stmt)
        await self.db.commit()

    async def create_model(self, name: str) -> int:
        """Inserts a new model architecture profile."""
        stmt = models_table.insert().values(name=name)
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params) as cursor:
            await self.db.commit()
            return cursor.lastrowid

    async def get_model_by_name(self, name: str) -> Dict[str, Any] | None:
        """Finds a model record by its unique name string."""
        stmt = models_table.select().where(models_table.c.name == name)
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {"id": row[0], "name": row[1]}

    async def get_all_models_with_experiment_count(self) -> List[Dict[str, Any]]:
        """Fetches all models and calculates how many experiments belong to them via LEFT JOIN."""
        stmt = select(
            models_table.c.id, 
            models_table.c.name, 
            func.count(experiments_table.c.id).label("exp_count")
        ).select_from(
            models_table.outerjoin(experiments_table, models_table.c.id == experiments_table.c.model_id)
        ).group_by(models_table.c.id)
        
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params) as cursor:
            rows = await cursor.fetchall()
            return [{"id": r[0], "name": r[1], "experiment_count": r[2]} for r in rows]

    async def close(self) -> None:
        if self.db:
            await self.db.close()


class SqlAlchemyModelRepository(object): 
    def __init__(self, database_url: str = "sqlite+aiosqlite:///ml_lab.db"):
        self.engine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def connect(self) -> None:
        # metadata.create_all handles table sequencing safely automatically
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def create_model(self, name: str) -> int:
        """Inserts a new model architecture profile."""
        async with self.session_factory() as session:
            stmt = models_table.insert().values(name=name)
            result = await session.execute(stmt)
            await session.commit()
            return result.inserted_primary_key[0]

    async def get_model_by_name(self, name: str) -> Dict[str, Any] | None:
        """Finds a model record by its unique name string."""
        async with self.session_factory() as session:
            stmt = select(models_table).where(models_table.c.name == name)
            result = await session.execute(stmt)
            row = result.fetchone()
            if not row:
                return None
            return {"id": row[0], "name": row[1]}

    async def get_all_models_with_experiment_count(self) -> List[Dict[str, Any]]:
        """Fetches all models and calculates how many experiments belong to them via LEFT JOIN."""
        async with self.session_factory() as session:
            stmt = select(
                models_table.c.id, 
                models_table.c.name, 
                func.count(experiments_table.c.id)
            ).select_from(
                models_table.outerjoin(experiments_table, models_table.c.id == experiments_table.c.model_id)
            ).group_by(models_table.c.id)
            
            result = await session.execute(stmt)
            rows = result.fetchall()
            return [{"id": r[0], "name": r[1], "experiment_count": r[2]} for r in rows]

    async def close(self) -> None:
        await self.engine.dispose()
