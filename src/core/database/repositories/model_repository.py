import json
import aiosqlite
from typing import Any, Dict, List
from sqlalchemy import select, func
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.schema import CreateTable

from ..tables.experiments import ExperimentTable
from ..tables.base import Base 
from ..tables.models import ModelTable
from ...exceptions.models import ModelNotFoundError

class AioSqliteModelRepository(object):
    def __init__(self, db_path: str = "ml_lab.db"):
        self.db_path = db_path
        self.db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db = await aiosqlite.connect(self.db_path)
        await self.db.execute("PRAGMA journal_mode=WAL;")
        await self.db.execute("PRAGMA foreign_keys=ON;")
        
        create_stmt = str(CreateTable(ModelTable.__table__).compile(dialect=sqlite.dialect()))
        await self.db.execute(create_stmt)
        await self.db.commit()

    async def create_model(self, name: str, model_file_path: str | None = None) -> int:
        stmt = ModelTable.__table__.insert().values(name=name, model_file_path=model_file_path)
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params) as cursor:
            await self.db.commit()
            return cursor.lastrowid

    async def get_model_by_name(self, name: str) -> Dict[str, Any] | None:
        stmt = ModelTable.__table__.select().where(ModelTable.__table__.c.name == name)
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {"id": row[0], "name": row[1], "model_file_path": row[2]}

    async def get_model_by_id(self, model_id: int) -> Dict[int, Any] | None:
        stmt = ModelTable.__table__.select().where(ModelTable.__table__.c.id == model_id)
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {"id": row[0], "name": row[1], "model_file_path": row[2]}

    async def update_model_file_path(self, model_id: int, model_file_path: str) -> None:
        stmt = ModelTable.__table__.update().where(
            ModelTable.__table__.c.id == model_id
        ).values(model_file_path=model_file_path)
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params):
            await self.db.commit()

    async def get_all_models_with_experiment_count(self) -> List[Dict[str, Any]]:
        m_t = ModelTable.__table__
        e_t = ExperimentTable.__table__
        
        stmt = select(m_t.c.id, m_t.c.name, func.count(e_t.c.id)).select_from(
            m_t.outerjoin(e_t, m_t.c.id == e_t.c.model_id)
        ).group_by(m_t.c.id)
        
        compiled = stmt.compile(dialect=sqlite.dialect())
        async with self.db.execute(str(compiled), compiled.params) as cursor:
            rows = await cursor.fetchall()
            return [{"id": r[0], "name": r[1], "experiment_count": r[2]} for r in rows]
        
    async def delete_model(self, model_id: int) -> bool:
            check_stmt = ModelTable.__table__.select().where(ModelTable.__table__.c.id == model_id)
            compiled_check = check_stmt.compile(dialect=sqlite.dialect())
            async with self.db.execute(str(compiled_check), compiled_check.params) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return False

            delete_stmt = ModelTable.__table__.delete().where(ModelTable.__table__.c.id == model_id)
            compiled_delete = delete_stmt.compile(dialect=sqlite.dialect())
            async with self.db.execute(str(compiled_delete), compiled_delete.params):
                await self.db.commit()

            return True
    
    async def close(self) -> None:
        if self.db:
            await self.db.close()


class SqlAlchemyModelRepository(object): 
    def __init__(self, database_url: str = "sqlite+aiosqlite:///ml_lab.db"):
        self.engine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def connect(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create_model(self, name: str, model_file_path: str | None = None) -> int:
        async with self.session_factory() as session:
            stmt = ModelTable.__table__.insert().values(name=name, model_file_path=model_file_path)
            result = await session.execute(stmt)
            await session.commit()
            return result.inserted_primary_key[0]

    async def get_model_by_name(self, name: str) -> Dict[str, Any] | None:
        async with self.session_factory() as session:
            stmt = select(ModelTable).where(ModelTable.name == name)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if not row:
                return None
            return {"id": row.id, "name": row.name, "model_file_path": row.model_file_path}

    async def get_model_by_id(self, model_id: int) -> Dict[int, Any] | None:
        async with self.session_factory() as session: 
            stmt = select(ModelTable).where(ModelTable.id == model_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if not row:
                return None
            return {"id": row.id, "name": row.name, "model_file_path": row.model_file_path}

    async def update_model_file_path(self, model_id: int, model_file_path: str) -> None:
        async with self.session_factory() as session:
            stmt = ModelTable.__table__.update().where(
                ModelTable.__table__.c.id == model_id
            ).values(model_file_path=model_file_path)
            await session.execute(stmt)
            await session.commit()

    async def get_all_models_with_experiment_count(self) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            stmt = select(
                ModelTable.id, 
                ModelTable.name, 
                func.count(ExperimentTable.id)
            ).select_from(
                ModelTable
            ).outerjoin(
                ExperimentTable, ModelTable.id == ExperimentTable.model_id
            ).group_by(ModelTable.id)
            
            result = await session.execute(stmt)
            rows = result.fetchall()
            return [{"id": r[0], "name": r[1], "experiment_count": r[2]} for r in rows]

    async def delete_model(self, model_id: int) -> bool:
            async with self.session_factory() as session:
                stmt = select(ModelTable).where(ModelTable.id == model_id)
                result = await session.execute(stmt)
                model = result.scalar_one_or_none()

                if model is None:
                    return False

                await session.delete(model)
                await session.commit()

            return True

    async def close(self) -> None:
        await self.engine.dispose()
