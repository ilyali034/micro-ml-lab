
from uuid import UUID

from sqlalchemy import String
from typing import List, Optional

from sqlalchemy.orm import Mapped, mapped_column, relationship

from .experiments import ExperimentTable
from .base import Base 

class ModelTable(Base):
    __tablename__ = "models"

    id: Mapped[UUID] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    model_file_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    experiments: Mapped[List["ExperimentTable"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )
