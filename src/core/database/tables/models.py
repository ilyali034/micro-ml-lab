
from sqlalchemy import String
from typing import List

from sqlalchemy.orm import Mapped, mapped_column, relationship


class ModelTable(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    experiments: Mapped[List["ExperimentTable"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )