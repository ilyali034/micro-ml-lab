from typing import Optional

from sqlalchemy import ForeignKey, Text, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

class ExperimentTable(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    run_name: Mapped[str] = mapped_column(String, nullable=False)
    
    params: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_file_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    model: Mapped["ModelTable"] = relationship(back_populates="experiments")
