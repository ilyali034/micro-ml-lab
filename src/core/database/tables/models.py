from sqlalchemy import MetaData, Table, Column, Integer, String, Text

metadata = MetaData()

models_table = Table(
    "models",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False, unique=True), 
)
