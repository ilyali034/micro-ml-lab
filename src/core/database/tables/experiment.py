from sqlalchemy import MetaData, Table, Column, Integer, String, Text

metadata = MetaData()

experiments_table = Table(
    "experiments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False),
    Column("params", Text),
    Column("metrics", Text),
)
