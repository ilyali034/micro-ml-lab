from sqlalchemy import ForeignKey, MetaData, Table, Column, Integer, String, Text

metadata = MetaData()

experiments_table = Table(
    "experiments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("model_id", Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False),
    Column("run_name", String, nullable=False), 
    Column("params", Text),                     
    Column("metrics", Text),                    
    Column("model_file_path", String, nullable=True)
)

