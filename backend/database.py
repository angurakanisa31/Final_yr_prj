import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Use SQLite local file database for development (target SQLite in user workspace)
DATABASE_URL = "sqlite:///c:/Users/angur/OneDrive/Documents/Final Yr Project/backend/ecommerce.db"

# Create database engine
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base class for models
Base = declarative_base()

def get_db():
    """
    FastAPI dependency that yields a database session and closes it when done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
