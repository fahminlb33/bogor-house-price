from datetime import datetime

from sqlalchemy import String, DateTime, create_engine
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, Mapped, scoped_session, sessionmaker, mapped_column

from appv2.utils.shared import get_settings

engine = create_engine(get_settings().SQLALCHEMY_DATABASE_URI)
db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine))


class Base(DeclarativeBase):
  id: Mapped[str] = mapped_column(String(36), primary_key=True)
  created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
