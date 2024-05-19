from sqlalchemy import String, Integer, Float, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from utils.db import Base

# yapf: disable

class Prediction(Base):
  __tablename__ = "predictions"

  request: Mapped[str] = mapped_column(Text)
  predicted: Mapped[float] = mapped_column(Float)



class Session(Base):
  __tablename__ = "sessions"

  session_token: Mapped[str] = mapped_column(String(36))

  chats: Mapped[list["Chat"]] = relationship(back_populates="session")


class Chat(Base):
  __tablename__ = "chats"

  session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
  prompt: Mapped[str] = mapped_column(Text)
  response: Mapped[str] = mapped_column(Text)
  model: Mapped[str] = mapped_column(String(100))
  role: Mapped[str] = mapped_column(String(100))
  prompt_tokens: Mapped[int] = mapped_column(Integer)
  completion_tokens: Mapped[int] = mapped_column(Integer)

  session: Mapped[Session] = relationship(back_populates="chats")
  retrieved_docs: Mapped[list["RetrievedDocument"]] = relationship(back_populates="chats")


class RetrievedDocument(Base):
  __tablename__ = "retrieved_documents"

  city: Mapped[str] = mapped_column(String(255))
  district: Mapped[str] = mapped_column(String(255))
  price: Mapped[float] = mapped_column(Float)
  chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id"))
  qdrant_document_id: Mapped[str] = mapped_column(String(36))

  chats: Mapped[Chat] = relationship(back_populates="retrieved_docs")

# yapf: enable
