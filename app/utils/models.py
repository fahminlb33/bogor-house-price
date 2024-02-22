from typing import Optional
from datetime import datetime

from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class UserSession(Base):
    __tablename__ = "sessions"

    session_token: Mapped[str] = mapped_column(String(36))

    prompts: Mapped[list["OpenAIPrompt"]] = relationship(
        back_populates="session")
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="session")


class Prediction(Base):
    __tablename__ = "predictions"

    request: Mapped[str] = mapped_column(Text)
    predicted: Mapped[float] = mapped_column(Float)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))

    session: Mapped[UserSession] = relationship(back_populates="predictions")


class OpenAIPrompt(Base):
    __tablename__ = "openai_prompts"

    prompt: Mapped[str] = mapped_column(String(1024))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))

    session: Mapped[UserSession] = relationship(back_populates="prompts")
    usages: Mapped[list["OpenAIUsage"]] = relationship(back_populates="prompt")
    responses: Mapped[list["OpenAIResponse"]] = relationship(
        back_populates="prompt")
    retrieved_docs: Mapped[list["RetrievedDocument"]] = relationship(
        back_populates="prompt")


class OpenAIResponse(Base):
    __tablename__ = "openai_responses"

    contents: Mapped[str] = mapped_column(Text)
    prompt_id: Mapped[str] = mapped_column(ForeignKey("openai_prompts.id"))

    prompt: Mapped[OpenAIPrompt] = relationship(back_populates="responses")
    usages: Mapped[list["OpenAIUsage"]] = relationship(
        back_populates="response")


class OpenAIUsage(Base):
    __tablename__ = "openai_usages"

    model: Mapped[str] = mapped_column(String(50))
    usage_type: Mapped[str] = mapped_column(String(50))
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    total_tokens: Mapped[int] = mapped_column(Integer)
    prompt_id: Mapped[str] = mapped_column(ForeignKey("openai_prompts.id"))
    response_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("openai_responses.id"))

    prompt: Mapped[OpenAIPrompt] = relationship(back_populates="usages")
    response: Mapped[OpenAIResponse] = relationship(back_populates="usages")


class RetrievedDocument(Base):
    __tablename__ = "retrieved_documents"

    city: Mapped[str] = mapped_column(String(255))
    district: Mapped[str] = mapped_column(String(255))
    price: Mapped[float] = mapped_column(Float)
    document_id: Mapped[str] = mapped_column(String(36))
    prompt_id: Mapped[str] = mapped_column(ForeignKey("openai_prompts.id"))

    prompt: Mapped[OpenAIPrompt] = relationship(back_populates="retrieved_docs")
