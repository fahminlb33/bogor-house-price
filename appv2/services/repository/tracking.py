import uuid
import json

from sqlalchemy.dialects.mysql import insert

from appv2.utils.db import db_session
from appv2.utils.models import (Prediction, Session, Chat, RetrievedDocument)
from appv2.services.rag.chat import ChatItem
from appv2.services.rag.tools import HousePricePredictionTool


def session_id_by_token(token: str) -> str:
  # find the session
  session = db_session.query(Session).filter(
      Session.session_token == token).first()

  # if no session, create one
  if not session:
    session = Session(id=str(uuid.uuid4()), session_token=token)

    db_session.add(session)
    db_session.commit()

  return str(session.id)


def get_messages(session_id: str) -> list[Chat]:
  return db_session.query(Chat).filter(Chat.session_id == session_id).all()


def track_prediction(payload: dict, predicted: float):
  # save prediction
  prediction = Prediction(
      id=str(uuid.uuid4()), request=json.dumps(payload), predicted=predicted)

  db_session.add(prediction)
  db_session.commit()


def track_prompt(session_id: str, archive: list[ChatItem]):
  # save all history
  for item in archive:
    # save chat
    chat = Chat(
        id=str(uuid.uuid4()),
        session_id=session_id,
        prompt=item.prompt,
        response=item.response,
        model=item.model,
        role=item.role,
        prompt_tokens=item.prompt_tokens,
        completion_tokens=item.completion_tokens,
    )

    db_session.add(chat)

    # check if this is a RAG tool
    if item.role == "tool" and item.model != HousePricePredictionTool.tool_name:
      # get the data
      for doc in item.tool_result.documents:
        # save retrieved document
        db_session.add(
            RetrievedDocument(
                id=str(uuid.uuid4()),
                chat_id=chat.id,
                qdrant_document_id=doc.id,
                city=doc.city,
                district=doc.district,
                price=doc.price))

  # commit transaction
  db_session.commit()
