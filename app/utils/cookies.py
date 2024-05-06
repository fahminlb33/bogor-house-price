import uuid


def ensure_user_has_session(_manager):
  if not _manager.get("session_id"):
    _manager.set("session_id", str(uuid.uuid4()))


def get_session_id(_manager):
  return _manager.get("session_id")
