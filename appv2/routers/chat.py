import os
import uuid

import filetype
from flask import Blueprint, request, render_template, abort, send_from_directory

from appv2.utils.shared import cache, get_settings
from appv2.services.rag.chat import rag, ChatHistory
from appv2.services.rag.tools import HousePricePredictionTool
from appv2.services.repository.tracking import session_id_by_token, get_messages, track_prompt

router = Blueprint('chat', __name__)


@router.route('/chat')
@cache.cached()
def page():
  return render_template(f'pages/chat.html')


@router.route('/chat/completion', methods=['POST'])
def completion():
  settings = get_settings()

  # get or create session
  session_id = session_id_by_token(request.form['session_token'])

  # get prompt or file upload
  prompt = ""
  if 'file' not in request.files:
    # get the prompt text
    prompt = request.form['prompt']
  else:
    # get the uploaded file
    file = request.files['file']

    # check if the file exists
    if file.filename == '':
      abort(400)

    # check file type
    ftype = filetype.guess(file.stream)
    print(ftype)
    if ftype is None or ftype.extension not in ["jpg", "jpeg", "png"]:
      abort(400)

    # save file
    save_name = f"{uuid.uuid4()}.jpg"
    file.save(os.path.join(settings.UPLOAD_DIR, save_name))

    # build prompt
    prompt = "Recommend me a house based on this image file: " + save_name

  # get message history
  messages = get_messages(session_id)
  history = [
      ChatHistory(
          role=message.role, content=message.prompt, tool_name=message.model)
      for message in messages
  ]

  # get response
  reply, archive = rag.chat(prompt, history)

  # create links if this is a RAG tool
  links = []
  for arch in archive:
    if arch.role == "tool" and arch.model != HousePricePredictionTool.tool_name:
      for doc in arch.tool_result.documents:
        links.append({
            "address": f"{doc.district}, {doc.city}",
            "price": doc.price,
            "image_url": doc.image_url,
            "url": doc.url
        })

  # save archive
  track_prompt(session_id, archive)

  return {"content": reply, "links": links}


@router.route('/chat/uploads/<filename>')
def serve_uploaded_files(filename):
  return send_from_directory(get_settings().UPLOAD_DIR, filename)
