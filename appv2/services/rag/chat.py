import json
import dataclasses
from typing import Any

from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator

from appv2.utils.shared import get_settings
from appv2.services.rag.prompts import PROMPT_FOR_AGENT
from appv2.services.rag.tools import (BaseTool, OpenAIUsage, HouseDocument,
                                      HouseRecommendationTool,
                                      HouseImageSearchTool,
                                      HousePricePredictionTool)


@dataclasses.dataclass
class ChatHistory:
  role: str
  content: str
  tool_name: str


@dataclasses.dataclass
class ChatItem:
  prompt: str
  response: str
  model: str
  role: str
  prompt_tokens: int
  completion_tokens: int
  tool_result: Any


class ChatRAG:

  def __init__(self) -> None:
    self.settings = get_settings()

    # create OpenAI chat generator
    self.chat_generator = OpenAIChatGenerator(
        model=self.settings.OPENAI_CHAT_MODEl)

    # init tools
    self.tools: dict[str, BaseTool] = {
        HouseRecommendationTool.tool_name: HouseRecommendationTool(),
        HouseImageSearchTool.tool_name: HouseImageSearchTool(),
        HousePricePredictionTool.tool_name: HousePricePredictionTool()
    }

    self.available_tools = [
        tool.get_tool_schema() for tool in self.tools.values()
    ]

  def chat(self, question: str,
           history: list[ChatHistory]) -> tuple[str, list[ChatItem]]:
    if len(history) == 0:
      history = [ChatMessage.from_system(PROMPT_FOR_AGENT)]

    # build message history
    chat_archive: list[ChatItem] = []
    chat_messages: list[ChatMessage] = []
    for h in history:
      if h.role == "user":
        chat_messages.append(ChatMessage.from_user(h.content))
      elif h.role == "assistant":
        chat_messages.append(ChatMessage.from_assistant(h.content))
      elif h.role == "tool":
        chat_messages.append(ChatMessage.from_function(h.content, h.tool_name))

    chat_messages.append(ChatMessage.from_user(question))

    # call chat
    response = self.chat_generator.run(
        chat_messages, generation_kwargs={"tools": self.available_tools})

    # CHATTING NOW!!!
    while True:
      # check if reply is a tool call
      if response and response["replies"][0].meta["finish_reason"] == "tool_calls": # yapf: disable
        # parse function calls
        function_calls = json.loads(response["replies"][0].content)

        # archive call
        chat_archive.append(
            ChatItem(
                prompt=chat_messages[-1].content,
                response="TOOL CALL",
                model=self.settings.OPENAI_CHAT_MODEl,
                role="tool_intro",
                prompt_tokens=0,
                completion_tokens=0,
                tool_result=None))

        # process each tool call requests
        for function_call in function_calls:
          # parse function calling information
          fname = function_call["function"]["name"]
          fargs = json.loads(function_call["function"]["arguments"])

          # find the correspoding function and call it with the given arguments
          fresult, fusage = self.tools[fname](**fargs)
          print("FUNC RESULT", fresult)

          # remove documents if exists in the fresult
          fresult_safe = dataclasses.asdict(fresult)
          fresult_safe.pop("documents", None)
          print("FUNC RES SAFE", fresult_safe)

          # append to archive
          chat_archive.append(
              ChatItem(
                  prompt=chat_messages[-1].content,
                  response=function_call["function"]["arguments"],
                  model=fname,
                  role="tool",
                  prompt_tokens=fusage.prompt_tokens,
                  completion_tokens=fusage.completion_tokens,
                  tool_result=fresult))

          # append function response to the messages list
          chat_messages.append(
              ChatMessage.from_function(
                  content=json.dumps(fresult_safe), name=fname))

          # call chat again
          response = self.chat_generator.run(
              messages=chat_messages,
              generation_kwargs={"tools": self.available_tools})

      # regular conversation
      else:
        # parse usage
        usage = OpenAIUsage.from_dict(response["replies"][0].meta)

        # save to archive
        chat_archive.append(
            ChatItem(
                prompt=chat_messages[-1].content,
                response=response["replies"][0].content,
                model=self.settings.OPENAI_CHAT_MODEl,
                role="assistant",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                tool_result=None))

        # save to history
        chat_messages.append(response["replies"][0])
        break

    # create result set
    return response["replies"][0].content, chat_archive


rag = ChatRAG()
