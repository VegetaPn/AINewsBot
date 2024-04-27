import logging
import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-3.5-turbo", request_timeout=30)


def chat(messages):
    ai_message = llm.invoke(messages)
    return ai_message
