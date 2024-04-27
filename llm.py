import logging
import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def chat(messages):
    llm = ChatOpenAI(model="gpt-4-turbo", request_timeout=30)
    ai_message = llm.invoke(messages)
    return ai_message
