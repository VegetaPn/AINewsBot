import logging
import os
import sys

from dotenv import load_dotenv

from openai import OpenAI


console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.terminator = ""
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_handler.terminator = ""
nonl_logger = logging.getLogger('nonl')
nonl_logger.setLevel(logging.INFO)
nonl_logger.addHandler(console_handler)
nonl_logger.addHandler(file_handler)

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

client = OpenAI()


def chat(messages):
    stream = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        stream=True,
        temperature=1.0,
        max_tokens=4095
    )

    ai_message = ""
    # nonl_logger.info(f"[LLM] Response:")
    print(f"[LLM] Response: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            # nonl_logger.info(content)
            print(f"{content}", end="")
            ai_message = ai_message + content
    print("\n")
    # logging.info("\n")

    return ai_message
