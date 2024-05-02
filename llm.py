import json
import logging
import os
import sys
import time

from dotenv import load_dotenv

from openai import OpenAI


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
    print(f"[LLM] Response: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            # nonl_logger.info(content)
            print(f"{content}", end="")
            ai_message = ai_message + content
    print("\n")

    return ai_message


def batch_chat(file_path, description):
    batch_input_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": description
        }
    )

    while batch.status == "validating" or batch.status == "in_progress" or batch.status == "finalizing" or batch.status == "cancelling":
        logging.info(f"[LLM] Batch in {batch.status}.")
        time.sleep(60)
        batch = client.batches.retrieve(batch.id)

    logging.info(f"[LLM] Batch in {batch.status}")
    logging.info(f"[LLM] Batch: {batch}")

    if batch.status != "completed":
        logging.error(f"[LLM] Batch error. {batch}")
        return None

    content = client.files.content(batch.output_file_id)
    logging.info(f"[LLM] Batch output: {content}")

    lines = content.text.split("\n")
    messages = []
    for line in lines:
        r = json.loads(line)
        if r["response"]["status_code"] != 200:
            logging.error(f"[LLM] Batch result error. {line}")
        messages.append(r["response"]["body"]["choices"][0]["message"]["content"])
    return messages
