import logging

import markdownify
from bs4 import BeautifulSoup


class HTMLCleanAgent:
    def __init__(self, task_id, input_file_path, output_file_path) -> None:
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path

    def execute(self):
        with open(self.input_file_path, 'r') as input_file:
            html = input_file.read()
            soup = BeautifulSoup(html, 'lxml')
            h1 = soup.find('h1', id="part-2-detailed-by-channel-summaries-and-links")
            if not h1:
                logging.error(f"[HTMLCleanAgent] Not found context.")
                return

            next_sibling = h1.find_next_sibling()
            while next_sibling:
                sibling_to_remove = next_sibling
                next_sibling = sibling_to_remove.next_sibling
                sibling_to_remove.extract()
            h1.extract()

        with open(self.output_file_path, 'w') as output_file:
            output_file.write(soup.prettify())
        logging.info(f"[HTMLCleanAgent] Written file.")


class HTML2MarkdownAgent:
    def __init__(self, task_id, input_file_path, output_file_path) -> None:
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path

    def execute(self):
        with open(self.input_file_path, 'r') as input_file:
            html = input_file.read()
            markdown = markdownify.markdownify(html)

        if markdown is None:
            logging.error(f"[HTML2MarkdownAgent] Returned None.")
            return

        with open(self.output_file_path, 'w') as output_file:
            output_file.write(markdown)

        logging.info(f"[HTML2MarkdownAgent] Written file.")
