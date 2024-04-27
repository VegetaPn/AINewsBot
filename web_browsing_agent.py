import logging

import requests
from bs4 import BeautifulSoup


class WebContentCheckAgent:
    def __init__(self, task_id, url, target_date) -> None:
        self.url = url
        self.target_date = target_date

    def execute(self):

        response = requests.get(self.url)
        html_content = response.content

        soup = BeautifulSoup(html_content, 'html.parser')
        element_list = soup.find('div', class_='email-list')

        for element in element_list.find_all('a'):
            email_content = element.get_text()
            if self.target_date in email_content:
                link = element.get('href')
                if link and link.startswith("http"):
                    logging.info(f"[WebContentCheckAgent] Found link with date {self.target_date}: {link}")
                    return link

        print(f"[WebContentCheckAgent] Did not find link with date {self.target_date}.")
        return None


class WebContentFetchAgent:
    def __init__(self, task_id, url, target_desc, output_file_path) -> None:
        self.url = url
        self.target_desc = target_desc
        self.output_file_path = output_file_path

    def execute(self):
        response = requests.get(self.url)

        soup = BeautifulSoup(response.content, 'html.parser')
        target = soup.find('div', self.target_desc)

        if not target:
            logging.error("[WebContentFetchAgent] Target not found.")
            return

        with open(self.output_file_path, 'w') as output_file:
            output_file.write(str(target))
