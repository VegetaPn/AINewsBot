import logging
import datetime
import sys

from translation_agent import ContentTranslationAgent
from file_processing_agent import HTMLCleanAgent, HTML2MarkdownAgent
from web_browsing_agent import WebContentCheckAgent, WebContentFetchAgent


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='app.log',
        filemode='w'
    )
logger = logging.getLogger()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


if __name__ == '__main__':
    task_id = "debug_" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    archive_url = "https://buttondown.email/ainews/archive/"
    today = datetime.date.today() - datetime.timedelta(days=2)
    today = today.strftime('%B %d, %Y')

    web_content_check_agent = WebContentCheckAgent(task_id, archive_url, today)
    news_url = web_content_check_agent.execute()
    if not news_url:
        logging.warning(F"Not found news with date {today}")
        exit(0)

    web_content_fetch_agent = WebContentFetchAgent(task_id, news_url, {'class': 'email-body-content'}, "output/debug.html")
    html_clean_agent = HTMLCleanAgent(task_id, "output/debug.html", "output/debug_cleaned.html")
    html2markdown_agent = HTML2MarkdownAgent(task_id, "output/debug_cleaned.html", "output/debug.md")
    translation_agent = ContentTranslationAgent(task_id, "output/debug.md", "output/debug_translated.md")

    web_content_fetch_agent.execute()
    html_clean_agent.execute()
    html2markdown_agent.execute()
    translation_agent.execute()
    print("Done.")
