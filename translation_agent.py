import logging
import re
import time

import llm


class ContentTranslationAgent:
    def __init__(self, task_id, input_file_path, output_file_path) -> None:
        self.input_file_path = input_file_path
        self.output_path = output_file_path
        self.system_prompt = '''你是一位精通简体中文的专业翻译，曾参与《纽约时报》和《经济学人》中文版的翻译工作，因此对于新闻和时事文章的翻译有深入的理解。我希望你能帮我将以下英文新闻段落翻译成中文，风格与上述杂志的中文版相似。
        规则：
        - 翻译时要准确传达新闻事实和背景。
        - 即使意译也要保留原始段落格式，以及保留术语，例如LLM，FLAC，JPEG 等。保留公司缩写，例如 Microsoft, Amazon, OpenAI 等。
        - 人名不翻译。
        - 同时要保留引用的论文，例如 [20] 这样的引用。
        - 对于 Figure 和 Table，翻译的同时保留原有格式，例如：“Figure 1: ”翻译为“图 1: ”，“Table 1: ”翻译为：“表 1: ”。
        - 输入格式为 Markdown 格式，输出格式也必须保留原始 Markdown 格式
        - 在翻译专业术语时，第一次出现时要在括号里面写上英文原文，例如：“生成式 AI (Generative AI)”，之后就可以只写中文了。
        - 以下是常见的 AI 相关术语词汇对应表（English -> 中文）：
            * Transformer -> Transformer
            * Token -> Token
            * LLM/Large Language Model -> 大语言模型
            * Zero-shot -> 零样本
            * Few-shot -> 少样本
            * AI Agent -> AI 智能体
            * AGI -> 通用人工智能
            * Token -> Token
            * Stability.ai -> Stability.ai
            * LlaMa -> LlaMa
        策略：
        分三步进行翻译工作，并打印每步的结果：
        1. 根据英文内容直译，保持原有格式，不要遗漏任何信息。
        2. 根据第一步直译的结果，指出其中存在的具体问题，要准确描述，不宜笼统的表示，也不需要增加原文不存在的内容或格式，包括不仅限于：
            - 不符合中文表达习惯，明确指出不符合的地方。
            - 语句不通顺，指出位置，不需要给出修改意见，意译时修复。
            - 晦涩难懂，不易理解，可以尝试给出解释。
        3. 根据第一步直译的结果和第二步指出的问题，重新进行意译，保证内容的原意的基础上，使其更易于理解，更符合中文的表达习惯，同时保持原有的格式不变。
        返回格式如下，"{xxx}"表示占位符：
        ### 直译
        {直译结果}
        
        ***
        
        ### 问题
        {直译的具体问题列表}
        
        ***
        
        ### 意译
        ```
        {意译结果}
        ```
        
        现在请按照上面的要求从第一行开始翻译以下内容为简体中文：
        '''
        self.translated_pattern = r'### 意译\n```(.*?)```'

    def execute(self):
        with open(self.input_file_path, 'r', encoding='utf-8') as input_file:
            content = input_file.read()

        if content is None:
            logging.error(f"[Content Translate Agent] Input is None.")
            return

        paragraphs = content.split("\n\n\n")
        length = len(paragraphs)
        logging.info(f"[Content Translate Agent] content length: {length}")

        round = 0
        for paragraph in paragraphs:
            round += 1
            logging.info(f"[Content Translate Agent] Current round: {round}")

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": paragraph}
            ]
            logging.info(f"[Content Translate Agent] REQ: {messages}")
            ai_message = llm.chat(messages)
            logging.info(f"[Content Translate Agent] AI: {ai_message}")

            translated = self.parse_message(ai_message)
            if translated is None:
                logging.error(f"[Content Translate Agent] Translated is None.")
                continue

            logging.info(f"[Content Translate Agent] Translated: {translated}")
            with open(self.output_path, 'a') as file:
                file.write(translated + "\n")
            logging.info(f"[Content Translate Agent] Written to file.")
            time.sleep(1)

    def parse_message(self, ai_message):
        match = re.search(self.translated_pattern, ai_message, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1)
        else:
            return None
