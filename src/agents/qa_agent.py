from playwright.sync_api import sync_playwright
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv

import os

from prompts.qa_agent_prompt import qa_agent_prompt

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class WebQAAgent:
    def __init__(self, url: str, headless: bool = True):
        self.url = url
        self.headless = headless
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
        self.page.goto(self.url)

        @tool(description="Take a screenshot of the current web page")
        def screenshot_webapp(file_path: str = "screenshot.png") -> str:
            """Take a screenshot of the current web page and save it to a file."""
            self.page.screenshot(path=file_path, full_page=True)
            return f"Screenshot saved to {file_path}"

        tools = [screenshot_webapp]
        agent = create_tool_calling_agent(self.model, tools, qa_agent_prompt)
        self.chain = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.browser.close()
        self.playwright.stop()