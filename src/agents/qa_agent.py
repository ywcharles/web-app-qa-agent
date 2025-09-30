from playwright.sync_api import sync_playwright
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv

import os

from prompts.qa_agent_prompt import qa_agent_prompt
from utils.screenshots import take_screenshot

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class WebQAAgent:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

        # Create screenshots directory structure
        self.screenshots_dir = os.path.join(ROOT_DIR, "screenshots")

    def navigate(self, url: str):
        self.page.goto(url)
        self.page_title = self.page.title()
        
    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
        self.page_title = self.page.title()

        @tool(description="Take a labeled screenshot of the current web page")
        def screenshot_webapp(label: str = "state") -> str:
            return f"Screenshot saved to {take_screenshot(self.page, self.page_title, self.screenshots_dir, label)}"
        
        @tool(description="Grab the full HTML content of the current page")
        def grab_html() -> str:
            return self.page.content()

        tools = [screenshot_webapp, grab_html]
        agent = create_tool_calling_agent(self.model, tools, qa_agent_prompt)
        self.chain = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.browser.close()
        self.playwright.stop()