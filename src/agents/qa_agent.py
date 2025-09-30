from playwright.sync_api import sync_playwright
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

import base64
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

        @tool(description="Take a labeled screenshot of the current web page and return it for analysis")
        def screenshot_webapp(label: str = "state") -> HumanMessage:
            screenshot_path = take_screenshot(self.page, self.page_title, self.screenshots_dir, label)
            
            # Encode the screenshot as base64
            with open(screenshot_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            # Return both label and the screenshot as a multimodal input
            return HumanMessage(
                content=[
                    {"type": "text", "text": f"Screenshot labeled '{label}'"},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{b64_img}"}
                ]
            )
        
        @tool(description="Grab the full HTML content of the current page")
        def grab_html() -> str:
            return self.page.content()

        tools = [screenshot_webapp, grab_html]
        agent = create_tool_calling_agent(self.model, tools, qa_agent_prompt)
        self.chain = AgentExecutor(agent=agent, tools=tools, verbose=True)

        @tool(description="Click an element by selector")
        def click_element(selector: str) -> str:
            try:
                self.page.click(selector, timeout=5000)
                return f"Successfully clicked element: {selector}"
            except Exception as e:
                return f"Failed to click element {selector}: {str(e)}"
        
        @tool(description="Fill a form input by selector")
        def fill_input(selector: str, value: str) -> str:
            try:
                self.page.fill(selector, value, timeout=5000)
                return f"Successfully filled {selector} with: {value}"
            except Exception as e:
                return f"Failed to fill {selector}: {str(e)}"
        
        @tool(description="Get text content from an element")
        def get_text(selector: str) -> str:
            try:
                text = self.page.text_content(selector, timeout=5000)
                return f"Text content: {text}"
            except Exception as e:
                return f"Failed to get text from {selector}: {str(e)}"

        tools = [screenshot_webapp, grab_html, click_element, fill_input, get_text]
        agent = create_tool_calling_agent(self.model, tools, qa_agent_prompt)
        self.chain = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        return self
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.browser.close()
        self.playwright.stop()