from playwright.sync_api import sync_playwright
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

import base64
import os
import json
import re
from typing import List, Dict
from datetime import datetime

from prompts.qa_agent_prompt import qa_agent_prompt
from utils.screenshots import take_screenshot

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WebQAAgent:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY
        )
        self.screenshots_dir = os.path.join(ROOT_DIR, "screenshots")
        self.current_page = ""

    def navigate(self, url: str): 
        self.current_page = url
        self.page.goto(url)
        self.page_title = self.page.title()

    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
        self.page_title = self.page.title()

        @tool(
            description="Take a labeled screenshot of the current web page and return it for analysis, useful for testing an applications UI"
        )
        def screenshot_webapp(label: str = "state") -> HumanMessage:
            screenshot_path = take_screenshot(
                self.page, self.page_title, self.screenshots_dir, label
            )

            with open(screenshot_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            return HumanMessage(
                content=[
                    {"type": "text", "text": f"Screenshot labeled '{label}'"},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{b64_img}",
                    },
                ]
            )

        @tool(description="Grab the full HTML content of the current page")
        def grab_html() -> str:
            return self.page.content()

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.browser.close()
        self.playwright.stop()

    def generate_test_plan(self, max_steps: int = 10) -> str:
        response = self.chain.invoke(
            {
                "input": "Take a screenshot of the web application and parse its HTML content. "
                "Analyze the code and UI to create a comprehensive test plan. "
                f"Create a numbered list of specific with no more than {max_steps} actionable test tasks. "
                "Each task should be atomic and executable in sequence. "
                "Format your response as a clean numbered list, one test per line. "
                "Focus on testing key functionality, edge cases, and UI validation."
                "Return a numbered list, do not return any text besides the list."
                "Everything in the output must be a numbered step\n"
                "Output Example:\n"
                "1. Click the 'Add Node' button with value 5\n"
                "2. Verify node 5 appears in the tree\n"
                "3. Click the 'Delete Node' button\n\n"
            }
        )
         
        return response["output"]

    def parse_plan(self, plan: str) -> List[str]:
        steps = plan.split("\n")
        return steps

    def execute_plan(self, plan_steps: List[str]):
        task_outputs = []
        for step in plan_steps:
            response = self.chain.invoke(
                {
                    "input": 
                    f"Task: {step}\n\n"
                    "1. Execute the task\n"
                    "2. Take a screenshot of the post task execution state\n"
                    "3. Check the current HTML of the application\n"
                    "4. See if the result from the task execution was what you expected\n"
                    "5. Check for any UI issues in the new states"
                    "Output the following: \n"
                    "- Executed Task\n"
                    "- State was updated as expected: true or false"
                    "- UI contains flaws: true or false"
                    "- Comments in case state or UI has bugs"
                }
            )
            task_outputs.append(response)

        return task_outputs