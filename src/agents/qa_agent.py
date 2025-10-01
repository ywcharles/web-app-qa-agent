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
        self.test_plan = None
        self.test_results = []
        self.current_html_path = None
        self.original_html = None

    def navigate(self, url: str):
        # Store the file path if it's a local file
        if url.startswith("file://"):
            self.current_html_path = url.replace("file://", "")
            # Read and store original HTML
            with open(self.current_html_path, 'r', encoding='utf-8') as f:
                self.original_html = f.read()
        
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

    def generate_test_plan(self) -> str:
        """
        Generate a test plan for the current web application.
        Returns the plan as text and stores it internally.
        """
        print("=" * 60)
        print("GENERATING TEST PLAN")
        print("=" * 60 + "\n")

        response = self.chain.invoke(
            {
                "input": "Take a screenshot of the web application and parse its HTML content. "
                "Analyze the code and UI to create a comprehensive test plan. "
                "Create a numbered list of specific, actionable test tasks. "
                "Each task should be atomic and executable in sequence. "
                "Format your response as a clean numbered list, one test per line. "
                "Example:\n"
                "1. Click the 'Add Node' button with value 5\n"
                "2. Verify node 5 appears in the tree\n"
                "3. Click the 'Delete Node' button\n\n"
                "Focus on testing key functionality, edge cases, and UI validation."
            }
        )

        self.test_plan = response["output"]

        print("\nGENERATED TEST PLAN:")
        print("-" * 60)
        print(self.test_plan)
        print("=" * 60 + "\n")

        return self.test_plan

    def parse_plan(self, plan_text: str = None) -> List[Dict]:
        """
        Parse the test plan into structured steps.
        If no plan_text provided, uses self.test_plan.
        """
        if plan_text is None:
            plan_text = self.test_plan

        if not plan_text:
            return []

        steps = []
        lines = plan_text.split("\n")

        for line in lines:
            line = line.strip()
            # Look for numbered items (1. 2. etc.)
            if line and len(line) > 2 and line[0].isdigit() and "." in line[:4]:
                steps.append(
                    {
                        "description": line,
                        "completed": False,
                        "result": None,
                        "issues_found": [],
                    }
                )

        return steps

    def execute_step(self, step: Dict, step_number: int) -> Dict:
        """
        Execute a single test step.
        """
        print(f"\n{'='*60}")
        print(f"Executing Step {step_number}: {step['description']}")
        print(f"{'='*60}\n")

        response = self.chain.invoke(
            {
                "input": f"Execute this test step: {step['description']}\n\n"
                f"Instructions:\n"
                f"1. Take a screenshot labeled 'step_{step_number}_before'\n"
                f"2. Perform the action described in the step\n"
                f"3. Take a screenshot labeled 'step_{step_number}_after'\n"
                f"4. Verify the result by checking the HTML and screenshots\n"
                f"5. Report any bugs, issues, or unexpected behavior\n\n"
                f"Provide a clear summary of:\n"
                f"- What action you performed\n"
                f"- What you observed\n"
                f"- Whether the test passed or failed\n"
                f"- Any issues or bugs found"
            }
        )

        step["completed"] = True
        step["result"] = response["output"]

        # Simple check for issues in the response
        output_lower = response["output"].lower()
        if any(
            word in output_lower
            for word in ["bug", "issue", "error", "failed", "incorrect", "unexpected"]
        ):
            step["issues_found"].append(response["output"])

        return step

    def execute_plan(self, plan_text: str = None) -> List[Dict]:
        """
        Execute the test plan step by step.
        If no plan_text provided, uses self.test_plan.
        """
        print("\n" + "=" * 60)
        print("EXECUTING TEST PLAN")
        print("=" * 60 + "\n")

        test_steps = self.parse_plan(plan_text)

        if not test_steps:
            print("Warning: No test steps found to execute.")
            return []

        self.test_results = []

        for i, step in enumerate(test_steps, 1):
            try:
                executed_step = self.execute_step(step, i)
                self.test_results.append(executed_step)

            except Exception as e:
                print(f"\nError executing step {i}: {str(e)}")
                step["completed"] = False
                step["result"] = f"Error: {str(e)}"
                step["issues_found"].append(f"Execution error: {str(e)}")
                self.test_results.append(step)

        return self.test_results

    def generate_summary(self) -> Dict:
        """
        Generate a summary of the test execution.
        """
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60 + "\n")

        total_steps = len(self.test_results)
        completed_steps = sum(1 for r in self.test_results if r["completed"])
        failed_steps = total_steps - completed_steps

        all_issues = []
        for result in self.test_results:
            all_issues.extend(result.get("issues_found", []))

        summary = {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "total_issues_found": len(all_issues),
            "issues": all_issues,
        }

        print(f"Total Steps: {total_steps}")
        print(f"Completed: {completed_steps}")
        print(f"Failed: {failed_steps}")
        print(f"Issues Found: {len(all_issues)}")

        if all_issues:
            print("\nIssues Detected:")
            for i, issue in enumerate(all_issues, 1):
                print(f"\n{i}. {issue[:200]}...")

        print("\n" + "=" * 60)

        return summary

    def analyze_bugs(self) -> Dict:
        """
        Analyze the bugs found during testing to prepare for fixing.
        """
        print("\n" + "=" * 60)
        print("ANALYZING BUGS")
        print("=" * 60 + "\n")

        if not self.test_results:
            print("No test results available for bug analysis.")
            return {"bugs_found": False, "analysis": None}

        all_issues = []
        for result in self.test_results:
            all_issues.extend(result.get("issues_found", []))

        if not all_issues:
            print("No bugs found during testing!")
            return {"bugs_found": False, "analysis": None}

        # Compile all issues into a single report
        issues_report = "\n\n".join([f"Issue {i+1}:\n{issue}" for i, issue in enumerate(all_issues)])

        print(f"Analyzing {len(all_issues)} issue(s)...")
        
        # Use the LLM to analyze bugs without tools
        analysis_prompt = f"""You are analyzing bugs found in a web application during automated testing.

Here is the original HTML code:
{self.original_html[:10000]}  # Truncate if too long

Here are the bugs found during testing:
{issues_report}

Please analyze these bugs and provide:
1. Root cause analysis for each bug
2. Specific code changes needed to fix each bug
3. Priority level (High/Medium/Low) for each bug

Format your response as a structured analysis."""

        response = self.model.invoke(analysis_prompt)
        analysis = response.content

        print("\nBUG ANALYSIS:")
        print("-" * 60)
        print(analysis)
        print("=" * 60 + "\n")

        return {
            "bugs_found": True,
            "analysis": analysis,
            "issues_list": all_issues
        }

    def fix_bugs(self, bug_analysis: Dict) -> str:
        """
        Generate fixed HTML based on bug analysis.
        Returns the path to the fixed HTML file.
        """
        print("\n" + "=" * 60)
        print("GENERATING BUG FIXES")
        print("=" * 60 + "\n")

        if not bug_analysis.get("bugs_found"):
            print("No bugs to fix!")
            return None

        if not self.original_html or not self.current_html_path:
            print("Error: Original HTML not available for fixing.")
            return None

        # Use the LLM to generate fixed HTML
        fix_prompt = f"""You are a web developer fixing bugs in an HTML application.

Here is the ORIGINAL HTML CODE:
```html
{self.original_html}
```

Here is the BUG ANALYSIS:
{bug_analysis['analysis']}

Please provide the COMPLETE FIXED HTML CODE with all bugs corrected.
Output ONLY the complete HTML code, no explanations or markdown formatting.
Make sure to preserve all functionality while fixing the identified bugs."""

        print("Generating fixed HTML code...")
        response = self.model.invoke(fix_prompt)
        fixed_html = response.content

        # Clean up the response (remove markdown code blocks if present)
        fixed_html = re.sub(r'^```html\s*\n', '', fixed_html)
        fixed_html = re.sub(r'\n```\s*$', '', fixed_html)
        fixed_html = fixed_html.strip()

        # Generate filename for fixed HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = os.path.basename(self.current_html_path)
        name_without_ext = os.path.splitext(original_name)[0]
        fixed_filename = f"{name_without_ext}_FIXED_{timestamp}.html"
        fixed_path = os.path.join(os.path.dirname(self.current_html_path), fixed_filename)

        # Save fixed HTML
        with open(fixed_path, 'w', encoding='utf-8') as f:
            f.write(fixed_html)

        print(f"\n✓ Fixed HTML saved to: {fixed_path}")
        print("=" * 60 + "\n")

        return fixed_path

    def run_full_test(self) -> Dict:
        """
        Complete workflow: generate plan, execute it, analyze bugs, fix them, and return results.
        """
        # Stage 1: Generate plan
        self.generate_test_plan()

        # Stage 2: Execute plan
        results = self.execute_plan()

        # Stage 3: Generate summary
        summary = self.generate_summary()

        # Stage 4: Analyze bugs
        bug_analysis = self.analyze_bugs()

        # Stage 5: Fix bugs if any were found
        fixed_file_path = None
        if bug_analysis.get("bugs_found"):
            fixed_file_path = self.fix_bugs(bug_analysis)

        return {
            "plan": self.test_plan,
            "results": results,
            "summary": summary,
            "bug_analysis": bug_analysis,
            "fixed_file": fixed_file_path
        }

    def save_results(self, filename: str = "test_results.json"):
        """
        Save test results to a JSON file.
        """
        results_file = os.path.join(ROOT_DIR, filename)

        output = {
            "plan": self.test_plan,
            "results": self.test_results,
            "summary": self.generate_summary() if self.test_results else None,
        }

        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {results_file}")
        return results_file