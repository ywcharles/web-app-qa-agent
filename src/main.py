import os

from agents.qa_agent import WebQAAgent
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_file = os.path.join(project_root, "web_apps", "Insertion Sort_ The Bookshelf.html")

with WebQAAgent(
    False,
) as agent:
    agent.navigate(f"file://{html_file}",)
    response = agent.chain.invoke(
            {
                "input": "Take a screenshot of the web application and parse it's html content. "
                "Now analyze the code and the UI and come on with a plan on how you as an agent would test or debug this application. "
                "Create a list of specific actions you would do. "
                "Then execute your plan, taking screenshots at any state change like when you click a button and checking, and point out any UI errors or bugs you find",
            }
        )
    print(response["output"])
    # agent.page.screenshot(path="sc.png", full_page=True)
