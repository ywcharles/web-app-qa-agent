import os

from agents.qa_agent import WebQAAgent
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_file = os.path.join(project_root, "web_apps", "Insertion Sort_ The Bookshelf.html")

with WebQAAgent(
    False,
) as agent:
    agent.navigate(f"file://{html_file}",)
    print(
        agent.chain.invoke(
            {
                "input": "Take a screenshot from the current page and also extract it's html",
            }
        )
    )
    # agent.page.screenshot(path="sc.png", full_page=True)
