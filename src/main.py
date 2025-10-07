import os
from agents.qa_agent import WebQAAgent

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# html_file = os.path.join(project_root, "web_apps", "Interactive Search Tree Visualization.html")
html_file = os.path.join(project_root, "web_apps", "Insertion Sort_ The Bookshelf.html")
# html_file = os.path.join(project_root, "web_apps", "Interactive Insertion Sort.html")
# html_file = os.path.join(project_root, "web_apps", "Interactive Binary Search Tree.html")


with WebQAAgent(False) as agent:
    agent.navigate(f"file://{html_file}")
    # TODO do the entire folder
    response = agent.generate_improved_html()
    report, html = response.values()
    print(report)
    print(html)