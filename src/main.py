import os
from agents.qa_agent import WebQAAgent

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# html_file = os.path.join(project_root, "web_apps", "Interactive Search Tree Visualization.html")
html_file = os.path.join(project_root, "web_apps", "Insertion Sort_ The Bookshelf.html")
# html_file = os.path.join(project_root, "web_apps", "Interactive Insertion Sort.html")
# html_file = os.path.join(project_root, "web_apps", "Interactive Binary Search Tree.html")


with WebQAAgent(False) as agent:
    agent.navigate(f"file://{html_file}")
    # plan = agent.generate_test_plan(max_steps=5)
    # print(plan)
    # plan_steps = agent.parse_plan(plan)
    # print(plan_steps)
    # steps_outputs = agent.execute_plan(plan_steps=plan_steps)
    # for s in steps_outputs:
    #     print(s)
    
    ui_analysis = agent.analyzed_ui()
    print(ui_analysis)