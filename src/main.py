import os
from agents.qa_agent import WebQAAgent

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_file = os.path.join(project_root, "web_apps", "Interactive Search Tree Visualization.html")

with WebQAAgent(False) as agent:
    agent.navigate(f"file://{html_file}")
    results = agent.run_full_test()
    
    # Check results
    if results['fixed_file']:
        print(f"Fixed file created: {results['fixed_file']}")