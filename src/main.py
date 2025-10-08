import os
from agents.qa_agent import WebQAAgent
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
web_apps_dir = os.path.join(project_root, "web_apps")
output_dir = os.path.join(project_root, "fixed-webapps-10-08")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all HTML files in web_apps directory
html_files = [f for f in os.listdir(web_apps_dir) if f.endswith('.html')]

print(f"Found {len(html_files)} HTML files to process\n")

with WebQAAgent(False) as agent:
    for html_file in html_files:
        print(f"Processing: {html_file}")
        print("-" * 80)
        
        # Full path to input file
        input_path = os.path.join(web_apps_dir, html_file)
        
        # Navigate to the file
        agent.navigate(f"file://{input_path}")
        
        # Generate improved HTML
        response = agent.generate_improved_html()
        report, html = response.values()
        
        # Print report
        print(report)
        print(html)
        
        # Save improved HTML to output directory
        output_path = os.path.join(output_dir, html_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"Saved to: {output_path}")
        print("=" * 80)
        print()