from playwright.sync_api import sync_playwright

from agents.qa_agent import WebQAAgent

with WebQAAgent(
    "https://storage.googleapis.com/appsplain.firebasestorage.app/DSAD_t102_g1.html",
    False,
) as agent:
    print(agent.page.title())
    print(
        agent.chain.invoke(
            {
                "input": "Take a screenshot from the current page",
            }
        )
    )
    # agent.page.screenshot(path="sc.png", full_page=True)
