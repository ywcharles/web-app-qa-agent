from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

qa_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a QA assistant specialized in HTML, JavaScript and CSS applications."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])