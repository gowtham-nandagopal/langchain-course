from dotenv import load_dotenv
import os
load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Search the internet for real-time information such as weather,
    news, current events, or any factual query requiring up-to-date data.
    Always use this tool when asked about current conditions or recent events.
    Args:
        query (str): The search query string
    Returns:
        str: The search results
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
tools = [search]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that can use tools to answer questions. Always use the search tool for current information."
)

agent_response = agent.invoke(
    {"messages": [HumanMessage(content="Search for 3 job openings for AI engineers in San Francisco.")]}
)
print(agent_response["messages"][-1].content)
