from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient

tavily = TavilyClient()

class Source(BaseModel):
    """
    Schema for a source used in the agent's response, including the URL of the source.
    """
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for the agent's response, including the final answer and the sources used to arrive at that answer.
    """
    answer: str = Field(description="The final answer to the user's query, based on the content and sources")
    sources: List[Source] = Field(default_factory=list,description="A list of sources used to arrive at the final answer")

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

# llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=1.0)
tools = [search]

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
    system_prompt="You are a helpful assistant that can use tools to answer questions. Always use the search tool for current information."
)

agent_response = agent.invoke(
    {"messages": [HumanMessage(content="Search for 3 job openings for AI engineers in San Francisco.")]}
)
print(agent_response["messages"][-1].content)
