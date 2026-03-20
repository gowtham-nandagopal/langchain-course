from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0.7,
    max_tokens=1024,
    convert_system_message_to_human=True,
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of Tamil Nadu?")
]

response = llm.invoke(messages)
print(response.content)