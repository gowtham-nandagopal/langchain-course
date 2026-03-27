from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

# ============================================================
# STEP 1 — Define schema with Pydantic
# ============================================================

class WeatherInput(BaseModel):
    location: str = Field(..., description="City name")

# ============================================================
# STEP 2 — Define function with @tool decorator
# LangChain reads Pydantic schema automatically ✅
# No need to manually write tools=[{name, description...}]
# ============================================================

@tool(args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """Gets weather for a city"""  # ← LangChain reads this as description
    return f"Weather in {location}: 18°C and Sunny"

# ============================================================
# STEP 3 — Create LLM and bind tools
# Just one line — no manual schema writing ✅
# ============================================================

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=1.0)

# ============================================================
# STEP 4 — LangChain handles EVERYTHING automatically
# No manual tool_call parsing ✅
# No second LLM call ✅
# No feeding result back manually ✅
# ============================================================

tools = [get_weather]
agent = create_agent(model=llm, tools=tools)


result = agent.invoke({
    "messages": [
        {"role": "system",  "content": """You are a weather expert assistant. 
                                        Always respond in a friendly tone and 
                                        mention if the weather is good for 
                                        outdoor activities."""},
        {"role": "user", "content": "What is the weather in Paris?"}
    ]
})

# Print ALL messages — this is the scratchpad ✅
for message in result["messages"]:
    print(message)

# Output:
# HumanMessage:    "What is the weather in Paris?"
# AIMessage:       tool_call → get_weather(Paris)      ← LLM decision
# ToolMessage:     "Weather in Paris: 18°C and Sunny"  ← Tool result
# AIMessage:       "The weather in Paris is 18°C and Sunny!" ← Final answer

# Last message is always the final answer
print("\n", result["messages"][-1].content)
