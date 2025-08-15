from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, RunContextWrapper
import os
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

_: bool = load_dotenv(find_dotenv())  # Load environment variables from .env file
gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Which LLM Model?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

lite_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash-lite",
    openai_client=external_client
)

special_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-pro",
    openai_client=external_client
)

planning_agent: Agent = Agent(
    name="PlanningAgent",
    instructions="You are a planning agent. Look at user request and use scientific reasoning to plan the next steps. In response always include the scientific principles you used.",
    model=special_model,
)

web_search_agent: Agent = Agent(
    name="WebSearchAgent",
    instructions="You are a web search assistant. Look at user request, plan and use web search to find relevant information",
    model=llm_model
)

reflective_agent: Agent = Agent(
    name="ReflectiveAgent",
    model=lite_model,
    instructions="You are a reflective assistant. Look at user request and reflective on the best approuch to take."
)

@function_tool
def current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def dynamic_instructions(context: RunContextWrapper, agent: Agent) -> str:
    
    dynamic_date=datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
    Your main goal is deep search for each user query.
    
    
    We follow a structured process for each deep seach request.
    1. If work is related to Planing then delegate the work to PlanningAgent.
    2. spawn multiple web search agents 
    3. perform reflection to decide if deep goal is achieved.
    
    Finally get reflection to know if the task is achieved. Current Date {dynamic_date}
    """
    
    return prompt

orchestrator_agent: Agent = Agent(
    name="DeepAgent",
    instructions=dynamic_instructions,
    model=special_model,
    tools=[
        current_date,
        planning_agent.as_tool(
            tool_name="planning_agent", 
            tool_description="A planning agent that uses scientific reasoning to plan next steps."
        ),
        web_search_agent.as_tool(
            tool_name="web_search_tool", 
            tool_description="A web search agent that finds relevant information on the web."
        ),
        reflective_agent.as_tool(
            tool_name="ReflectiveAgent", 
            tool_description="A web search agent that finds relevant information on the web."
        ),
    ],
    handoffs=[planning_agent]
)


res = Runner.run_sync(orchestrator_agent, "Is OK full form is UnKnown?")
print(res.final_output)