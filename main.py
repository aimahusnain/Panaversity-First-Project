import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from tavily import TavilyClient

# ------------------------------
# Environment Setup
# ------------------------------
load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY", "")
tavily_api_key = os.getenv("TAVILY_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# ------------------------------
# Models
# ------------------------------
flash_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client)
pro_model = OpenAIChatCompletionsModel(model="gemini-2.5-pro", openai_client=external_client)

# ------------------------------
# User Profile
# ------------------------------
user_profile = {
    "name": "Husnain",
    "city": "Rawalpindi",
    "topic": "AI",
    "preferences": {
        "summary_style": "concise",
        "include_links_only": False,
        "max_results": 5
    },
    "last_query": "",
    "notes": "Developer"
}

# ------------------------------
# Structured Logging
# ------------------------------
log_history = []

def structured_log(agent_name, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": timestamp, "agent": agent_name, "message": message}
    log_history.append(entry)
    print(f"[{timestamp}] [{agent_name}] {message}")

# ------------------------------
# Tools
# ------------------------------
@function_tool
def web_search(query: str) -> list:
    """Perform web search using Tavily and return structured results."""
    try:
        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(query)
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("snippet"),
                "source_type": "unknown"
            })
        structured_log("SearchAgent", f"Found {len(results)} results for '{query}'")
        return results
    except Exception as e:
        structured_log("SearchAgent", f"Web search failed: {e}")
        return []

@function_tool
def dynamic_instructions(instruction_type: str) -> str:
    if instruction_type.lower() == "search deeper":
        return "Increase the number of sources and give more detailed summaries."
    elif instruction_type.lower() == "give me just links":
        return "Provide only the links without summaries."
    return ""

@function_tool
def log_status(agent_name: str, message: str) -> str:
    structured_log(agent_name, message)
    return f"Logged: {message}"

# ------------------------------
# Agents
# ------------------------------
requirement_agent = Agent(
    name="RequirementGatheringAgent",
    instructions=f"""
        Interact with the user.
        - Ask clarifying questions until requirements are fully understood.
        - Personalize context for {user_profile['name']} from {user_profile['city']} interested in {user_profile['topic']}.
        - Delegate to PlanningAgent.
    """,
    model=flash_model,
    tools=[log_status, dynamic_instructions]
)

planning_agent = Agent(
    name="PlanningAgent",
    instructions=f"""
        Break broad queries into structured research tasks.
        - Keep {user_profile['name']}'s preferences in mind.
        - Pass structured tasks to LeadResearchAgent.
    """,
    model=flash_model,
    tools=[log_status, dynamic_instructions]
)

search_agent = Agent(
    name="SearchAgent",
    instructions=f"""
        Find relevant sources using Tavily.
        - Return structured results with title, URL, snippet.
        - Personalize results for {user_profile['name']}.
    """,
    model=flash_model,
    tools=[web_search, log_status, dynamic_instructions]
)

source_checker_agent = Agent(
    name="SourceCheckerAgent",
    instructions="""
        Evaluate source credibility:
        - High: .gov, .edu, major news orgs
        - Medium: Wikipedia, industry sites
        - Low: blogs, forums
        - Update 'source_type' in results
    """,
    model=flash_model,
    tools=[log_status]
)

reflection_agent = Agent(
    name="ReflectionAgent",
    instructions="""
        Detect conflicts and supporting evidence between sources.
        - Highlight contradictions explicitly.
        - Include confidence level per claim.
    """,
    model=flash_model,
    tools=[log_status]
)

synthesis_agent = Agent(
    name="SynthesisAgent",
    instructions=f"""
        Organize research findings into structured report.
        - Group by themes/trends/insights.
        - Explain relationships and implications.
        - Personalize for {user_profile['name']}.
    """,
    model=pro_model,
    tools=[log_status]
)

citations_agent = Agent(
    name="CitationsAgent",
    instructions="""
        Attach inline citations [1],[2] and a full reference list.
        - Use URLs from search results.
    """,
    model=flash_model,
    tools=[log_status]
)

lead_research_agent = Agent(
    name="LeadResearchAgent",
    instructions="""
        Orchestrate research workflow:
        1. Execute SearchAgent
        2. Run SourceCheckerAgent
        3. Run ReflectionAgent
        4. Run SynthesisAgent
        5. Run CitationsAgent
    """,
    model=pro_model,
    tools=[
        search_agent.as_tool("search_agent", "Performs web searches"),
        source_checker_agent.as_tool("source_checker", "Rates credibility"),
        reflection_agent.as_tool("reflection_agent", "Highlights conflicts/support"),
        synthesis_agent.as_tool("synthesis_agent", "Organizes insights"),
        citations_agent.as_tool("citations_agent", "Adds citations"),
        log_status
    ]
)

# ------------------------------
# Handoffs
# ------------------------------
requirement_agent.handoffs = [planning_agent]
planning_agent.handoffs = [lead_research_agent]

# ------------------------------
# Parallel Search with Retry
# ------------------------------
async def retry_async(func, *args, retries=3, delay=2, **kwargs):
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                raise e

async def parallel_search(tasks):
    coroutines = [retry_async(Runner.run_async, search_agent, task) for task in tasks]
    results = await asyncio.wait_for(asyncio.gather(*coroutines), timeout=30)
    return results

# ------------------------------
# Run Research
# ------------------------------
def run_research(query: str):
    try:
        print("="*60)
        print("🚀 Deep Research Agent Workflow")
        print("="*60)

        if "search deeper" in query.lower() or "just links" in query.lower():
            log_status("System", f"Dynamic instruction detected: {query}")

        # Step 1: Requirement agent
        structured_task_output = Runner.run_sync(requirement_agent, query)
        subtasks = getattr(structured_task_output, "final_output", [query])

        # Step 2: LeadResearchAgent handles full workflow
        lead_output = Runner.run_sync(lead_research_agent, subtasks)

        # Step 3: Print final report
        print("\n" + "="*60)
        print("✅ Final Research Report")
        print("="*60)
        if hasattr(lead_output, "final_output"):
            print(lead_output.final_output)

        # Step 4: Structured logs
        print("\n" + "="*60)
        print("📑 Structured Agent Logs")
        print("="*60)
        for entry in log_history:
            print(f"[{entry['timestamp']}] [{entry['agent']}] {entry['message']}")

        return lead_output.final_output if hasattr(lead_output, "final_output") else None

    except Exception as e:
        print(f"❌ Error: {e}")
        return "I'm sorry but I cannot respond at the moment. Please try again."

# ------------------------------
# Example Run
# ------------------------------
if __name__ == "__main__":
    query = "Compare renewable energy policies in the US, Germany, and China"
    run_research(query)