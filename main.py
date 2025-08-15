from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
import os
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
from tavily import AsyncTavilyClient
import json
import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

_: bool = load_dotenv(find_dotenv())

# ------------------------------
# Environment Setup
# ------------------------------
gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
tavily_api_key = os.getenv("TAVILY_API_KEY", "")
tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

user_info = {
    "name": "Muhammad Husnain",
    "city": "Pakistan"
}

# ------------------------------
# Professional Citation System
# ------------------------------
class CitationManager:
    def __init__(self):
        self.sources: List[Dict[str, Any]] = []
        self.citation_counter = 0
        
    def add_source(self, source_data: Dict[str, Any]) -> int:
        """Add a source and return its citation number."""
        self.citation_counter += 1
        citation_entry = {
            "id": self.citation_counter,
            "title": source_data.get("title", "Unknown Title"),
            "url": source_data.get("url", ""),
            "content": source_data.get("content", ""),
            "published_date": source_data.get("published_date", ""),
            "domain": urlparse(source_data.get("url", "")).netloc,
            "accessed_date": datetime.now().strftime("%Y-%m-%d"),
            "score": source_data.get("score", 0),
            "raw_content": source_data.get("raw_content", "")
        }
        self.sources.append(citation_entry)
        return self.citation_counter
    
    def get_inline_citation(self, source_ids: List[int]) -> str:
        """Generate inline citation in format [1,2,3]."""
        if not source_ids:
            return ""
        return f"[{','.join(map(str, source_ids))}]"
    
    def generate_bibliography(self, style: str = "apa") -> str:
        """Generate bibliography in specified style."""
        if not self.sources:
            return "No sources cited."
            
        bibliography = "\n## References\n\n"
        
        for source in self.sources:
            if style.lower() == "apa":
                citation = self._format_apa_citation(source)
            elif style.lower() == "mla":
                citation = self._format_mla_citation(source)
            elif style.lower() == "chicago":
                citation = self._format_chicago_citation(source)
            else:
                citation = self._format_web_citation(source)
            
            bibliography += f"[{source['id']}] {citation}\n\n"
        
        return bibliography
    
    def _format_apa_citation(self, source: Dict[str, Any]) -> str:
        """Format citation in APA style."""
        title = source["title"]
        url = source["url"]
        domain = source["domain"]
        accessed_date = source["accessed_date"]
        published_date = source.get("published_date", "n.d.")
        
        # Clean title
        title = re.sub(r'\s+', ' ', title).strip()
        
        if published_date and published_date != "n.d.":
            date_part = f"({published_date})"
        else:
            date_part = "(n.d.)"
        
        return f"{title}. {date_part}. {domain}. Retrieved {accessed_date}, from {url}"
    
    def _format_mla_citation(self, source: Dict[str, Any]) -> str:
        """Format citation in MLA style."""
        title = source["title"]
        url = source["url"]
        domain = source["domain"]
        accessed_date = source["accessed_date"]
        
        # Convert date to MLA format
        try:
            date_obj = datetime.strptime(accessed_date, "%Y-%m-%d")
            mla_date = date_obj.strftime("%d %b %Y")
        except:
            mla_date = accessed_date
        
        return f'"{title}." {domain}, Web. {mla_date}. <{url}>.'
    
    def _format_chicago_citation(self, source: Dict[str, Any]) -> str:
        """Format citation in Chicago style."""
        title = source["title"]
        url = source["url"]
        domain = source["domain"]
        accessed_date = source["accessed_date"]
        
        return f'"{title}." {domain}. Accessed {accessed_date}. {url}.'
    
    def _format_web_citation(self, source: Dict[str, Any]) -> str:
        """Format basic web citation."""
        title = source["title"]
        url = source["url"]
        domain = source["domain"]
        accessed_date = source["accessed_date"]
        
        return f"{title}. {domain}. Accessed: {accessed_date}. URL: {url}"
    
    def get_source_summary(self) -> str:
        """Get summary of all sources."""
        if not self.sources:
            return "No sources found."
        
        summary = f"\n## Source Summary\n\n"
        summary += f"Total sources: {len(self.sources)}\n"
        
        # Group by domain
        domains = {}
        for source in self.sources:
            domain = source["domain"]
            if domain not in domains:
                domains[domain] = 0
            domains[domain] += 1
        
        summary += f"Unique domains: {len(domains)}\n"
        summary += "Domain distribution:\n"
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            summary += f"  - {domain}: {count} sources\n"
        
        return summary

# Global citation manager
citation_manager = CitationManager()

# ------------------------------
# Models
# ------------------------------
flash_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client)
pro_model = OpenAIChatCompletionsModel(model="gemini-2.5-pro", openai_client=external_client)

# ------------------------------
# Enhanced Tools
# ------------------------------
@function_tool
async def tavily_search_with_citations(query: str, max_results: int = 5) -> dict:
    """Search the web for current information and automatically register citations."""
    print(f"🔍 SEARCHING WEB: {query}")
    try:
        results = await tavily_client.search(query=query, max_results=max_results)
        
        # Process results and add citations
        search_results = results.get('results', [])
        cited_results = []
        
        for result in search_results:
            # Add to citation manager
            citation_id = citation_manager.add_source(result)
            
            # Add citation ID to result
            result['citation_id'] = citation_id
            cited_results.append(result)
        
        print(f"✅ SEARCH COMPLETE: Found {len(cited_results)} results with citations")
        
        return {
            "results": cited_results,
            "query": query,
            "total_results": len(cited_results)
        }
        
    except Exception as e:
        print(f"❌ SEARCH FAILED: {str(e)}")
        return {"error": str(e)}

@function_tool
def add_manual_citation(title: str, url: str, content: str = "", published_date: str = "") -> str:
    """Manually add a citation source."""
    source_data = {
        "title": title,
        "url": url,
        "content": content,
        "published_date": published_date
    }
    citation_id = citation_manager.add_source(source_data)
    return f"Added citation [{citation_id}]: {title}"

@function_tool
def get_inline_citation(source_ids: str) -> str:
    """Get properly formatted inline citation for given source IDs (comma-separated)."""
    try:
        ids = [int(x.strip()) for x in source_ids.split(',')]
        return citation_manager.get_inline_citation(ids)
    except:
        return f"[Error: Invalid source IDs: {source_ids}]"

@function_tool
def generate_bibliography(style: str = "apa") -> str:
    """Generate complete bibliography in specified style (apa, mla, chicago, or web)."""
    return citation_manager.generate_bibliography(style)

@function_tool
def get_citation_summary() -> str:
    """Get summary of all citations collected."""
    return citation_manager.get_source_summary()

@function_tool
def log_status(message: str) -> str:
    """Log current status to console."""
    print(f"📋 STATUS: {message}")
    return f"Logged: {message}"

# ------------------------------
# Enhanced Agents with Citation Support
# ------------------------------

# 1. Planning Agent
planning_agent = Agent(
    name="PlanningAgent",
    instructions="""Create a comprehensive step-by-step research plan for the user's request.
    Consider what types of sources will be needed and how they should be cited.
    Always use log_status to communicate your planning process.""",
    model=pro_model,
    tools=[log_status]
)

# 2. Enhanced Search Agent  
search_agent = Agent(
    name="SearchAgent", 
    instructions="""Search the web for relevant information using tavily_search_with_citations.
    This tool automatically creates citations for all sources found.
    Focus on finding credible, recent sources. Use log_status to update on search progress.
    Always mention the citation IDs when referring to specific sources.""",
    model=flash_model,
    tools=[tavily_search_with_citations, log_status, add_manual_citation]
)

# 3. Enhanced Citations Agent
citations_agent = Agent(
    name="CitationsAgent",
    instructions="""You are a professional citation specialist. Your responsibilities:
    1. Review all sources for credibility and relevance
    2. Generate properly formatted inline citations using get_inline_citation
    3. Create professional bibliographies using generate_bibliography
    4. Ensure all claims are properly cited
    5. Recommend citation style based on research topic
    
    Use academic standards and be meticulous about citation accuracy.
    Always use log_status to communicate your citation work.""",
    model=pro_model,
    tools=[get_inline_citation, generate_bibliography, get_citation_summary, log_status]
)

# 4. Reflection Agent
reflection_agent = Agent(
    name="ReflectionAgent", 
    instructions="""Review the complete research for:
    1. Citation completeness - are all claims properly cited?
    2. Source quality and credibility
    3. Research comprehensiveness
    4. Citation format consistency
    5. Overall research quality
    
    Suggest specific improvements and identify any gaps.
    Use log_status to communicate your review process.""",
    model=flash_model,
    tools=[get_citation_summary, log_status]
)

# 5. Enhanced Orchestrator Agent
orchestrator_agent = Agent(
    name="OrchestratorAgent",
    instructions=f"""You are coordinating professional research for {user_info['name']} in {user_info['city']}.

    WORKFLOW:
    1. Use planning_agent to create research plan
    2. Use search_agent to find and cite sources  
    3. Use citations_agent to format citations professionally
    4. Use reflection_agent to review quality
    5. Provide final report with proper citations

    CITATION REQUIREMENTS:
    - All factual claims must include inline citations
    - Use professional citation format throughout
    - Include complete bibliography at the end
    - Maintain high academic standards

    Always use log_status to keep the user informed of progress.
    Provide a comprehensive, well-cited final answer.""",
    model=pro_model,
    tools=[
        log_status,
        planning_agent.as_tool(
            tool_name="planning_agent",
            tool_description="Creates detailed research plans"
        ),
        search_agent.as_tool(
            tool_name="search_agent", 
            tool_description="Searches web and creates citations automatically"
        ),
        citations_agent.as_tool(
            tool_name="citations_agent",
            tool_description="Formats professional citations and bibliographies"
        ),
        reflection_agent.as_tool(
            tool_name="reflection_agent",
            tool_description="Reviews research and citation quality"
        ),
        generate_bibliography,
        get_citation_summary
    ]
)

# ------------------------------
# Enhanced Run System
# ------------------------------
def run_research(query: str, citation_style: str = "apa"):
    """Run the research system with professional citations."""
    global citation_manager
    
    # Reset citation manager for new research
    citation_manager = CitationManager()
    
    print("=" * 80)
    print("🚀 STARTING PROFESSIONAL RESEARCH SYSTEM")
    print(f"📝 QUERY: {query}")
    print(f"👤 USER: {user_info['name']} from {user_info['city']}")
    print(f"📚 CITATION STYLE: {citation_style.upper()}")
    print("=" * 80)
    
    try:
        # Add citation style to query context
        enhanced_query = f"{query}\n\nPlease use {citation_style.upper()} citation style for this research."
        
        res = Runner.run_sync(orchestrator_agent, enhanced_query)
        
        print("\n" + "=" * 80)
        print("✅ FINAL RESEARCH REPORT WITH CITATIONS")
        print("=" * 80)
        print(res.final_output)
        
        # Generate final bibliography
        print("\n" + "=" * 60)
        print("📚 COMPLETE BIBLIOGRAPHY")
        print("=" * 60)
        bibliography = citation_manager.generate_bibliography(citation_style)
        print(bibliography)
        
        # Show source summary
        print("\n" + "=" * 60)
        print("📊 RESEARCH STATISTICS")
        print("=" * 60)
        summary = citation_manager.get_source_summary()
        print(summary)
        
        return {
            "report": res.final_output,
            "bibliography": bibliography,
            "summary": summary,
            "total_sources": len(citation_manager.sources)
        }

    except Exception as e:
        error_msg = str(e)

        # Try to detect Google Gemini API quota error
        if "429" in error_msg and "quota" in error_msg.lower():
            print("❌ ERROR: Quota exceeded for Gemini API.")
            
            # Attempt to parse retry delay from error JSON if available
            try:
                err_data = json.loads(error_msg.replace("❌ ERROR: ", "").strip())
                retry_delay = None

                # Look for retry delay in details
                details = err_data.get("error", {}).get("details", [])
                for detail in details:
                    if "@type" in detail and "RetryInfo" in detail["@type"]:
                        retry_delay = detail.get("retryDelay")
                        break

                print(f"📌 Suggestion: Wait {retry_delay or 'some seconds'} before retrying.")
                print("📖 Docs: https://ai.google.dev/gemini-api/docs/rate-limits")
            except Exception:
                print("⚠️ Could not parse detailed error info from response.")

        else:
            print(f"❌ ERROR: {error_msg}")

        return None

# Additional utility functions
def set_citation_style(style: str):
    """Set default citation style."""
    valid_styles = ["apa", "mla", "chicago", "web"]
    if style.lower() in valid_styles:
        print(f"📚 Citation style set to: {style.upper()}")
        return style.lower()
    else:
        print(f"❌ Invalid citation style. Valid options: {', '.join(valid_styles)}")
        return "apa"

def get_available_citation_styles():
    """Get list of available citation styles."""
    return ["apa", "mla", "chicago", "web"]

# Test the enhanced system
if __name__ == "__main__":
    query = "Analyze the economic impact of remote work policies on small businesses vs large corporations, including productivity data and employee satisfaction trends"
    
    # Run with different citation styles
    print("🔬 Testing with APA citation style...")
    result = run_research(query, citation_style="apa")
    
    if result:
        print(f"\n📈 Research completed successfully!")
        print(f"📊 Total sources cited: {result['total_sources']}")
    else:
        print("❌ Research failed due to errors.")