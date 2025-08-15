# Professional Research & Citation System

This project is a **professional multi-agent research system** with **automatic citation management**.  
It integrates Google Gemini, OpenAI-compatible models, and the Tavily Search API to perform structured research with **APA, MLA, Chicago, or Web citation styles**.

---

## 📌 Features

- **Automated Research Workflow**
  - Plans research
  - Searches credible sources
  - Extracts and stores citations
  - Formats citations professionally
  - Reviews quality of research
- **Citation Manager**
  - Add sources automatically from search results or manually
  - Inline citations `[1,2,3]`
  - Bibliography generation in APA, MLA, Chicago, or Web format
  - Domain statistics for cited sources
- **Multi-Agent Architecture**
  - **Planning Agent** – creates research strategy
  - **Search Agent** – performs web searches with Tavily and cites sources
  - **Citations Agent** – formats and compiles citations
  - **Reflection Agent** – checks quality and completeness
  - **Orchestrator Agent** – coordinates all steps into a final report
- **Multiple Citation Styles**
  - APA, MLA, Chicago, or simple Web format

---

## 📂 Installation

1. **Clone the repository**  
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**  
   Create a `.env` file in the root directory with the following:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

---

## 🚀 Usage

Run the script directly:
```bash
python main.py
```

Example in `main.py`:
```python
query = "Analyze the economic impact of remote work policies on small businesses vs large corporations, including productivity data and employee satisfaction trends"
result = run_research(query, citation_style="apa")

if result:
    print("📈 Research completed successfully!")
    print(f"📊 Total sources cited: {result['total_sources']}")
```

---

## 🛠 Available Citation Styles

- **APA** – `"apa"`
- **MLA** – `"mla"`
- **Chicago** – `"chicago"`
- **Web** – `"web"`

You can change styles by:
```python
set_citation_style("mla")
```

---

## 📊 Example Output

```
================================================================================
🚀 STARTING PROFESSIONAL RESEARCH SYSTEM
📝 QUERY: Analyze the economic impact of remote work policies...
👤 USER: Husnain from Pakistan
📚 CITATION STYLE: APA
================================================================================

✅ FINAL RESEARCH REPORT WITH CITATIONS
...content with inline citations like [1,2]...

📚 COMPLETE BIBLIOGRAPHY
[1] Example Source Title. (2024). example.com. Retrieved 2025-08-15, from https://example.com

📊 RESEARCH STATISTICS
Total sources: 5
Unique domains: 4
Domain distribution:
  - example.com: 2 sources
  - another.com: 1 source
```

---

## 📦 Project Structure

```
main.py                # Main execution script
agents/                # Agent and model definitions
.env                   # API keys
README.md              # This file
requirements.txt       # Python dependencies
```

---

## ⚠ Notes

- Ensure you have **valid API keys** for **Google Gemini**, **OpenAI-compatible endpoint**, and **Tavily Search**.
- Tavily searches return results with metadata used for citation creation.
- This script is designed for **research purposes** and requires internet access for live search.
