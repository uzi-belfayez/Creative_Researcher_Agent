# Ultimate Research Agent

A modular, human-in-the-loop research agent for generating expert-level reports using multiple AI analyst personas.  
Built with LangGraph, LangChain, Gradio, and OpenAI.

## Features

- **Multi-Analyst Generation:** Automatically creates diverse analyst personas for your research topic.
- **Human Feedback Loop:** Review and modify analysts before proceeding.
- **Parallel Interviews:** Each analyst conducts an AI-powered interview and writes a memo.
- **Automated Report Writing:** Combines memos into a cohesive, markdown-formatted report with introduction and conclusion.
- **Interactive Gradio App:** User-friendly interface for topic entry, analyst review, feedback, and report generation.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd research_agent
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```


## Usage

### Gradio App

Run the Gradio app for interactive research:

```bash
uv run app.py
```

Or use the advanced and final ultimate app that uses all the features:

```bash
uv run app_final.py
```

- Enter your research topic and select the number of analysts.
- Review analyst profiles and provide feedback if needed.
- Generate the full report in markdown.

### Notebooks

Explore the logic and workflow in `/notebooks/research_agent.ipynb`.

## Main Files

- `ultimate_research_agent.py`: Core graph logic and agent orchestration.
- `app_final.py`: Gradio interfaces for interactive use.
- `notebooks/research_agent.ipynb`: Step-by-step notebook for experimentation.

## Customization

- Modify analyst instructions, interview logic, or report formatting in `ultimate_research_agent.py`.
- Add new tools or data sources as needed.

## License

MIT License

---

**Note:** This project requires valid API keys for OpenAI and other services.  
For best results, use GPT-4 or GPT-4o models.
