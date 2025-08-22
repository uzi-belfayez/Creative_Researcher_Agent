import gradio as gr
import os, getpass
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict

# --- Environment setup ---
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Analyst Model ---
class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations.")

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]

# --- Analyst Generation Logic ---
analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

def create_analysts(state: GenerateAnalystsState):
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    structured_llm = llm.with_structured_output(Perspectives)
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts
    )
    analysts = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Generate the set of analysts.")])
    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    pass

def should_continue(state: GenerateAnalystsState):
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    return END

# --- Build Analyst Graph ---
analyst_builder = StateGraph(GenerateAnalystsState)
analyst_builder.add_node("create_analysts", create_analysts)
analyst_builder.add_node("human_feedback", human_feedback)
analyst_builder.add_edge(START, "create_analysts")
analyst_builder.add_edge("create_analysts", "human_feedback")
analyst_builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])
memory = MemorySaver()
analyst_graph = analyst_builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

# --- Gradio App ---
def gradio_analyst_flow(topic, max_analysts, feedback):
    thread = {"configurable": {"thread_id": "1"}}
    # Step 1: Generate analysts
    state = {"topic": topic, "max_analysts": int(max_analysts)}
    analysts = []
    for event in analyst_graph.stream(state, thread, stream_mode="values"):
        if event.get('analysts', []):
            analysts = event['analysts']
    # Step 2: Apply feedback if provided
    if feedback:
        state["human_analyst_feedback"] = feedback
        analyst_graph.update_state(thread, state, as_node="human_feedback")
        state["human_analyst_feedback"] = None
        analyst_graph.update_state(thread, state, as_node="human_feedback")
        analysts = []
        for event in analyst_graph.stream(state, thread, stream_mode="values"):
            if event.get('analysts', []):
                analysts = event['analysts']
    # Format analysts for markdown
    if not analysts:
        analyst_md = "No analysts generated."
    else:
        analyst_md = "\n".join([
            f"**Name:** {a.name}\n**Affiliation:** {a.affiliation}\n**Role:** {a.role}\n**Description:** {a.description}\n---"
            for a in analysts
        ])
    return analyst_md

with gr.Blocks() as demo:
    gr.Markdown("# Analyst Generator (Human-in-the-Loop)")
    topic = gr.Textbox(label="Research Topic", placeholder="Enter your research topic")
    max_analysts = gr.Slider(label="Number of Analysts", minimum=1, maximum=10, step=1, value=3)
    feedback = gr.Textbox(label="Human Analyst Feedback", placeholder="E.g. Replace analyst, add HR manager, etc.")
    analyst_md = gr.Markdown(label="Analyst Profiles")

    gen_analysts_btn = gr.Button("Generate/Update Analysts")

    gen_analysts_btn.click(
        fn=gradio_analyst_flow,
        inputs=[topic, max_analysts, feedback],
        outputs=analyst_md
    )

if __name__ == "__main__":
    demo.launch()
