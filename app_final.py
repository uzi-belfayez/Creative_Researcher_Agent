import gradio as gr
from ultimate_research_agent import StateGraph
from ultimate_research_agent import ResearchGraphState
from ultimate_research_agent import  create_analysts
from ultimate_research_agent import human_feedback
from ultimate_research_agent import write_report
from ultimate_research_agent import write_introduction
from ultimate_research_agent import write_conclusion
from ultimate_research_agent import finalize_report
from ultimate_research_agent import initiate_all_interviews
from ultimate_research_agent import InterviewState
from ultimate_research_agent import generate_question
from ultimate_research_agent import search_web
from ultimate_research_agent import search_wikipedia
from ultimate_research_agent import generate_answer
from ultimate_research_agent import save_interview
from ultimate_research_agent import write_section
from ultimate_research_agent import route_messages


from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


from research_agent import compile_analyst_graph
from research_agent import Analyst
from research_agent import llm
from prompts import report_writer_instructions
from prompts import intro_conclusion_instructions

import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.types import Send
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import os, getpass
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import operator
from typing import  Annotated
from langgraph.graph import MessagesState
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string

# the small graph

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

# the big graph

# Add nodes and edges 
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report",write_report)
builder.add_node("write_introduction",write_introduction)
builder.add_node("write_conclusion",write_conclusion)
builder.add_node("finalize_report",finalize_report)

# Logic
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

# Compile
memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)



# def gradio_analyst_flow(topic, max_analysts, feedback, final_report):
#     thread = {"configurable": {"thread_id": "1"}}
#     # Step 1: Generate analysts
#     state = {"topic": topic, "max_analysts": int(max_analysts)}
#     analysts = []
#     for event in builder.stream(state, thread, stream_mode="values"):
#         if event.get('analysts', []):
#             analysts = event['analysts']
#     # Step 2: Apply feedback if provided
#     if feedback:
#         state["human_analyst_feedback"] = feedback
#         builder.update_state(thread, state, as_node="human_feedback")
#         state["human_analyst_feedback"] = None
#         builder.update_state(thread, state, as_node="human_feedback")
#         analysts = []
#         for event in builder.stream(state, thread, stream_mode="values"):
#             if event.get('analysts', []):
#                 analysts = event['analysts']
#     # Format analysts for markdown
#     if not analysts:
#         analyst_md = "No analysts generated."
#     else:
#         analyst_md = "\n".join([
#             f"**Name:** {a.name}\n**Affiliation:** {a.affiliation}\n**Role:** {a.role}\n**Description:** {a.description}\n---"
#             for a in analysts
#         ])

#     # Step3 : generate report
#     if final_report :
#         for event in graph.stream(None, thread, stream_mode="updates"):
#             print("--Node--")
#             node_name = next(iter(event.keys()))
#             print(node_name)

#         final_state = graph.get_state(thread)
#         report = final_state.values.get('final_report')

#         return report
#     return analyst_md


# with gr.Blocks() as demo:
#     gr.Markdown("# Analyst Generator (Human-in-the-Loop)")
#     topic = gr.Textbox(label="Research Topic", placeholder="Enter your research topic")
#     max_analysts = gr.Slider(label="Number of Analysts", minimum=1, maximum=10, step=1, value=3)
#     feedback = gr.Textbox(label="Human Analyst Feedback", placeholder="E.g. Replace analyst, add HR manager, etc.")
#     analyst_md = gr.Markdown(label="Analyst Profiles")
#     report = gr.Markdown(label= "Final report")

#     gen_analysts_btn = gr.Button("Generate/Update Analysts")
#     final_report_btn = gr.Button("Generate final report")

#     gen_analysts_btn.click(
#         fn=gradio_analyst_flow,
#         inputs=[topic, max_analysts, feedback, final_report],
#         outputs=analyst_md
#     )

#     final_report_btn.click(
#         fn=gradio_analyst_flow,
#         inputs=[topic, max_analysts, feedback, final_report],
#         outputs= report
#     )

# if __name__ == "__main__":
#     demo.launch()



# --- Shared thread so both buttons use same conversation ---
thread = {"configurable": {"thread_id": "1"}}

# Step 1: Generate or update analysts
def gradio_generate_analysts(topic, max_analysts, feedback):
    state = {"topic": topic, "max_analysts": int(max_analysts)}

    # Run until analysts are generated
    analysts = []
    for event in graph.stream(state, thread, stream_mode="values"):
        if event.get("analysts", []):
            analysts = event["analysts"]

    # Apply feedback if given
    if feedback:
        state["human_analyst_feedback"] = feedback
        graph.update_state(thread, state, as_node="human_feedback")
        state["human_analyst_feedback"] = None
        graph.update_state(thread, state, as_node="human_feedback")

        analysts = []
        for event in graph.stream(state, thread, stream_mode="values"):
            if event.get("analysts", []):
                analysts = event["analysts"]

    # Format analysts for markdown
    if not analysts:
        return "No analysts generated."
    return "\n".join([
        f"**Name:** {a.name}\n**Affiliation:** {a.affiliation}\n**Role:** {a.role}\n**Description:** {a.description}\n---"
        for a in analysts
    ])


# Step 2: Continue graph to generate final report
def gradio_generate_report(topic, max_analysts, feedback):
    # State is already saved in the thread via the previous step
    final_state = None
    for event in graph.stream(None, thread, stream_mode="updates"):
        node_name = next(iter(event.keys()))
        print("--Node--", node_name)

    final_state = graph.get_state(thread)
    report = final_state.values.get("final_report", "No report generated.")

    return report


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Analyst Generator (Human-in-the-Loop)")
    topic = gr.Textbox(label="Research Topic", placeholder="Enter your research topic")
    max_analysts = gr.Slider(label="Number of Analysts", minimum=1, maximum=10, step=1, value=3)
    feedback = gr.Textbox(label="Human Analyst Feedback", placeholder="E.g. Replace analyst, add HR manager, etc.")
    analyst_md = gr.Markdown(label="Analyst Profiles")
    report = gr.Markdown(label="Final Report")

    gen_analysts_btn = gr.Button("Generate/Update Analysts")
    final_report_btn = gr.Button("Generate Final Report")

    gen_analysts_btn.click(
        fn=gradio_generate_analysts,
        inputs=[topic, max_analysts, feedback],
        outputs=analyst_md
    )

    final_report_btn.click(
        fn=gradio_generate_report,
        inputs=[topic, max_analysts, feedback],
        outputs=report
    )

if __name__ == "__main__":
    demo.launch()
