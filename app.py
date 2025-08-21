import gradio as gr
import os, getpass
from research_agent import create_analysts, compile_analyst_graph, GenerateAnalystsState, Analyst

def generate_analysts_ui(topic, max_analysts, feedback):
    state = {
        "topic": topic,
        "max_analysts": int(max_analysts),
        "human_analyst_feedback": feedback,
        "analysts": []
    }
    result = create_analysts(state)
    analysts = result["analysts"]
    # Format each analyst persona as Markdown and join into a single string
    formatted = "\n---\n".join(
        [
            f"### Analyst {i+1}\n"
            f"**Name:** {a.name}\n\n"
            f"**Role:** {a.role}\n\n"
            f"**Affiliation:** {a.affiliation}\n\n"
            f"**Description:** {a.description}\n"
            for i, a in enumerate(analysts)
        ]
    )
    return formatted

def run_interview_ui(topic, max_analysts, feedback, analyst_idx):
    state = {
        "topic": topic,
        "max_analysts": int(max_analysts),
        "human_analyst_feedback": feedback,
        "analysts": []
    }
    result = create_analysts(state)
    analysts = result["analysts"]
    if not analysts:
        return "No analysts generated."
    analyst = analysts[int(analyst_idx)]
    interview_state = {
        "messages": [],
        "max_num_turns": 2,
        "context": [],
        "analyst": analyst,
        "interview": "",
        "sections": []
    }
    thread = {"configurable": {"thread_id": "1"}}
    graph = compile_analyst_graph()
    output = graph.invoke(interview_state, thread)
    return output.get("sections", ["No section generated."])[0]


with gr.Blocks() as demo:

    gr.Markdown("# Research Agent Gradio App")
    topic = gr.Textbox(label="Research Topic")
    max_analysts = gr.Number(label="Number of Analysts", value=3)
    feedback = gr.Textbox(label="Human Analyst Feedback")
    generate_btn = gr.Button("Generate Analysts")
    analysts_output = gr.Markdown(label="Analysts")
    analyst_idx = gr.Number(label="Select Analyst Index", value=0)
    interview_btn = gr.Button("Run Interview")
    interview_output = gr.Markdown(label="Interview Section")

    generate_btn.click(
        generate_analysts_ui,
        inputs=[topic, max_analysts, feedback],
        outputs=analysts_output
    )
    interview_btn.click(
        run_interview_ui,
        inputs=[topic, max_analysts, feedback, analyst_idx],
        outputs=interview_output
    )




if __name__ == "__main__":

    demo.launch()
