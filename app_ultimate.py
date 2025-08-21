import gradio as gr
from ultimate_research_agent import compile_ultimate_diagram
from research_agent import Analyst

# Helper to run the graph step-by-step
def run_graph(topic, max_analysts, human_analyst_feedback, progress=gr.Progress()):
    graph = compile_ultimate_diagram()
    # Initial state
    state = {
        "topic": topic,
        "max_analysts": int(max_analysts),
        "human_analyst_feedback": human_analyst_feedback,
        "analysts": [],
        "sections": [],
        "introduction": "",
        "content": "",
        "conclusion": "",
        "final_report": ""
    }
    thread = {"configurable": {"thread_id": "1"}}
    output = graph.invoke(state, thread)
    if "final_report" in output and output["final_report"]:
        return output["final_report"]
    return "No report generated."

with gr.Blocks() as demo:
    gr.Markdown("# Ultimate Research Agent")
    topic = gr.Textbox(label="Research Topic", placeholder="Enter your research topic")
    max_analysts = gr.Slider(label="Number of Analysts", minimum=1, maximum=10, step=1, value=3)
    human_analyst_feedback = gr.Textbox(label="Human Analyst Feedback", placeholder="Optional feedback for analysts")
    report_md = gr.Markdown("")

    generate_btn = gr.Button("Generate Full Report")

    def generate_report(topic, max_analysts, human_analyst_feedback, progress=gr.Progress()):
        # The progress argument will show a spinner automatically
        # Optionally, show a static loading message before running
        # return "⏳ Generating report, please wait..."  # If you want to show a message before running
        report = run_graph(topic, max_analysts, human_analyst_feedback, progress)
        return report

    generate_btn.click(
        fn=generate_report,
        inputs=[topic, max_analysts, human_analyst_feedback],
        outputs=report_md
    )

if __name__ == "__main__":
    demo.launch()

