import gradio as gr
from ultimate_research_agent import compile_ultimate_diagram

def create_analysts(topic, max_analysts, thread, graph):
    
    state = {"topic": topic, "max_analysts": int(max_analysts)}
    analysts = []
    # Fully consume the generator
    for event in graph.stream(state, thread, stream_mode="values"):
        if event.get('analysts', []):
            analysts = event['analysts']
    if not analysts:
        return "No analysts generated.", gr.update(visible=False)
    analyst_md = "\n".join([
        f"**Name:** {a.name}\n**Affiliation:** {a.affiliation}\n**Role:** {a.role}\n**Description:** {a.description}\n---"
        for a in analysts
    ])
    return analyst_md, gr.update(visible=True)

def update_analysts(topic, max_analysts, feedback, thread, graph):
    # Always pass full state including topic and max_analysts
    state = {"topic": topic, "max_analysts": int(max_analysts), "human_analyst_feedback": feedback}
    graph.update_state(thread, state, as_node="human_feedback")
    # Clear feedback for next step
    state["human_analyst_feedback"] = None
    graph.update_state(thread, state, as_node="human_feedback")
    analysts = []
    # Fully consume the generator
    for event in graph.stream(state, thread, stream_mode="values"):
        if event.get('analysts', []):
            analysts = event['analysts']
    if not analysts:
        return "No analysts generated.", gr.update(visible=False)
    analyst_md = "\n".join([
        f"**Name:** {a.name}\n**Affiliation:** {a.affiliation}\n**Role:** {a.role}\n**Description:** {a.description}\n---"
        for a in analysts
    ])
    return analyst_md, gr.update(visible=True)

def generate_report(topic, max_analysts, feedback):
    graph = compile_ultimate_diagram()
    thread = {"configurable": {"thread_id": "1"}}

    analyst_md, _ = create_analysts(topic, max_analysts, thread, graph)
    analyst_md, _ = update_analysts(topic, max_analysts, feedback, thread, graph)

    # Always pass full state including topic and max_analysts
    state = {"topic": topic, "max_analysts": int(max_analysts), "human_analyst_feedback": feedback}
    graph.update_state(thread, state, as_node="human_feedback")
    state["human_analyst_feedback"] = None
    graph.update_state(thread, state, as_node="human_feedback")
    final_state = graph.get_state(thread)
    report = final_state.values.get('final_report', "No report generated.")
    return report

with gr.Blocks() as demo:
    gr.Markdown("# Ultimate Research Agent")

    topic = gr.Textbox(label="Research Topic", placeholder="Enter your research topic")
    max_analysts = gr.Slider(label="Number of Analysts", minimum=1, maximum=10, step=1, value=3)
    feedback = gr.Textbox(label="Human Analyst Feedback", placeholder="E.g. Replace analyst, add HR manager, etc.")

    analyst_md = gr.Markdown(label="Analyst Profiles", visible=False)
    report_md = gr.Markdown(label="Report Section")

    with gr.Row():
        # gen_analysts_btn = gr.Button("Generate Analysts")
        # update_analysts_btn = gr.Button("Update Analysts with Feedback")
        gen_report_btn = gr.Button("Generate Full Report")

    # gen_analysts_btn.click(
    #     fn=create_analysts,
    #     inputs=[topic, max_analysts],
    #     outputs=[analyst_md, report_md]
    # )

    # update_analysts_btn.click(
    #     fn=update_analysts,
    #     inputs=[topic, max_analysts, feedback],
    #     outputs=[analyst_md, report_md]
    # )

    gen_report_btn.click(
        fn=generate_report,
        inputs=[topic, max_analysts, feedback],
        outputs=report_md
    )

if __name__ == "__main__":
    demo.launch()

