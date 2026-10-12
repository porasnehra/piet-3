    response = invoke_gemini(conversation)
    return {
        "response": extract_text(response.content),
        "urgency": "IN_SCOPE_DIAGNOSED",
        "diagnosis": "Automotive diagnosis generated successfully."
    }
def route_guardrail(state: BreakdownState) -> str:
    if state["is_car_related"]:
        return "diagnostic_node"
    return "out_of_scope_node"
# Build the Graph
workflow = StateGraph(BreakdownState)
workflow.add_node("guardrail_node", guardrail_node)
workflow.add_node("out_of_scope_node", out_of_scope_node)
workflow.add_node("diagnostic_node", diagnostic_node)
workflow.set_entry_point("guardrail_node")
workflow.add_conditional_edges(
    "guardrail_node",
    route_guardrail,
    {
        "diagnostic_node": "diagnostic_node",
        "out_of_scope_node": "out_of_scope_node",
    }
)
workflow.add_edge("diagnostic_node", END)
workflow.add_edge("out_of_scope_node", END)
breakdown_graph = workflow.compile()
