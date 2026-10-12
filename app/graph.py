from typing import TypedDict, List, Dict, Any
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from app.config import GOOGLE_API_KEY, GEMINI_MODEL


class BreakdownState(TypedDict):
    messages: List[Dict[str, str]]
    query: str
    is_car_related: bool
    urgency: str
    diagnosis: str
    response: str


def invoke_gemini(messages: List[BaseMessage]) -> Any:
    """Invokes Gemini using working models (gemini-flash-latest, gemini-2.5-flash)."""
    models_to_try = [GEMINI_MODEL, "gemini-flash-latest", "gemini-2.5-flash"]
    seen = set()
    unique_models = [m for m in models_to_try if not (m in seen or seen.add(m))]

    last_err = None
    for model_name in unique_models:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2,
            )
            return llm.invoke(messages)
        except Exception as e:
            last_err = e
            continue
    raise last_err


def extract_text(content: Any) -> str:
    """Ensures clean string output from Gemini response."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content)


# 1. Guardrail Classifier Node (Strict Vehicle Restriction)
def guardrail_node(state: BreakdownState) -> Dict[str, Any]:
    query = state["query"]

    guardrail_prompt = f"""You are a strict safety classifier for "CypherPunks Emergency Vehicle & Roadside Assistance AI".
Determine whether the user's message is related to:
- Cars, vehicles, bikes, trucks, or any motor vehicle issues
- In-vehicle problems (mechanical, engine, cooling, tyres, battery, brakes, AC, electrical, noises, dashboard lights)
- Roadside emergencies (stranded, flat tire, dead battery, low fuel, towing, jumpstarting)
- Driving safety, highway breakdown protocols, vehicle maintenance, or automotive mechanics

If the query IS related to any of the above, mark it as IN-SCOPE.
If the query is asking about ANY other topic (such as general knowledge, coding, politics, recipes, homework, movies, weather, greetings, or casual chat), mark it as OUT-OF-SCOPE.

User message: "{query}"

Respond with ONLY valid JSON:
{{"is_car_related": true or false, "reason": "short explanation"}}
"""

    try:
        response = invoke_gemini([
            SystemMessage(content="You are a strict JSON classifier. Return only valid JSON."),
            HumanMessage(content=guardrail_prompt)
        ])
        content = extract_text(response.content).strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content.strip())
        is_car = bool(data.get("is_car_related", False))
    except Exception:
        keywords = ["car", "vehicle", "engine", "tyre", "tire", "brake", "battery", "oil",
                    "coolant", "radiator", "gear", "clutch", "fuel", "smoke", "noise",
                    "overheat", "stranded", "mechanic", "roadside", "highway", "breakdown",
                    "steering", "headlight", "puncture", "ac", "spark", "transmission"]
        is_car = any(kw in query.lower() for kw in keywords)

    return {"is_car_related": is_car}


# 2. Out-of-Scope Responder Node (Stops replies to non-car queries)
def out_of_scope_node(state: BreakdownState) -> Dict[str, Any]:
    refusal_message = (
        "⛔ **Out of Scope Request**\n\n"
        "I am the **CypherPunks AI Roadside Assistance & Vehicle Emergency Assistant**.\n\n"
        "My system is exclusively trained to assist stranded drivers and resolve **car problems, in-vehicle faults, mechanical breakdowns, and roadside emergencies**.\n\n"
        "I cannot assist with general knowledge, chit-chat, or non-vehicle queries.\n\n"
        "👉 *If you are experiencing any car issue (e.g., flat tyre, overheating engine, battery jumpstart, weird sound, warning lights), please describe your vehicle symptoms and I will help you immediately.*"
    )
    return {
        "response": refusal_message,
        "urgency": "OUT_OF_SCOPE",
        "diagnosis": "Non-car query blocked by guardrails."
    }


# 3. Vehicle Diagnostics & Roadside Triage Node
def diagnostic_node(state: BreakdownState) -> Dict[str, Any]:
    query = state["query"]
    history = state.get("messages", [])

    system_prompt = """You are "CypherPunks Emergency Vehicle & Roadside Assistance AI", an expert automotive engineer and roadside diagnostic specialist.

Your task is to answer vehicle, car, and in-cabin breakdown problems according to the user's intent:

1. IF THE USER ASKS A SMALL / SIMPLE DOUBT:
   (e.g., "What does this dashboard light mean?", "Can I top up coolant with water?", "Why is my AC blowing hot air?", "What is recommended tyre pressure?", "Is squeaking brake normal?"):
   - Deliver a BRIEF, CONCISE, AND DIRECT SOLUTION (2-4 bullet points).
   - Do NOT output long essays or overwhelming blocks of text.
   - State the core cause directly with quick actionable steps.

2. IF THE USER IS STRANDED OR EXPERIENCING A MAJOR BREAKDOWN / EMERGENCY:
   (e.g., engine overheating, smoking, flat tyre on highway, car completely dead, brake failure):
   - Provide structured roadside triage:
     1. 🚨 Immediate Safety Action (hazard lights, parking safely)
     2. 🔍 Probable Diagnosis & Severity
     3. 🛠️ Step-by-Step Roadside Fix / Triage
     4. 🚦 Safe to Drive vs. Tow Recommendation
     5. 📞 2-Sentence Mechanic Summary

Keep all responses helpful, accurate, and easy to read on a mobile phone screen for a driver."""

    conversation = [SystemMessage(content=system_prompt)]
    for msg in history[-6:]:
        role = msg.get("role", "user")
        text = msg.get("content", "")
        if role == "user":
            conversation.append(HumanMessage(content=text))
        elif role == "assistant":
            conversation.append(AIMessage(content=text))

    conversation.append(HumanMessage(content=query))

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
