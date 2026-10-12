from typing import TypedDict, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from app.config import OPENAI_API_KEY, OPENAI_MODEL

# State Schema
class VehicleAssistantState(TypedDict):
    message: str
    location_context: Optional[str]
    vehicle_type: Optional[str]
    session_id: Optional[str]
    is_vehicle_related: bool
    category: str
    response: str


# Structured output model for query classification
class QueryClassification(BaseModel):
    is_vehicle_related: bool = Field(
        ...,
        description="True ONLY if the query is directly about a vehicle/car/automobile problem, malfunction, breakdown, roadside emergency, maintenance issue, or automotive troubleshooting. False for ANY other topic (including casual banter, coding, general trivia, politics, food, movies, personal questions, etc.)."
    )
    category: str = Field(
        ...,
        description="One of: 'emergency_breakdown', 'flat_tyre', 'engine_cooling_problem', 'battery_electrical', 'fuel_or_fluids', 'brakes_transmission', 'general_vehicle_issue', 'out_of_scope'"
    )
    reason: str = Field(
        ...,
        description="Short 1-sentence reason for classification."
    )


def get_llm(temperature: float = 0.2):
    """Initializes the ChatOpenAI client with model validation."""
    model_name = OPENAI_MODEL
    # If the user specified 'gpt-4.1-mini', map it gracefully or keep default 'gpt-4o-mini'
    if "4.1" in model_name:
        model_name = "gpt-4o-mini"

    return ChatOpenAI(
        model=model_name,
        api_key=OPENAI_API_KEY,
        temperature=temperature
    )


# Node 1: Classifier / Guardrail
def classify_query_node(state: VehicleAssistantState) -> dict:
    """Classifies whether the user request is strictly vehicle/automotive related."""
    user_query = state.get("message", "").strip()
    
    # Fast path heuristic refusal for clearly empty queries
    if not user_query:
        return {
            "is_vehicle_related": False,
            "category": "out_of_scope",
            "response": "Please describe your vehicle problem or roadside breakdown situation."
        }

    classifier_system_prompt = (
        "You are an automated guardrail classifier for 'CypherPunks Roadside & Vehicle Assistance AI'.\n"
        "Your sole task is to strictly determine whether the user's inquiry relates to:\n"
        "1. Vehicle problems, mechanical or electrical faults (engine, battery, transmission, brakes, tyres, fuel, radiator, lights, dashboard warnings, strange noises, overheating, stalling).\n"
        "2. Roadside breakdown emergencies (stuck on highways, rural areas, flat tyre, jump-starting, running out of fuel, towing safety, stranded driver protocols).\n"
        "3. Automotive troubleshooting and DIY safety steps.\n\n"
        "STRICT POLICY:\n"
        "- If the query is about ANYTHING else (e.g. recipes, general chat, coding, math, general health, politics, movies, weather forecasts, essay writing), you MUST classify it as is_vehicle_related = False and category = 'out_of_scope'.\n"
        "- Do not be lenient. Only allow vehicle, automotive, and roadside assistance topics."
    )

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(QueryClassification)
        classification = structured_llm.invoke([
            SystemMessage(content=classifier_system_prompt),
            HumanMessage(content=f"User Query: {user_query}")
        ])

        return {
            "is_vehicle_related": classification.is_vehicle_related,
            "category": classification.category
        }
    except Exception as e:
        # Fallback keyword inspection if API or structured output hits any glitch
        lowered = user_query.lower()
        vehicle_keywords = [
            "car", "vehicle", "engine", "tyre", "tire", "battery", "breakdown", "brake",
            "clutch", "radiator", "overheat", "overheating", "fuel", "gas", "diesel",
            "petrol", "puncture", "mechanic", "towing", "highway", "stranded", "oil",
            "spark plug", "transmission", "accelerat", "starter", "alternator", "smoke",
            "steering", "coolant", "jump start", "rsa"
        ]
        is_related = any(kw in lowered for kw in vehicle_keywords)
        return {
            "is_vehicle_related": is_related,
            "category": "general_vehicle_issue" if is_related else "out_of_scope"
        }


# Node 2: Out-of-scope rejection node
def out_of_scope_node(state: VehicleAssistantState) -> dict:
    """Strictly halts replying to unrelated queries and informs the user."""
    return {
        "response": (
            "⚠️ Request Denied: I am the CypherPunks Roadside & Vehicle Assistance AI, "
            "strictly dedicated to vehicle breakdowns, mechanical/electrical diagnostics, "
            "and roadside emergency troubleshooting.\n\n"
            "I cannot assist with general topics or unrelated queries. "
            "Please describe a specific problem with your vehicle (e.g., flat tyre, engine overheating, "
            "dead battery, strange sounds, highway breakdown, low fuel)."
        ),
        "category": "out_of_scope"
    }


# Node 3: Vehicle Assistant Troubleshooting Node
def vehicle_assistant_node(state: VehicleAssistantState) -> dict:
    """Generates expert, actionable emergency troubleshooting guidance for drivers."""
    user_query = state.get("message", "")
    loc = state.get("location_context") or "Not specified (assume potential highway/rural risk)"
    v_type = state.get("vehicle_type") or "Not specified"
    category = state.get("category", "breakdown_troubleshooting")

    system_prompt = (
        "You are the expert CypherPunks Roadside & Vehicle Assistance AI.\n"
        "You are assisting a driver whose vehicle has a problem or who may be stranded on a road/highway/rural area.\n\n"
        "CONTEXT OF THE PROBLEM:\n"
        "- Many drivers get stuck on highways, remote roads, or hilly areas with sudden breakdowns (flat tyres, overheating, dead battery, low fuel).\n"
        "- Drivers often lack RSA (Roadside Assistance) memberships, and services in rural/highway areas are slow or non-existent.\n\n"
        "YOUR CORE GOALS:\n"
        "1. IMMEDIATE SAFETY FIRST: Remind driver to pull safely to the shoulder/safe zone, turn on hazard warning lights, place warning triangle/cones if safe, and stay behind guardrails on busy expressways.\n"
        "2. ACCURATE DIAGNOSIS: Pinpoint probable causes based on reported symptoms (noises, smells, dashboard lights, smoke, vibrations).\n"
        "3. STEP-BY-STEP ACTIONABLE TROUBLESHOOTING: Provide clear, sequential, practical steps the driver can safely take on the spot (e.g. tyre changing precautions, safe cool-down procedure before opening bonnet/hood, jump start terminal connection order, fuse/relay inspection).\n"
        "4. CRITICAL SAFETY WARNINGS: Explicitly warn about life-threatening mistakes (e.g. NEVER open radiator cap while hot, NEVER get under a car supported only by a scissor jack, avoid cranking a seized engine).\n"
        "5. REMOTE / NO RSA ADVICE: If the driver cannot fix it themselves, guide them on how to communicate their exact location (GPS landmarks/mile markers), conserve battery/resources, and seek local help or emergency police/highway patrol assistance.\n\n"
        "Tone: Calm, authoritative, empathetic, safety-focused, and highly structured with bullet points."
    )

    human_prompt = (
        f"Driver's Problem Description: {user_query}\n"
        f"Vehicle Details: {v_type}\n"
        f"Location / Situation Context: {loc}\n"
        f"Identified Category: {category}\n\n"
        "Please provide immediate safety advice, likely root cause, step-by-step troubleshooting, and next steps."
    )

    llm = get_llm(temperature=0.3)
    response_msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return {
        "response": response_msg.content,
        "category": category
    }


# Conditional routing function
def route_query(state: VehicleAssistantState) -> str:
    if state.get("is_vehicle_related", False):
        return "vehicle_assistant"
    return "out_of_scope"


# Build the LangGraph workflow
def build_vehicle_graph():
    workflow = StateGraph(VehicleAssistantState)

    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("vehicle_assistant", vehicle_assistant_node)

    workflow.set_entry_point("classify_query")

    workflow.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "vehicle_assistant": "vehicle_assistant",
            "out_of_scope": "out_of_scope"
        }
    )

    workflow.add_edge("vehicle_assistant", END)
    workflow.add_edge("out_of_scope", END)

    return workflow.compile()


# Compile reusable graph instance
roadside_graph = build_vehicle_graph()
