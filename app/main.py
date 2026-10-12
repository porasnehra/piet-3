import uuid
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.graph import roadside_graph
from app.config import OPENAI_MODEL, OPENAI_API_KEY

app = FastAPI(
    title="CypherPunks Roadside & Vehicle Assistance AI API",
    description=(
        "Specialized AI backend for vehicle breakdown emergencies, roadside assistance, "
        "and automotive troubleshooting using FastAPI, LangChain, and LangGraph. "
        "Strictly filters out non-vehicle queries."
    ),
    version="1.0.0"
)

# Enable CORS for cross-platform integration (Web, Mobile, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    return {
        "project": "CypherPunks Roadside & Vehicle Assistance AI",
        "description": "AI assistance API for stranded drivers, vehicle mechanical/electrical issues, and roadside breakdowns.",
        "documentation": "/docs",
        "status": "online",
        "chat_endpoint": "/api/chat"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy",
        service="CypherPunks Roadside Assistance API",
        version="1.0.0",
        model=OPENAI_MODEL
    )


@app.get("/api/info", tags=["Information"])
async def project_info():
    return {
        "problem_statement": (
            "Every day, thousands of drivers get stuck on highways, remote roads, or hilly areas "
            "because of sudden vehicle breakdowns such as flat tyres, engine problems, or low fuel. "
            "Drivers often lack RSA plans, and existing services are slow, limited, and urban-centric."
        ),
        "solution": (
            "An intelligent AI-powered automotive diagnostic and roadside assistant that immediately "
            "triages emergencies, delivers step-by-step DIY safety and repair instructions, handles remote/highway "
            "scenarios, and strictly rejects any non-vehicle inquiries."
        ),
        "team": "CypherPunks",
        "stack": ["FastAPI", "LangChain", "LangGraph", "OpenAI GPT-4"]
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Primary API endpoint for drivers and roadside assistance queries.
    
    - Evaluates if the query is strictly vehicle / automotive / breakdown related.
    - If unrelated: Strictly stops and returns a polite rejection notice.
    - If vehicle-related: Invokes LangGraph to generate immediate safety protocols and troubleshooting steps.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY is not configured in the backend environment."
        )

    session_id = request.session_id or str(uuid.uuid4())

    try:
        initial_state = {
            "message": request.message,
            "location_context": request.location_context,
            "vehicle_type": request.vehicle_type,
            "session_id": session_id,
            "is_vehicle_related": False,
            "category": "pending",
            "response": ""
        }

        # Execute through the compiled LangGraph workflow
        final_state = await roadside_graph.ainvoke(initial_state)

        is_vehicle = final_state.get("is_vehicle_related", False)
        status_flag = "success" if is_vehicle else "rejected_out_of_scope"

        return ChatResponse(
            response=final_state.get("response", ""),
            is_vehicle_related=is_vehicle,
            category=final_state.get("category", "out_of_scope"),
            session_id=session_id,
            status=status_flag
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing roadside assistance workflow: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
