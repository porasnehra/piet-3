from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path
import uuid

from app.graph import breakdown_graph
from app.config import GOOGLE_API_KEY, APP_API_KEY

app = FastAPI(
    title="CypherPunks Emergency Vehicle & Roadside Assistance AI",
    description="Intelligent in-vehicle diagnostic and roadside triage chatbot powered by Google Gemini.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_STORE: Dict[str, List[Dict[str, str]]] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., example="Why is my car AC blowing hot air?")
    session_id: Optional[str] = Field(default=None, example="driver-session-123")
    api_key: Optional[str] = Field(default=None, example="cypherpunks-secret-key-2024")


class ChatResponse(BaseModel):
    session_id: str
    is_car_related: bool
    response: str
    urgency: str


class SOSRequest(BaseModel):
    driver_name: Optional[str] = "Stranded Driver"
    phone: Optional[str] = "Unknown"
    location_desc: str = Field(..., example="NH44, 25km before toll plaza")
    issue_type: str = Field(..., example="Flat tyre and no spare jack")
    api_key: Optional[str] = None


def verify_client_key(header_key: Optional[str], body_key: Optional[str], query_key: Optional[str]):
    if not APP_API_KEY:
        return True
    provided = header_key or body_key or query_key
    if not provided or provided.strip() != APP_API_KEY.strip():
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing or invalid API key. Set 'X-API-Key' header or 'api_key' in body."
        )
    return True


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "CypherPunks AI Roadside Assistant",
        "llm_engine": "Google Gemini",
        "google_key_configured": bool(GOOGLE_API_KEY and len(GOOGLE_API_KEY) > 10),
        "api_key_protection_active": bool(APP_API_KEY)
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Query(None)
):
    verify_client_key(x_api_key, payload.api_key, api_key)

    user_query = payload.message.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = payload.session_id or str(uuid.uuid4())
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = []

    history = SESSION_STORE[session_id]

    inputs = {
        "messages": history,
        "query": user_query,
        "is_car_related": False,
        "urgency": "PENDING",
        "diagnosis": "",
        "response": ""
    }

    try:
        result = breakdown_graph.invoke(inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI execution error: {str(e)}")

    is_car = result.get("is_car_related", False)
    raw_response = result.get("response", "Could not process request.")

    if isinstance(raw_response, list):
        bot_response = "\n".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in raw_response])
    else:
        bot_response = str(raw_response)

    urgency = str(result.get("urgency", "UNKNOWN"))

    SESSION_STORE[session_id].append({"role": "user", "content": user_query})
    SESSION_STORE[session_id].append({"role": "assistant", "content": bot_response})
    if len(SESSION_STORE[session_id]) > 20:
        SESSION_STORE[session_id] = SESSION_STORE[session_id][-20:]

    return ChatResponse(
        session_id=session_id,
        is_car_related=is_car,
        response=bot_response,
        urgency=urgency
    )


@app.post("/api/sos")
async def trigger_sos(payload: SOSRequest):
    sos_id = f"SOS-{uuid.uuid4().hex[:6].upper()}"
    return {
        "sos_id": sos_id,
        "status": "DISPATCH_PENDING",
        "message": f"Emergency roadside broadcast created for {payload.location_desc}.",
        "details": payload.model_dump()
    }


static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(static_dir / "index.html"))
