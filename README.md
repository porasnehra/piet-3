# CypherPunks Roadside & Vehicle Assistance AI Backend 🚗🛠️

A specialized AI assistant API built with **FastAPI**, **LangChain**, and **LangGraph** designed specifically to help stranded drivers solve unexpected vehicle breakdowns (such as flat tyres, engine overheating, dead batteries, or running out of fuel), especially in remote highway and rural locations where roadside assistance (RSA) is slow or inaccessible.

---

## 🎯 Strict Guardrail Policy
As per the requirement:
- **Vehicle-Related Inquiries**: The assistant provides immediate driver safety precautions, root-cause diagnosis, step-by-step DIY triage, and remote emergency protocols.
- **Unrelated Inquiries**: If a user asks about *anything else* (coding, recipes, general chit-chat, politics, sports, etc.), the system **strictly halts replying to that query** and returns a polite rejection notice explaining its focused automotive mission.

---

## 🏗️ Architecture & LangGraph Workflow

```
             [ User Request ]
                    │
                    ▼
          ┌───────────────────┐
          │   classify_query   │  (LangChain + Guardrail Classifier)
          └─────────┬─────────┘
                    │
         Is vehicle related?
         ┌──────────┴──────────┐
      [YES]                  [NO]
         ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│ vehicle_assistant│  │   out_of_scope   │  (Direct Rejection without token waste)
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
                 [ END ]
```

---

## 🚀 Getting Started

### 1. Configuration (.env)
The `.env` file is located in the root directory:
```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
PORT=8000
HOST=0.0.0.0
```

### 2. Start the Server
Run using Uvicorn:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Or directly with Python:
```bash
python app/main.py
```

Interactive Swagger API docs will be available at:
👉 **`http://localhost:8000/docs`**

---

## 📡 API Endpoints

### 1. Chat Endpoint: `POST /api/chat`
#### Request Body:
```json
{
  "message": "My car temperature gauge is in the red zone and steam is coming from the hood on the highway.",
  "location_context": "Highway NH-48, 25km before next town",
  "vehicle_type": "Sedan, Petrol"
}
```

#### Response (Vehicle-Related Query):
```json
{
  "response": "⚠️ IMMEDIATE SAFETY FIRST:\n1. Pull over immediately to the highway shoulder...\n2. Turn on your hazard lights and place the warning triangle 50m behind your vehicle.\n3. Turn off the engine.\n\n⚠️ CRITICAL WARNING: DO NOT OPEN THE RADIATOR CAP while the engine is hot!...",
  "is_vehicle_related": true,
  "category": "engine_cooling_problem",
  "session_id": "8b51d5a3-7...",
  "status": "success"
}
```

#### Request (Unrelated Query):
```json
{
  "message": "Can you write me a Python script to sort an array?"
}
```

#### Response (Strictly Rejected):
```json
{
  "response": "⚠️ Request Denied: I am the CypherPunks Roadside & Vehicle Assistance AI, strictly dedicated to vehicle breakdowns, mechanical/electrical diagnostics, and roadside emergency troubleshooting.\n\nI cannot assist with general topics or unrelated queries. Please describe a specific problem with your vehicle (e.g., flat tyre, engine overheating, dead battery, strange sounds, highway breakdown, low fuel).",
  "is_vehicle_related": false,
  "category": "out_of_scope",
  "session_id": "a904d1c2-3...",
  "status": "rejected_out_of_scope"
}
```

---

### 2. Health Check: `GET /health`
```json
{
  "status": "healthy",
  "service": "CypherPunks Roadside Assistance API",
  "version": "1.0.0",
  "model": "gpt-4o-mini"
}
```

### 3. Project Information: `GET /api/info`
Returns the CypherPunks problem statement and solution overview.
