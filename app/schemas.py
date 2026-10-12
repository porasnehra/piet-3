from typing import Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user query or emergency roadside situation description.", example="My engine is overheating and white smoke is coming from under the hood while on the highway.")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID for tracking.")
    location_context: Optional[str] = Field(None, description="Optional location details (e.g., Highway NH-48, remote hilly road, 15 km before toll).")
    vehicle_type: Optional[str] = Field(None, description="Optional vehicle details (e.g., Petrol Hatchback, Diesel SUV, Automatic, EV).")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The AI assistant diagnosis, safety advice, troubleshooting, or polite rejection.")
    is_vehicle_related: bool = Field(..., description="True if the query relates to vehicle issues, roadside breakdown, or automotive assistance; False otherwise.")
    category: str = Field(..., description="Categorization of the query (e.g., 'breakdown_troubleshooting', 'emergency_safety', 'out_of_scope').")
    session_id: Optional[str] = Field(None, description="Session ID echoing the request or newly assigned.")
    status: str = Field("success", description="Status code or indicator ('success' or 'rejected_out_of_scope').")

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model: str
