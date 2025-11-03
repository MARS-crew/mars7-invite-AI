from pydantic import BaseModel
from typing import Optional, List

class StartChatResponse(BaseModel):
    session_id: str
    response_message: str
    next_step: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response_message: str
    next_step: str