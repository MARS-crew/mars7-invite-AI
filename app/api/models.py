from pydantic import BaseModel
from typing import Optional, List


class StartChatResponse(BaseModel):
    session_id: str
    response_message: str
    next_step: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ProfileData(BaseModel):
    name: Optional[str]
    department: Optional[str]
    age: Optional[str]
    phone_number: Optional[str]
    positions: Optional[List[str]]
    motivation: Optional[str]


class ChatResponse(BaseModel):
    session_id: str
    response_message: str
    next_step: str
    profile_data: Optional[ProfileData] = None
