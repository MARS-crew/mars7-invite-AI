from pydantic import BaseModel

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