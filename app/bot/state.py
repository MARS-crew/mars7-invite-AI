from typing import List, TypedDict, Annotated, Literal, Optional
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage



class UserInfo(BaseModel):
    name: Optional[str] = Field(default=None, description="사용자의 이름")
    department: Optional[str] = Field(default=None, description="사용자의 학과 (예: 컴퓨터공학과)")
    age: Optional[str] = Field(default=None, description="사용자의 나이 또는 학번 (예: 23살, 21학번)")
    phone_number: Optional[str] = Field(default=None, description="사용자의 전화번호 (예: 010-1234-5678)")


class PositionInfo(BaseModel):
    positions: List[str] = Field(description="사용자가 관심있는 포지션 목록 (예: ['개발자', '디자이너'])")


class QASessionIntent(BaseModel):
    intent: Literal["continue_chat", "end_chat"] = Field(
        description="사용자가 '종료', '그만', '됐어', '지원서 생성해줘', '마무리' 등 대화 종료 의사를 명확히 밝히면 'end_chat', 그 외 모든 질문이나 대답은 'continue_chat'."
    )


class ApplicationFormState(TypedDict):
    name: Optional[str]
    department: Optional[str]
    age: Optional[str]
    phone_number: Optional[str]
    positions: List[str]
    resume_summary: Optional[str]
    initial_motivation: Optional[str]

    messages: Annotated[List[BaseMessage], operator.add]

    next_question: Literal[
        "intro", "position",
        "process_initial_motivation",
        "qa_session",
        "generate_resume",
        "done"
    ]