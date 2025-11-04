from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from uuid import uuid4
from langchain_core.messages import HumanMessage
from .models import StartChatResponse, ChatRequest, ChatResponse, ProfileData
from ..bot.state import ApplicationFormState

router = APIRouter()


def get_langgraph_app(request: Request):
    app = request.app.state.langgraph_app
    if not app:
        raise HTTPException(status_code=500, detail="LangGraph 앱이 초기화되지 않았습니다.")
    return app


@router.post("/chat/start", response_model=StartChatResponse)
async def start_chat(app=Depends(get_langgraph_app)):
    session_id = str(uuid4())
    config = {"configurable": {"thread_id": session_id}}

    try:
        response_state: ApplicationFormState = await app.ainvoke({}, config=config)

        last_message = response_state['messages'][-1].content
        next_step = response_state['next_question']

        return StartChatResponse(
            session_id=session_id,
            response_message=last_message,
            next_step=next_step
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대화 시작 중 오류 발생: {str(e)}")


@router.post("/chat/send", response_model=ChatResponse)
async def send_chat_message(request: ChatRequest, app=Depends(get_langgraph_app)):
    config = {"configurable": {"thread_id": request.session_id}}

    try:
        response_state = await app.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )

        next_step = response_state.get('next_question')
        if next_step == "generate_resume":
            print(f"[{request.session_id}] 'generate_resume' 신호 감지. 연쇄 호출 실행.")

            response_state = await app.ainvoke(
                {"messages": [HumanMessage(content=" ")]},
                config=config
            )
            next_step = response_state.get('next_question')

        last_message = response_state['messages'][-1].content

        final_profile_data: Optional[ProfileData] = None

        if next_step == "done":
            print(f"[{request.session_id}] 대화 종료. 프로필 데이터를 응답에 포함합니다.")

            final_profile_data = ProfileData(
                name=response_state.get("name"),
                department=response_state.get("department"),
                age=response_state.get("age"),
                phone_number=response_state.get("phone_number"),
                positions=response_state.get("positions"),
                motivation=response_state.get("motivation")
            )
            print(final_profile_data)

        return ChatResponse(
            session_id=request.session_id,
            response_message=last_message,
            next_step=next_step,
            profile_data=final_profile_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메시지 처리 중 오류 발생: {str(e)}")
