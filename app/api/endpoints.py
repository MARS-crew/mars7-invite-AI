import json
from fastapi import APIRouter, HTTPException, Request, Depends
from uuid import uuid4
from langchain_core.messages import HumanMessage
from .models import StartChatResponse, ChatRequest, ChatResponse
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
        if request.message.lower() in ['exit', '종료']:
            current_state: ApplicationFormState = await app.aget_state(config)

            if current_state and current_state.values.get("next_question") == "qa_session":
                print(f"[{request.session_id}] '종료' 감지, 프로필 생성 시작")
                await app.ainvoke({"messages": [HumanMessage(content="종료")]}, config=config)
                response_state = await app.ainvoke({"messages": [HumanMessage(content="")]}, config=config)
            else:
                response_state = current_state.values

        else:
            response_state = await app.ainvoke(
                {"messages": [HumanMessage(content=request.message)]},
                config=config
            )

        last_message = response_state['messages'][-1].content
        next_step = response_state['next_question']

        return ChatResponse(
            session_id=request.session_id,
            response_message=last_message,
            next_step=next_step
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메시지 처리 중 오류 발생: {str(e)}")