import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from uuid import uuid4
from langchain_core.messages import HumanMessage


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


from .bot.graph import create_app, ApplicationFormState


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("서버 시작: LangGraph 앱을 로드합니다...")
    langgraph_app = create_app()
    app.state.langgraph_app = langgraph_app
    print("LangGraph 앱 로드 완료.")
    yield
    print("서버 종료.")
    app.state.langgraph_app = None


app = FastAPI(lifespan=lifespan)


def get_langgraph_app(request: Request):
    if not hasattr(request.app.state, "langgraph_app") or not request.app.state.langgraph_app:
        raise HTTPException(status_code=500, detail="LangGraph 앱이 초기화되지 않았습니다.")
    return request.app.state.langgraph_app


@app.get("/")
def read_root():
    return {"message": "최고 코딩 동아리 챗봇 API입니다. /docs 로 이동하세요."}


@app.post("/chat/start", response_model=StartChatResponse)
async def start_chat(request: Request):
    langgraph_app = get_langgraph_app(request)
    session_id = str(uuid4())
    config = {"configurable": {"thread_id": session_id}}

    try:
        response_state: ApplicationFormState = await langgraph_app.ainvoke({}, config=config)
        last_message = response_state['messages'][-1].content
        next_step = response_state['next_question']
        return StartChatResponse(
            session_id=session_id,
            response_message=last_message,
            next_step=next_step
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대화 시작 중 오류 발생: {str(e)}")


@app.post("/chat/send", response_model=ChatResponse)
async def send_chat_message(request_body: ChatRequest, request: Request):
    langgraph_app = get_langgraph_app(request)
    config = {"configurable": {"thread_id": request_body.session_id}}

    try:
        if request_body.message.lower() in ['exit', '종료']:
            current_state: ApplicationFormState = await langgraph_app.aget_state(config)
            if current_state and current_state.values.get("next_question") == "qa_session":
                print(f"[{request_body.session_id}] '종료' 감지, 프로필 생성 시작")
                await langgraph_app.ainvoke({"messages": [HumanMessage(content="종료")]}, config=config)
                response_state = await langgraph_app.ainvoke({"messages": [HumanMessage(content="")]}, config=config)
            else:
                response_state = current_state.values
        else:
            response_state = await langgraph_app.ainvoke(
                {"messages": [HumanMessage(content=request_body.message)]},
                config=config
            )

        last_message = response_state['messages'][-1].content
        next_step = response_state['next_question']

        return ChatResponse(
            session_id=request_body.session_id,
            response_message=last_message,
            next_step=next_step
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메시지 처리 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)