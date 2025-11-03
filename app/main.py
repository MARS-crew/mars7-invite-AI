import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .api.endpoints import router as api_router
from .bot.graph import create_app as create_langgraph_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("서버 시작: LangGraph 앱을 로드합니다...")
    langgraph_app = create_langgraph_app()
    app.state.langgraph_app = langgraph_app
    print("LangGraph 앱 로드 완료.")

    yield

    print("서버 종료.")
    app.state.langgraph_app = None


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "마스외전 챗봇 API입니다."}


app.include_router(api_router, prefix="")

# 로컬에서 직접 실행하기 위한 코드
if __name__ == "__main__":
    # 터미널에서 uvicorn app.main:app --reload
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)