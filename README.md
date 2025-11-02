Markdown

### 📁 파일 구조

```
/mars7-invite-AI
├── app/
│ ├── api/ (1. 웹 API (FastAPI) 폴더)
│ │ ├── endpoints.py     # API 경로 (/chat/start, /submit 등)
│ │ └── models.py        # API 입/출력 Pydantic 모델
│ │
│ ├── bot/ (2. 챗봇 로직 (LangGraph) 폴더)
│ │ ├── graph.py         # 챗봇의 '설계도/조립'
│ │ ├── nodes.py         # 챗봇의 '실제 행동/부품'
│ │ └── state.py         # 챗봇의 '데이터 구조/기억'
│ │
│ ├── config.py (3. 중앙 설정 파일)
│ └── main.py (4. FastAPI 서버 실행 파일)
│
├── mars_info.json      # 챗봇이 참조하는 동아리 정보 원본
├── .env                # (직접 생성) API 키 저장
├── requirements.txt    # 필요한 파이썬 패키지 목록
└── README.md (현재 파일)
```

---

### 🚀 실행 방법

프로젝트 루트 폴더(`mars7-invite-AI`)에서 아래 5단계를 순서대로 실행하세요.

**1. (필수) Python 3.8+ 가상환경 생성 및 활성화**
```bash
1. 가상환경 생성
python3 -m venv .venv

2. 활성화 (macOS/Linux)
source .venv/bin/activate
(Windows의 경우: .\.venv\Scripts\activate)
```

**2. 라이브러리 설치**
```
pip install -r requirements.txt
```

**3. .env 파일 생성 루트 폴더에 .env 파일을 만들고 아래 내용을 입력합니다.**
```
GOOGLE_API_KEY="여기에_발급받은_Gemini_API_키를_입력하세요"
```

**4. 동아리 정보 수정 mars_info.json 파일의 내용을 원하는 정보로 수정합니다.**

**5. 서버 실행**
```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
