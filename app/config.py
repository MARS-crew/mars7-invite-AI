import os
import json
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("오류: GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")
    exit()

json_path = "./mars_info.json"
try:
    with open(json_path, "r", encoding="utf-8") as f:
        club_info = json.load(f)
except FileNotFoundError:
    print(f"오류: '{json_path}' 파일을 찾을 수 없습니다.")
    print("서버를 프로젝트 루트 폴더(mars7-invite-AI)에서 실행해야 합니다.")
    exit()


CLUB_NAME = club_info.get("clubName", "동아리 이름 없음")
CLUB_INTRO = club_info.get("introduction", "동아리 소개 없음")

CLUB_POSITIONS = "기획자, 디자이너, 프론트엔드, 백엔드, AI 포지션"

data_lines = []

# 활동 내용
data_lines.append("- 주요 활동:")
for activity in club_info.get("activities", []):
    data_lines.append(f"  - {activity['name']}: {activity['description']}")

# 모집 대상
data_lines.append(f"\n- 모집 대상: {club_info.get('targetAudience', '정보 없음')}")

# 모집 기간 및 방법
recruit_info = club_info.get("recruitment", {})
data_lines.append(f"\n- 모집 기간: {recruit_info.get('period', '정보 없음')}")
data_lines.append(f"- 지원 방법: {recruit_info.get('howToApply', '정보 없음')}")

# FAQ
data_lines.append("\n- 자주 묻는 질문(FAQ):")
for faq in club_info.get("faq", []):
    data_lines.append(f"  - Q: {faq['question']} A: {faq['answer']}")

# 문의
data_lines.append(f"\n- 문의: {club_info.get('contact', '정보 없음')}")

# 모든 라인을 하나의 문자열로 결합
CLUB_DATA = "\n".join(data_lines)