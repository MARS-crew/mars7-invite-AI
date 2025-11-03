import json
import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from app.config import GOOGLE_API_KEY, CLUB_NAME, CLUB_INTRO, CLUB_POSITIONS, CLUB_DATA
from .state import ApplicationFormState, UserInfo, PositionInfo, QASessionIntent
from langgraph.graph import END

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0, api_key=GOOGLE_API_KEY)

intro_extractor = llm.with_structured_output(UserInfo)
position_extractor = llm.with_structured_output(PositionInfo)
intent_classifier_llm = llm.with_structured_output(QASessionIntent)


def start_node(state: ApplicationFormState):
    first_question = """좋았어! 이제 네 얘기도 좀 들려주라. 간단하게 자기소개 한번 해줄 수 있어?"""
    return {
        "messages": [AIMessage(content=f"{first_question}")],
        "next_question": "intro"
    }


def process_introduction(state: ApplicationFormState):
    prompt = SystemMessage(content="사용자의 최신 응답에서 이름, 학과, 나이, 전화번호를 추출해. 만약 특정 정보가 언급되지 않았다면, 그 값은 반드시 None으로 남겨둬.")
    extracted_data: UserInfo = intro_extractor.invoke([prompt] + state["messages"])
    user_name = extracted_data.name if extracted_data.name else "지원자"
    next_question = f"{user_name[-2:]}!, 그렇구나 너는 어떤 포지션에 관심 있니?"
    return {
        "messages": [AIMessage(content=next_question)],
        "name": extracted_data.name,
        "department": extracted_data.department,
        "age": extracted_data.age,
        "phone_number": extracted_data.phone_number,
        "next_question": "position"
    }


def process_position(state: ApplicationFormState):
    prompt = SystemMessage(content=f"사용자의 최신 응답에서 관심있는 포지션 목록을 추출해. 선택지는 {CLUB_POSITIONS}이야.")
    try:
        extracted_data: PositionInfo = position_extractor.invoke([prompt] + state["messages"])
        if not extracted_data.positions: raise ValueError("포지션이 선택되지 않음")
        next_question_text = "좋아! 이제 동아리에 지원하게 된 동기를 편하게 말해줄래?"
        return {
            "messages": [AIMessage(content=next_question_text)],
            "positions": extracted_data.positions,
            "next_question": "process_initial_motivation"
        }
    except Exception as e:
        retry_message = f"포지션을 제대로 이해하지 못어, {CLUB_POSITIONS} 중에서 관심있는 포지션을 다시 말해줘!"
        return {"messages": [AIMessage(content=retry_message)], "next_question": "position"}


def process_initial_motivation_node(state: ApplicationFormState):
    ack_message = """이야기해줘서 정말 고마워! 덕분에 네가 어떤 멋진 생각을 하고 있는지 잘 알 수 있었어. 자 이제부터는 내가 너의 질문에 대답해 줄 차례야 우리 동아리에 대해 궁금했던거, 활동은 어떻게 하는지 등 모든지 편하게 물어봐!"""
    initial_motivation_text = state["messages"][-1].content
    return {
        "messages": [AIMessage(content=ack_message)],
        "initial_motivation": initial_motivation_text,
        "next_question": "qa_session"
    }


def qa_session_node(state: ApplicationFormState):
    user_message = state["messages"][-1].content
    classification: QASessionIntent = intent_classifier_llm.invoke(f"사용자 메시지: '{user_message}'\n\n이 사용자의 의도를 분류하세요. ('종료', '그만', '됐어', '지원서 생성')는 'end_chat', 그 외는 'continue_chat'입니다.")

    if classification.intent == "end_chat":
        print("대화 종료 감지됨 (qa_session_node)")
        return {"next_question": "generate_resume"}

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
            당신은 [{CLUB_NAME}]의 홍보 담당 챗봇입니다.
            사용자의 질문에 친근하고 정확하게 답변하세요.
            아래 [동아리 정보]를 바탕으로 답변해야 합니다.
            [동아리 정보]
            {CLUB_DATA}
            {CLUB_INTRO}
            {CLUB_POSITIONS}
            ---
            사용자가 "종료" 신호를 보내기 전까지 대화를 계속 이어가세요.
            Markdown 헤더(##), 제목, 이모티콘, 또는 기타 서식을 절대 포함하지 마세요.
            """),
        MessagesPlaceholder(variable_name="history")
    ])
    qa_chain = qa_prompt | llm | StrOutputParser()
    response = qa_chain.invoke({"history": state["messages"][-10:]})
    return {
        "messages": [AIMessage(content=response)],
        "next_question": "qa_session"
    }


def generate_resume_node(state: ApplicationFormState):
    print("🤖 챗봇: 프로필을 생성하고 있습니다...")

    info = {
        "name": state.get("name") or "정보 없음",
        "department": state.get("department") or "정보 없음",
        "age": state.get("age") or "정보 없음",
        "phone_number": state.get("phone_number") or "정보 없음",
        "positions": ", ".join(state.get("positions") or ["미지정"]),
    }
    initial_motivation = state.get("initial_motivation")

    messages = state["messages"]
    qa_conversation = ""
    start_index = -1
    for i, msg in enumerate(messages):
        if "잘 들었습니다!" in msg.content and isinstance(msg, AIMessage):
            start_index = i + 1
            break
    if start_index != -1:
        qa_texts = [f"- {msg.content}" for msg in messages[start_index:] if isinstance(msg, HumanMessage)]
        if qa_texts:
            last_message = qa_texts[-1].replace("- ", "")
            check_intent: QASessionIntent = intent_classifier_llm.invoke(f"'{last_message}'의 의도를 분류해줘.")
            if check_intent.intent == "end_chat":
                qa_texts.pop()
        qa_conversation = "\n".join(qa_texts)
    if not qa_conversation:
        qa_conversation = "추가 질문 없음"

    resume_prompt = f"""
                            # 지시사항
                            당신은 지원자 본인 입니다. 아래 [지원자 정보]와 [대화 내역]을 바탕으로, 오직 '지원 동기 및 포부'에 대한 문단(paragraph)만 작성해 주세요.
                            [대화 내역]에 흩어져 있는 지원자의 생각과 질문들을 하나의 통일된 '지원 동기' 스토리로 엮어내는 것이 핵심입니다.

                            # [지원자 정보]
                            이름: {info['name']}
                            소속: {info['department']} ({info['age']})
                            연락처: {info['phone_number']}
                            희망 포지션: {info['positions']}

                            # [초기 지원 동기] (사용자가 처음에 밝힌 핵심 동기)
                            {initial_motivation}

                            # [추가 Q&A 내역] (이후 대화에서 드러난 관심사)
                            {qa_conversation}

                            # 출력 규칙
                            - 300자 내외로 작성하세요.
                            - 이름, 학과 등 개인정보를 반복하지 마세요.
                            - Markdown 헤더(##), 제목, 또는 기타 서식을 절대 포함하지 마세요.
                            - 오직 '지원 동기 및 포부' 문단 자체만 응답하세요.
                            """

    generated_resume = llm.invoke(resume_prompt).content
    motivation = generated_resume

    profile_data = {
        "name": state.get("name"),
        "department": state.get("department"),
        "age": state.get("age"),
        "phone_number": state.get("phone_number"),
        "positions": state.get("positions"),
        "motivation": motivation
    }

    final_message = ""
    try:
        submit_url = "https://d1ixjsazi0u8mj.cloudfront.net/submit"

        print(f"🚀 [전송 시도] Endpoint: {submit_url}\n{profile_data}")
        response = httpx.post(submit_url, json=profile_data)

        response.raise_for_status()

    except httpx.HTTPStatusError as e:
        print(f"[전송 실패] 서버가 오류를 반환: {e}")
        final_message = "프로필 생성에 성공했으나, 최종 제출 서버에 오류가 발생했습니다."
    except httpx.RequestError as e:
        print(f"[전송 실패] 서버에 연결할 수 없음: {e}")
        final_message = "프로필 생성에 성공했으나, 제출 서버에 연결할 수 없습니다."
    except Exception as e:
        print(f"[전송 중 알 수 없는 오류]: {e}")
        final_message = "프로필 생성 또는 제출 중 알 수 없는 오류가 발생했습니다."

    return {
        "messages": [AIMessage(content=final_message)],
        "motivation": motivation,
        "next_question": "done"
    }


def router(state: ApplicationFormState):
    if not state.get("messages", []):
        return "start"

    next_step = state.get("next_question", "intro")

    if next_step == "intro": return "process_introduction"
    if next_step == "position": return "process_position"
    if next_step == "process_initial_motivation": return "process_initial_motivation_node"
    if next_step == "qa_session": return "qa_session_node"
    if next_step == "generate_resume": return "generate_resume_node"
    if next_step == "done": return END