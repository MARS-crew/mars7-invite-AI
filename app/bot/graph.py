from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from app.bot.state import ApplicationFormState
from app.bot.nodes import (
    start_node,
    process_introduction,
    process_position,
    process_initial_motivation_node,
    qa_session_node,
    generate_resume_node,
    router
)


def create_app():
    memory = InMemorySaver()

    workflow = StateGraph(ApplicationFormState)

    workflow.add_node("start", start_node)
    workflow.add_node("process_introduction", process_introduction)
    workflow.add_node("process_position", process_position)
    workflow.add_node("process_initial_motivation_node", process_initial_motivation_node)
    workflow.add_node("qa_session_node", qa_session_node)
    workflow.add_node("generate_resume_node", generate_resume_node)

    workflow.set_conditional_entry_point(router)

    workflow.add_edge("start", END)
    workflow.add_edge("process_introduction", END)
    workflow.add_edge("process_position", END)
    workflow.add_edge("process_initial_motivation_node", END)
    workflow.add_edge("qa_session_node", END)
    workflow.add_edge("generate_resume_node", END)

    app = workflow.compile(checkpointer=memory)

    print("LangGraph 앱이 컴파일되었습니다. (In-Memory Checkpointer 사용)")
    return app
