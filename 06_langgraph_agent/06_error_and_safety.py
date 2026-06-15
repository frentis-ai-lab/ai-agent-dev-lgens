"""06 - 무한 루프의 근본 원인을 짚고, 오류 회신·재귀 한도로 도구 루프를 안전하게 지킨다.

이 예제 하나만으로 다음을 익힙니다.
  - 도구 루프가 끝나는 조건은 단 하나(모델이 도구를 그만 부름)임을 이해한다.
  - ToolNode가 기본값(handle_tool_errors=True)으로 도구 예외를 잡아, 오류를 담은 ToolMessage로
    돌려줘 모델이 인자를 고쳐 다시 시도하게 한다(자기 교정).
  - 빈 결과 대신 '읽을 수 있는 실패 메시지'를 돌려주는 것이 무한 루프의 근본 처방임을 본다.
  - recursion_limit으로 단계 수에 상한을 둬, 무엇이 잘못되든 일정 횟수에서 루프를 끊는다.

안전한 도구 루프는 세 겹으로 지킵니다. (1) 의미 있는 실패: 도구가 실패를 모델이 읽을 수 있는
메시지로 돌려줌. (2) 상한(recursion_limit): 마지막 그물. (3) 위험 작업 차단: 커스텀 그래프의
검증·승인 노드(01에서 자리만 짚었고, 여기서는 (1)·(2)에 집중합니다).

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 06_langgraph_agent/06_error_and_safety.py

키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# GraphRecursionError는 그래프가 단계 상한(recursion_limit)을 넘었을 때 던지는 예외입니다.
from langgraph.errors import GraphRecursionError
from typing_extensions import TypedDict

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


# ---------------------------------------------------------------------------
# 도구 — 빈 결과 대신 '읽을 수 있는 메시지'를 돌려주는 설계가 핵심입니다.
# ---------------------------------------------------------------------------

# 사내 문서 색인 흉내 (실무라면 검색 엔진·데이터베이스 자리입니다).
_DOCS = {
    "연차": "연차는 입사 1년차 15일, 매 2년마다 1일씩 늘어 최대 25일입니다.",
    "출장비": "출장비는 일비 3만원, 숙박비 실비(상한 10만원)로 정산합니다.",
}


@tool
def search_doc(query: str) -> str:
    """사내 문서·규정에서 query에 해당하는 내용을 찾는다."""
    # 부분 일치로 가장 단순하게 조회합니다.
    for key, text in _DOCS.items():
        if key in query:
            return text
    # 핵심: 못 찾았을 때 빈 문자열 ""을 돌려주면 모델이 '없음'을 못 읽고 같은 검색을 반복합니다.
    # 대신 읽을 수 있는 메시지를 돌려주면, 모델이 '없음'을 이해하고 솔직하게 답하며 재시도를 멈춥니다.
    return "검색 결과 없음. 해당 내용은 사내 문서에 없습니다."


@tool
def divide(a: int, b: int) -> int:
    """두 정수를 나눈다 (b가 0이면 실패한다)."""
    # 일부러 예외를 낼 수 있는 도구입니다. b=0이면 ZeroDivisionError가 납니다.
    # ToolNode가 이 예외를 잡아 오류 메시지로 돌려주면, 모델이 인자를 고쳐 다시 시도할 수 있습니다.
    return a // b


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_safe_graph():
    """ToolNode의 오류 회신을 켠 수동 그래프를 만든다 (01과 같은 배선 + 오류 처리 강조).

    create_agent로도 같은 동작을 얻지만, handle_tool_errors가 ToolNode의 옵션임을 보이려
    여기서는 ToolNode를 직접 배선합니다. 도구 실행 전 검증·승인 노드를 끼울 자리가 있는 길도 이쪽입니다.
    """
    tools = [search_doc, divide]
    model_with_tools = init_chat_model(MODEL).bind_tools(tools)

    def call_model(state: State):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("model", call_model)
    # handle_tool_errors=True(기본값)면, 도구가 예외를 던져도 ToolNode가 잡아
    # "Error: ... Please fix your mistakes." 형태의 ToolMessage로 모델에 돌려줍니다.
    # 문자열을 주면 그 문자열로, 콜러블을 주면 그 함수가 만든 메시지로 통일할 수 있고, False면 예외를 그대로 전파합니다.
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")   # 관찰을 들고 다시 추론으로(순환)
    return builder.compile()


def demo_meaningful_failure(agent) -> None:
    """둘째 겹 — 도구가 빈 결과 대신 읽을 수 있는 메시지를 돌려주면 모델이 멈춘다."""
    # 사내 문서에 없는 내용을 물어, search_doc이 "검색 결과 없음" 메시지를 돌려주게 합니다.
    print("=== 의미 있는 실패: 없는 내용을 물으면 모델이 '없다'고 솔직히 답한다 ===")
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "사내 헬스장 이용 규정 알려줘"}]}
    )
    print("최종 답변:", out["messages"][-1].content)
    # 체크포인트: "없습니다" 류로 답하고 같은 검색을 무한히 반복하지 않으면, 의미 있는 실패가 동작한 것입니다.


def demo_tool_exception_recovery(agent) -> None:
    """ToolNode의 오류 회신 — 도구가 예외를 던져도 모델이 인자를 고쳐 다시 시도한다."""
    # 0으로 나누기를 유도해, divide가 예외를 낼 수 있는 상황을 만듭니다.
    # ToolNode가 예외를 잡아 오류 메시지로 돌려주면, 모델은 그것을 읽고 스스로 정정하거나 솔직히 설명합니다.
    print("\n=== 도구 예외 회복: 예외를 오류 메시지로 받아 모델이 정정·설명한다 ===")
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "10을 0으로 나눈 값을 알려줘"}]}
    )
    print("최종 답변:", out["messages"][-1].content)
    # 체크포인트: 그래프가 예외로 죽지 않고, 0으로 나눌 수 없다는 취지로 답하면 오류 회신이 동작한 것입니다.


def demo_recursion_limit(agent) -> None:
    """첫째 겹(마지막 그물) — recursion_limit으로 단계 수에 상한을 둔다."""
    # recursion_limit은 한 번의 invoke에서 그래프가 밟을 수 있는 단계 수의 상한입니다.
    # 모델 호출 한 번, 도구 실행 한 번이 각각 한 단계로 세집니다. 한도를 넘으면 GraphRecursionError가 납니다.
    # (기본값은 라이브러리가 정한 넉넉한 수입니다. 정상 흐름에서는 닿지 않습니다.)
    print("\n=== 재귀 한도: 일부러 1로 낮춰 한도 초과를 유발한다 ===")
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": "연차 규정 알려줘"}]},
            {"recursion_limit": 1},   # 비현실적으로 낮춰, 도구를 다 돌기 전에 막히게 함
        )
        print("  (한도 안에 끝났습니다 — 모델이 도구를 거의 안 썼을 수 있습니다)")
    except GraphRecursionError as e:
        # 운영에서는 이 예외를 잡아 사용자 안내 메시지로 바꾸고, 로그로 어떤 요청이 한도에 닿았는지 남깁니다.
        print("  스텝 상한 초과로 중단(GraphRecursionError) — 운영에서는 잡아 안내·로깅:", type(e).__name__)

    print("\n=== 재귀 한도: 넉넉한 기본값이면 정상 종료한다 ===")
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "연차 규정 알려줘"}]}
    )
    print("  최종 답변:", out["messages"][-1].content)
    # 체크포인트: 낮은 한도에서 GraphRecursionError가, 기본값에서 정상 답이 나오면 안전망이 동작한 것입니다.
    #   다만 한도를 올리는 것은 마지막 그물일 뿐, 근본은 도구가 실패를 읽을 수 있게 돌려주는 것(둘째 겹)입니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return

    # 무엇을: 도구 루프를 안전하게 지키는 세 겹을 차례로 시연합니다.
    #   (둘째 겹) 의미 있는 실패 → (오류 회신) 도구 예외 회복 → (첫째 겹) 재귀 한도.
    print("무엇을: 도구 루프 안전장치 세 겹을 같은 그래프로 차례대로 시연합니다.")
    print("  1) 의미 있는 실패  2) 도구 예외 회복(handle_tool_errors)  3) 재귀 한도(recursion_limit)\n")

    agent = build_safe_graph()
    demo_meaningful_failure(agent)
    demo_tool_exception_recovery(agent)
    demo_recursion_limit(agent)

    print("\n출력 요약: 없는 내용엔 '없다'고 답하고, 0으로 나누기 예외에도 그래프가 죽지 않으며,")
    print("        한도를 1로 낮추면 GraphRecursionError로 안전하게 끊겼습니다. 세 겹이 모두 동작했습니다.")


if __name__ == "__main__":
    main()
