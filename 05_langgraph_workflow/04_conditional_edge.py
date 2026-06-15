"""04 - 조건부 엣지로 흐름을 갈라 본다 (add_conditional_edges와 라우터).

이 예제 하나만으로 다음을 익힙니다.
  - 라우터(router)가 무엇인지: 상태를 보고 '다음에 갈 곳'을 문자열 키로 돌려주는 함수.
  - 그래프에 붙이기 전, 입력만 바꿔 가며 라우터가 어떤 키를 돌려주는지 본다(모델 없이).
  - add_conditional_edges로 라우터를 노드 뒤에 달아, 키를 노드/END에 매핑한다.
  - 짧은 입력은 모델 호출 없이 바로 종료되고, 긴 입력만 요약 노드를 거친다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 05_langgraph_workflow/04_conditional_edge.py

조건부 엣지 자체는 모델 없이 동작하지만, 요약 노드가 모델을 부르므로
실제 분기 실행에는 OPENAI_API_KEY가 필요합니다. 라우터 함수만 보는 첫 부분은 키 없이도 됩니다.
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


def show_router_only() -> None:
    """라우터 함수가 무엇인지부터 본다 (그래프에 붙이기 전, 모델 없이)."""
    # 라우터는 "상태를 보고 다음에 어디로 갈지"를 문자열 키로 돌려주는 평범한 함수입니다.
    class State(TypedDict):
        text: str

    def route(state: State) -> str:
        # 길이가 20자를 넘으면 "summarize", 아니면 "end"라는 키를 돌려줍니다.
        # len(...)은 글자 수를 셉니다. 삼항식 "A if 조건 else B"는 조건이 참이면 A, 아니면 B입니다.
        return "summarize" if len(state["text"]) > 20 else "end"

    print("[짧은 입력] route ->", route({"text": "안녕"}))
    print("[긴 입력]   route ->", route({"text": "오늘 회의에서 신제품 출시 일정과 마케팅 예산을 논의했다"}))

    # 체크포인트: 입력에 따라 라우터가 다른 문자열을 돌려주면, 라우터는 '분기 판단 함수'임을 이해한 것입니다.


def run_conditional_graph(model) -> None:
    """라우터를 조건부 엣지로 그래프에 연결한다 (add_conditional_edges)."""
    # add_edge가 "항상 다음 노드로"라면, add_conditional_edges는 "상황에 따라 다른 노드로" 보냅니다.
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def classify(state: State) -> dict:
        # 조건부 엣지는 출발 노드가 필요해 둔 통과(pass-through) 노드입니다.
        # 상태를 바꾸지 않고, 실제 분기 판단은 아래 route 라우터가 맡습니다.
        return {"messages": []}  # 빈 리스트를 돌려주므로 상태가 그대로 유지됩니다.

    def route(state: State) -> str:
        # 라우터가 돌려주는 문자열이 곧 '다음에 갈 곳'의 키입니다.
        # messages[-1]은 가장 마지막 메시지이고, .content는 그 내용 텍스트입니다.
        last = state["messages"][-1].content
        return "summarize" if len(last) > 20 else "end"

    def summarize(state: State) -> dict:
        # 긴 입력만 이 요약 노드를 거칩니다.
        resp = model.invoke(
            [HumanMessage(f"다음을 한 줄로 요약해줘: {state['messages'][-1].content}")]
        )
        return {"messages": [resp]}

    b = StateGraph(State)
    b.add_node("classify", classify)
    b.add_node("summarize", summarize)
    b.add_edge(START, "classify")
    # 라우터 반환값("summarize"/"end")을 실제 노드 또는 END에 매핑합니다 (세 번째 인자 = path_map).
    # 매핑을 생략하면 반환값이 곧 노드 이름이 되지만, 명시하면 의도가 분명해집니다.
    b.add_conditional_edges("classify", route, {"summarize": "summarize", "end": END})
    b.add_edge("summarize", END)
    graph = b.compile()

    print("[긴 입력] summarize 노드를 거칩니다:")
    long_result = graph.invoke(
        {"messages": [HumanMessage("오늘 회의에서 신제품 출시 일정과 마케팅 예산을 논의했다")]}
    )
    print("  ", long_result["messages"][-1].content)

    print("[짧은 입력] summarize 없이 바로 종료합니다:")
    short_result = graph.invoke({"messages": [HumanMessage("안녕")]})
    print("  ", short_result["messages"][-1].content)  # 모델 호출 없이 입력 그대로 반환

    # 체크포인트: 짧은 입력은 route가 "end"를 돌려 모델 호출 없이 끝나면 분기가 동작한 것입니다.
    # 변형 포인트: route의 기준 길이(20)만 바꾸면 분기 경계가 이동합니다.
    #   라우터 함수만 고치면 그래프 구조를 건드리지 않고 흐름을 조정할 수 있다는 점이 핵심입니다.


def main() -> None:
    # 라우터 함수만 보는 첫 부분은 모델이 필요 없으므로 키 검사보다 먼저 실행합니다.
    print("=== 라우터 함수만 보기 (모델 없음) ===")
    show_router_only()

    # 여기부터는 요약 노드가 모델을 부릅니다. 키가 없으면 안내만 하고 끝냅니다.
    if not os.getenv("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY가 없어 조건부 엣지 실행 예시는 건너뜁니다.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return

    model = init_chat_model(MODEL)
    print("\n=== 라우터를 조건부 엣지로 연결 ===")
    run_conditional_graph(model)


if __name__ == "__main__":
    main()
