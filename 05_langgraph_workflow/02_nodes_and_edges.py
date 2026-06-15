"""02 - 노드를 여러 개로 늘리고, 엣지로 실행 순서를 잇는다.

이 예제 하나만으로 다음을 익힙니다.
  - 한 노드에 다 몰아넣지 않고, 단계를 노드 둘로 쪼갠다.
  - 엣지로 잇지 않은 노드는 실행되지 않음을 확인한다.
  - 엣지 한 줄(write_draft → polish)로 두 노드를 직선으로 잇는다.
  - 앞 노드가 채운 칸을 뒤 노드가 받아 다음 칸을 채운다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 05_langgraph_workflow/02_nodes_and_edges.py

노드 안에서 모델을 부르므로 OPENAI_API_KEY가 필요합니다.
키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

# os는 파이썬에 기본 내장된 모듈로, 환경변수(컴퓨터에 저장된 설정값)를 읽을 때 씁니다.
import os

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# .env 파일이 있으면 환경변수로 읽어 들입니다 (로컬 실행 시 키를 .env에 둡니다).
load_dotenv()

# 모델은 "벤더:모델명" 문자열 하나로 지정합니다. 이 줄만 바꾸면 전체 코드가 다른 모델을 씁니다.
# Gemini로 전환하려면: MODEL = "google-genai:gemini-3.5-flash" (GOOGLE_API_KEY 필요)
MODEL = "openai:gpt-5.4-mini"


# 상태에는 단계마다 산출물을 담을 칸을 따로 둡니다.
# 노드가 늘면, 노드 사이에 값을 주고받을 칸도 함께 늘어납니다.
class State(TypedDict):
    topic: str     # 입력: 글감
    draft: str     # 1단계 산출물: 초안
    polished: str  # 2단계 산출물: 다듬은 글


# build_unconnected_graph는 "polish를 등록만 하고 엣지로 잇지는 않은" 그래프를 만들어 돌려줍니다.
# 엣지로 잇지 않은 노드가 실행되지 않음을 먼저 눈으로 확인하려는 의도입니다.
def run_unconnected(model) -> None:
    """노드를 2개 등록하되, START는 1단계에만 잇는다 (polish는 실행되지 않음)."""
    def write_draft(state: State) -> dict:
        # 1단계 노드: 주제로 한 문장 초안을 만듭니다. draft 칸만 갱신합니다.
        # model.invoke(...)는 모델을 한 번 부르는 호출이고, .content로 답 텍스트를 꺼냅니다.
        resp = model.invoke(f"'{state['topic']}'에 대한 한 문장 초안을 써줘.")
        return {"draft": resp.content}

    def polish(state: State) -> dict:
        # 2단계 노드를 정의는 해 둡니다. 이 함수는 엣지로 잇지 않으면 호출되지 않습니다.
        resp = model.invoke(f"다음 문장을 더 매끄럽게 다듬어줘: {state['draft']}")
        return {"polished": resp.content}

    b = StateGraph(State)
    b.add_node("write_draft", write_draft)
    b.add_node("polish", polish)        # 등록은 해 두되,
    b.add_edge(START, "write_draft")    # 아직 START는 1단계에만 연결합니다.
    b.add_edge("write_draft", END)      # 1단계만 돌고 끝납니다.
    graph = b.compile()

    result = graph.invoke({"topic": "재택근무의 장점", "draft": "", "polished": ""})
    print("[초안]     ", result["draft"])
    # or 뒤의 값은 앞 값이 비어 있을(빈 문자열) 때 대신 출력할 안내입니다.
    print("[다듬은 글]", result["polished"] or "(아직 비어 있음 — polish를 엣지로 잇지 않았습니다)")

    # 체크포인트: 노드를 등록만 하고 엣지로 잇지 않으면 그 노드는 실행되지 않음을 확인하면 됩니다.


def run_connected(model) -> None:
    """엣지 한 줄로 두 노드를 직선으로 잇는다 (엣지가 곧 실행 순서)."""
    # 노드 함수는 위와 같습니다. 달라지는 것은 엣지 한 줄뿐입니다.
    def write_draft(state: State) -> dict:
        resp = model.invoke(f"'{state['topic']}'에 대한 한 문장 초안을 써줘.")
        return {"draft": resp.content}

    def polish(state: State) -> dict:
        # 1단계가 채운 draft를 받아 더 매끄럽게 다듬습니다.
        resp = model.invoke(f"다음 문장을 더 매끄럽게 다듬어줘: {state['draft']}")
        return {"polished": resp.content}

    b = StateGraph(State)
    b.add_node("write_draft", write_draft)
    b.add_node("polish", polish)
    b.add_edge(START, "write_draft")     # 시작 → 1단계
    b.add_edge("write_draft", "polish")  # 1단계 → 2단계 (이 한 줄이 두 노드를 잇습니다)
    b.add_edge("polish", END)            # 2단계 → 종료
    graph = b.compile()

    result = graph.invoke({"topic": "재택근무의 장점", "draft": "", "polished": ""})
    print("[초안]     ", result["draft"])
    print("[다듬은 글]", result["polished"])

    # 체크포인트: draft가 먼저 채워지고 그 값을 polish가 받아 다듬으면 순서 연결이 동작한 것입니다.


def main() -> None:
    # if는 "조건이 참일 때만 안쪽 블록을 실행하라"는 분기문입니다. 들여쓰기로 블록 범위를 표시합니다.
    # not은 참/거짓을 뒤집습니다. 즉 "키가 없으면(not ...)" 안쪽을 실행합니다.
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return  # return은 함수를 여기서 끝내고 빠져나간다는 뜻입니다.

    # 모델은 한 번만 만들어 두 예시가 함께 씁니다.
    model = init_chat_model(MODEL)

    print("=== 엣지로 잇지 않은 노드는 실행되지 않는다 ===")
    run_unconnected(model)

    print("\n=== 엣지로 두 노드를 직선으로 잇는다 ===")
    run_connected(model)


if __name__ == "__main__":
    main()
