"""03 - 리듀서로 상태를 '덮어쓸지 누적할지' 정한다 (add_messages·operator.add).

이 예제 하나만으로 다음을 익힙니다.
  - 리듀서(reducer)가 무엇인지: 노드 반환값을 기존 상태에 어떻게 합칠지 정하는 규칙.
  - 리듀서가 없으면 기본은 '덮어쓰기'라서 이전 값이 통째로 사라진다.
  - Annotated[list, add_messages]를 붙이면 메시지가 '누적'된다.
  - 같은 원리로 Annotated[int, operator.add]는 숫자를 호출마다 더해 쌓는다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 05_langgraph_workflow/03_reducers.py

이 예제는 모델을 부르지 않습니다(미리 만든 메시지로 누적·덮어쓰기 차이만 봅니다).
따라서 API 키 없이도 그대로 돌아갑니다.
"""

# operator는 +, * 같은 연산을 함수처럼 쓸 수 있게 해 주는 기본 모듈입니다.
# operator.add는 "두 값을 더하는 함수"라서, 숫자 칸의 리듀서(합치기 규칙)로 쓸 수 있습니다.
import operator

# Annotated[타입, 부가정보]는 "타입에 메모를 덧붙이는" 표기입니다.
# LangGraph는 그 메모 자리에 리듀서 함수를 넣어 "이 칸은 이렇게 합쳐라"라고 읽습니다.
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
# add_messages는 메시지 목록 전용 리듀서입니다 (v1 권장 경로).
from langgraph.graph.message import add_messages

MODEL = "openai:gpt-5.4-mini"  # 형식을 맞추기 위한 상수. 이 예제는 모델을 부르지 않습니다.


def run_overwrite() -> None:
    """리듀서가 없으면 노드 반환값이 기존 상태를 통째로 덮어쓴다."""
    # messages를 평범한 list로 두면, 리듀서가 없으므로 기본 동작은 '덮어쓰기'입니다.
    class State(TypedDict):
        messages: list

    def overwrite(state: State) -> dict:
        # 새 메시지 하나만 돌려주면, 리듀서가 없으니 입력 메시지가 사라지고 이 값으로 교체됩니다.
        return {"messages": [AIMessage("덮어썼습니다")]}

    b = StateGraph(State)
    b.add_node("overwrite", overwrite)
    b.add_edge(START, "overwrite")
    b.add_edge("overwrite", END)
    graph = b.compile()

    result = graph.invoke({"messages": [HumanMessage("이 입력은 사라집니다")]})
    print("[리듀서 없음] 메시지 수:", len(result["messages"]))  # 예: 1 (입력이 덮어써져 사라짐)

    # 체크포인트: 입력 메시지가 사라지고 1개만 남으면, 기본 동작이 '덮어쓰기'임을 확인한 것입니다.


def run_accumulate() -> None:
    """add_messages 리듀서를 붙이면 메시지가 누적된다 (위와 비교)."""
    # 같은 그래프인데 messages 타입에 add_messages 리듀서만 붙입니다.
    # 이 한 줄이 '덮어쓰기'를 '누적'으로 바꿉니다.
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def add_one(state: State) -> dict:
        # 새 메시지 하나를 추가합니다. add_messages 덕분에 기존 입력 메시지는 보존됩니다.
        # 모델 호출 대신 미리 만든 AIMessage를 붙여, 누적 동작 자체에 집중합니다.
        return {"messages": [AIMessage("누적했습니다")]}

    b = StateGraph(State)
    b.add_node("add_one", add_one)
    b.add_edge(START, "add_one")
    b.add_edge("add_one", END)
    graph = b.compile()

    result = graph.invoke({"messages": [HumanMessage("이 입력은 남습니다")]})
    print("[리듀서 있음] 메시지 수:", len(result["messages"]))  # 예: 2 (입력 + 새 메시지가 누적)
    # 누적된 메시지를 하나씩 꺼내 어떤 순서로 쌓였는지 봅니다.
    for m in result["messages"]:
        # m.type은 메시지 종류(human/ai 등), m.content는 그 내용입니다.
        print(f"   - [{m.type}] {m.content}")

    # 체크포인트: 같은 1회 실행인데 위(덮어쓰기)는 1개, 여기(누적)는 2개가 나오면
    # 누적과 덮어쓰기의 차이를 확인한 것입니다. 대화 맥락이 필요하면 add_messages가 필수입니다.


def run_number_reducer() -> None:
    """리듀서는 메시지뿐 아니라 어떤 타입에도 붙는다 (숫자 합산 예시)."""
    # 숫자 칸에 operator.add를 리듀서로 붙이면, 노드가 부분 값을 돌려줘도 그래프가 더해 쌓습니다.
    class State(TypedDict):
        total: Annotated[int, operator.add]  # 호출마다 더해지는 누적 합

    def add_three(state: State) -> dict:
        # 노드는 '더할 부분 값'만 돌려줍니다. 합산은 리듀서(operator.add)가 맡습니다.
        return {"total": 3}

    b = StateGraph(State)
    b.add_node("add_three", add_three)
    b.add_edge(START, "add_three")
    b.add_edge("add_three", END)
    graph = b.compile()

    # 시작값 10에 노드가 돌려준 3이 '더해져' 13이 됩니다(덮어쓰기였다면 3이 됩니다).
    result = graph.invoke({"total": 10})
    print("[숫자 리듀서] 10 + 3 =", result["total"])  # 예: 13

    # 체크포인트: 결과가 3이 아니라 13이면, 숫자 칸도 리듀서로 '합치기'가 됨을 확인한 것입니다.


def main() -> None:
    print("=== 리듀서 없음 — 반환값이 기존 상태를 덮어쓴다 ===")
    run_overwrite()

    print("\n=== add_messages 리듀서 — 메시지가 누적된다 ===")
    run_accumulate()

    print("\n=== operator.add 리듀서 — 숫자도 합쳐 쌓인다 ===")
    run_number_reducer()


if __name__ == "__main__":
    main()
