"""06 - 순환 그래프와 recursion_limit로 무한 루프를 안전하게 끊는다.

이 예제 하나만으로 다음을 익힙니다.
  - 노드에서 자기 자신으로 되돌아오는 '순환(loop)' 그래프를 만든다.
  - 종료 조건이 없는 순환은 영원히 돈다는 사실을 체험한다.
  - recursion_limit로 단계 수 상한을 두어, 한도 초과 시 GraphRecursionError로 안전하게 멈춘다.
  - 근본 해결은 라우터에 종료 조건을 두는 것이고, recursion_limit는 마지막 안전망임을 본다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 05_langgraph_workflow/06_loop_and_recursion.py

이 예제는 모델을 부르지 않습니다(순환 구조와 안전망만 봅니다).
따라서 API 키 없이도 그대로 돌아갑니다.
"""

import operator
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
# GraphRecursionError는 단계 수 상한을 넘겼을 때 LangGraph가 던지는 예외입니다 (v1 권장 경로).
from langgraph.errors import GraphRecursionError

MODEL = "openai:gpt-5.4-mini"  # 형식을 맞추기 위한 상수. 이 예제는 모델을 부르지 않습니다.


# 모델 없이 순환만 확인하는 예제라, messages 대신 단순 카운터를 둡니다.
class CounterState(TypedDict):
    # operator.add 리듀서로 step 값이 호출마다 누적되도록 합니다 (03_reducers와 같은 원리).
    step: Annotated[int, operator.add]


def run_endless_loop_with_limit() -> None:
    """종료 조건이 없는 순환을 recursion_limit로 안전하게 끊는다."""
    def tick(state: CounterState) -> dict:
        # 매번 step에 1을 더해 돌려줍니다. 종료 조건이 없으면 영원히 반복됩니다.
        # 누적된 step은 지금까지 tick이 돈 횟수와 같으므로, 회전이 보이도록 출력합니다.
        print(f"    tick 실행: 현재 step={state['step']} → +1 반환")
        return {"step": 1}

    def loop_route(state: CounterState) -> str:
        # 의도적으로 종료 조건을 넣지 않아 항상 자기 자신으로 되돌아가게 만듭니다.
        print(f"      route → 'loop' (종료 조건 없음, step={state['step']}여도 계속 tick으로)")
        return "loop"  # 실무에서는 여기서 조건을 만족하면 "end"를 돌려줘야 합니다.

    b = StateGraph(CounterState)
    b.add_node("tick", tick)
    b.add_edge(START, "tick")
    # tick 다음 라우터가 항상 "loop"를 돌려 tick으로 되돌아오는 순환을 만듭니다.
    b.add_conditional_edges("tick", loop_route, {"loop": "tick", "end": END})
    graph = b.compile()

    # recursion_limit는 그래프가 밟을 수 있는 단계 수의 상한입니다.
    # 지정하지 않으면 라이브러리가 정한 기본 상한이 적용되며, 호출 시 직접 지정해 올리거나 내릴 수 있습니다.
    # 여기서는 낮은 값으로 내려, 무한히 도는 대신 한도 초과로 안전하게 멈추는 모습을 봅니다.
    limit = 5
    print(f"[recursion_limit={limit}] 종료 조건 없는 순환을 안전하게 끊습니다:")
    print(f"  tick → route='loop' → tick → ... 한도({limit})를 넘기면 예외로 중단됩니다:")
    # try / except는 "안쪽을 시도하다가 정해진 예외가 나면 받아서 처리하라"는 구문입니다.
    try:
        graph.invoke({"step": 0}, {"recursion_limit": limit})
    except GraphRecursionError:
        print(f"  GraphRecursionError 발생: 한도({limit})를 넘겨 안전하게 중단되었습니다.")

    # 체크포인트: GraphRecursionError가 발생하면 한도 초과로 안전하게 멈춘 것입니다.


def run_loop_with_exit_condition() -> None:
    """근본 해결 — 라우터에 종료 조건을 두면 한도에 닿기 전에 정상 종료한다."""
    def tick(state: CounterState) -> dict:
        print(f"    tick 실행: 현재 step={state['step']} → +1 반환")
        return {"step": 1}

    def loop_route(state: CounterState) -> str:
        # 이번에는 종료 조건을 둡니다: step이 3 이상이면 "end", 아니면 "loop".
        # 이것이 무한 루프의 근본 해결책입니다. recursion_limit는 이 조건이 빠졌을 때의 안전망일 뿐입니다.
        decision = "end" if state["step"] >= 3 else "loop"
        print(f"      route → '{decision}' (step={state['step']}, 종료 조건 step>=3)")
        return decision

    b = StateGraph(CounterState)
    b.add_node("tick", tick)
    b.add_edge(START, "tick")
    b.add_conditional_edges("tick", loop_route, {"loop": "tick", "end": END})
    graph = b.compile()

    # 종료 조건이 있으니 recursion_limit를 넉넉히 둬도 세 번 돈 뒤 스스로 멈춥니다.
    print("  종료 조건이 있으니 한도에 닿기 전에 스스로 멈춥니다:")
    result = graph.invoke({"step": 0}, {"recursion_limit": 100})
    print("  [종료 조건 있음] 정상 종료, 최종 step =", result["step"])  # 예: 3

    # 체크포인트: 한도에 닿기 전에 step=3에서 정상 종료하면, 종료 조건이 근본 해결임을 확인한 것입니다.
    # recursion_limit를 무작정 키우면 비용만 늘 뿐, 멈추지 않는 그래프는 대개 라우터 설계가 빠진 것입니다.


def main() -> None:
    print("=== 종료 조건 없는 순환 — recursion_limit로 안전하게 끊기 ===")
    run_endless_loop_with_limit()

    print("\n=== 종료 조건 있는 순환 — 한도에 닿기 전에 정상 종료 ===")
    run_loop_with_exit_condition()


if __name__ == "__main__":
    main()
