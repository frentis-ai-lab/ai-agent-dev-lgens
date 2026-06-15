"""02 - stream으로 추론-행동-관찰 루프가 번갈아 도는 모습을 단계별로 관찰한다.

이 예제 하나만으로 다음을 익힙니다.
  - 01에서 배선한 것과 같은 수동 Agent 그래프를 만든다.
  - invoke가 "최종 답만" 돌려주는 것과 달리, stream은 "매 단계의 중간 결과"를 흘려보낸다.
  - stream_mode="values"로 매 단계의 누적 상태를, stream_mode="updates"로 노드별 변경분을 본다.
  - 추론(AIMessage 도구 호출) → 행동·관찰(ToolMessage) → 다시 추론이 번갈아 나오는 ReAct 루프를 눈으로 확인한다.

LO1의 추론-행동-관찰 루프가 한 번의 invoke 안에서 여러 바퀴 돈다는 사실은
최종 답만 봐서는 보이지 않습니다. stream으로 중간 단계를 펼쳐 그 순환을 눈으로 확인합니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 06_langgraph_agent/02_react_loop_observe.py

키가 없으면 안내만 출력하고 종료합니다.
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


@tool
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱한다."""
    return a * b


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_agent_graph():
    """01과 동일한 수동 Agent 그래프(모델 → ToolNode → 조건부 엣지 → 순환)를 컴파일한다."""
    tools = [add, multiply]
    # bind_tools: 모델에게 "이 도구들을 쓸 수 있다"고 알려, 모델이 도구 호출을 답으로 낼 수 있게 묶습니다.
    model_with_tools = init_chat_model(MODEL).bind_tools(tools)

    def call_model(state: State):
        # 현재까지 쌓인 messages를 모델에 넣고, 모델이 낸 답(텍스트 또는 도구 호출)을 messages에 더합니다.
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("model", call_model)
    # ToolNode: 모델이 요청한 도구 호출을 실제로 실행해 결과(ToolMessage)를 만들어 주는 노드입니다.
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    # tools_condition: 모델의 마지막 답을 보고 분기합니다. 도구 호출이 있으면 "tools"로, 없으면 END로 보냅니다.
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")   # 도구 결과(관찰)를 들고 다시 모델로 — 이 되돌아오는 엣지가 ReAct 순환을 만듭니다.
    return builder.compile()


def observe_values(agent) -> None:
    """stream_mode='values' — 매 단계가 끝날 때마다 누적된 전체 상태를 흘려보낸다."""
    # 각 단계의 마지막 메시지를 찍으면 추론-행동-관찰이 번갈아 나오는 모습이 보입니다.
    # .stream(...)은 invoke와 입력 형태가 같지만, 결과를 한 번에 주지 않고 단계마다 하나씩 내보냅니다.
    # for ...: 는 흘러나오는 단계 결과를 하나씩 받아 반복 처리합니다.
    print("[stream_mode='values' — 매 단계의 누적 상태에서 마지막 메시지]")
    for step in agent.stream(
        {"messages": [{"role": "user", "content": "3 더하기 5를 4와 곱하면?"}]},
        stream_mode="values",
    ):
        m = step["messages"][-1]  # 이번 단계에서 새로 쌓인 메시지(추론 / 행동 / 관찰 중 하나)
        # 메시지 객체면 pretty_print, dict로 오면 그대로 출력 (버전별 형태 방어)
        m.pretty_print() if hasattr(m, "pretty_print") else print(m)

    # 예상 흐름:
    #   HumanMessage(질문) → AIMessage(add 호출=행동) → ToolMessage(8=관찰)
    #   → AIMessage(multiply 호출=행동) → ToolMessage(32=관찰) → AIMessage(최종 답변)
    # 체크포인트: 도구 호출이 더 없는 마지막 AIMessage에서 tools_condition이 END로 보내 루프가 끝납니다.


def observe_updates(agent) -> None:
    """stream_mode='updates' — 각 노드가 끝날 때마다 '그 노드가 바꾼 부분'만 흘려보낸다."""
    # values가 "전체 누적 상태"를 준다면, updates는 "어느 노드가 무엇을 바꿨는지"를 노드 이름과 함께 줍니다.
    # 모델이 어떤 도구를 부르고 무엇을 관찰했는지, 과정을 노드 단위로 따라갈 수 있습니다.
    print("\n[stream_mode='updates' — 노드별 변경분]")
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "3 더하기 5를 4와 곱하면?"}]},
        stream_mode="updates",
    ):
        # chunk는 {노드이름: {상태 변경분}} 형태입니다. 노드별로 새 메시지를 꺼내 봅니다.
        # .items()는 딕셔너리의 (키, 값) 짝을 하나씩 돌려줍니다.
        for node_name, update in chunk.items():
            messages = update.get("messages", []) if isinstance(update, dict) else []
            for m in messages:
                kind = type(m).__name__                 # AIMessage / ToolMessage 등
                calls = getattr(m, "tool_calls", None)  # 도구 호출 요청이 담겼는지
                if calls:
                    # 추론 결과로 "어떤 도구를 어떤 인자로 부를지" 정한 단계
                    print(f"  [{node_name}] {kind} 도구 호출 →", [(c["name"], c["args"]) for c in calls])
                else:
                    # 도구 실행 결과(관찰) 또는 최종 답변 텍스트
                    print(f"  [{node_name}] {kind}:", getattr(m, "content", ""))
    # 체크포인트: model 노드에서 도구 호출이, tools 노드에서 관찰값이 번갈아 출력되면
    #   ReAct 루프의 중간 과정을 스트리밍으로 들여다본 것입니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    agent = build_agent_graph()

    # 무엇을 보려는가: invoke는 "최종 답만" 주지만, stream은 "매 단계의 중간 결과"를 흘려보냅니다.
    # 같은 질문을 두 가지 stream_mode로 흘려, 추론-행동-관찰 루프가 도는 모습을 두 각도에서 봅니다.
    print("무엇을: 추론-행동-관찰(ReAct) 루프가 한 번의 호출 안에서 여러 바퀴 도는 모습을 stream으로 관찰합니다.")
    print('입력: "3 더하기 5를 4와 곱하면?" — add로 8, multiply로 32. 도구가 두 번 필요합니다.\n')

    observe_values(agent)
    observe_updates(agent)

    print("\n출력 요약: 위 흐름에서 AIMessage(도구 호출)와 ToolMessage(관찰)가 번갈아 나오다,")
    print("        도구 호출이 더 없는 마지막 AIMessage(최종 답 32)에서 루프가 끝났습니다.")


if __name__ == "__main__":
    main()
