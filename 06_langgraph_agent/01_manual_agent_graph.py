"""01 - 수동 Agent 그래프를 한 줄씩 배선해 추론-행동-관찰 순환을 완성한다.

이 예제 하나만으로 다음을 익힙니다.
  - 도구를 모아 모델에 바인딩한다 (bind_tools).
  - 모델 노드(추론) → ToolNode(행동) → 조건부 엣지(tools_condition) → 되돌아오는 엣지로
    추론-행동-관찰 순환을 가진 Agent 그래프를 StateGraph로 직접 배선한다.
  - 그래프 부품을 한 줄씩 끼워 가며, 어느 부품이 ReAct 루프의 어느 단계를 맡는지 본다.

LO1에서 개념으로 배운 Agent의 추론(Reasoning)-행동(Action)-관찰(Observation) 루프를
여기서는 "코드로 옮깁니다". 모델 노드가 추론하고, ToolNode가 행동하며,
도구 결과(관찰)를 들고 다시 모델로 돌아가는 순환이 곧 ReAct 루프입니다.

이 파일은 자기완결입니다. 다른 파일이나 main에 의존하지 않으며, 아래 한 줄로 단독 실행됩니다.
  uv run python 06_langgraph_agent/01_manual_agent_graph.py

키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

# import는 "다른 파일(라이브러리)에 있는 기능을 끌어와 이 파일에서 쓰겠다"는 선언입니다.
import os
# typing의 Annotated는 "타입에 부가 정보를 덧붙이는" 도구입니다. 아래 State에서 리듀서를 붙일 때 씁니다.
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# @tool 데코레이터는 평범한 함수를 "모델이 부를 수 있는 도구"로 감쌉니다(LO4에서 익힌 개념).
from langchain.tools import tool

# StateGraph는 상태·노드·엣지로 그래프를 조립하는 빌더입니다. START는 진입점 특수 노드입니다.
# (END는 tools_condition이 내부에서 종착점으로 쓰므로, 여기서는 직접 import하지 않습니다.)
from langgraph.graph import StateGraph, START
# add_messages는 messages 칸을 "덮어쓰지 않고 누적"하게 만드는 리듀서입니다(LO5에서 익힌 개념).
from langgraph.graph.message import add_messages
# ToolNode·tools_condition은 LangGraph가 미리 만들어 둔 부품입니다(도구 실행 노드 · 도구 호출 유무 분기).
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

# .env 파일이 있으면 환경변수로 읽어 들입니다 (로컬 실행 시 키를 .env에 둡니다).
load_dotenv()

# MODEL은 "벤더:모델명" 문자열 하나로 모델을 지정합니다. 이 줄만 바꾸면 전체 코드가 다른 모델을 씁니다.
# Gemini로 전환하려면: MODEL = "google-genai:gemini-3.5-flash" (GOOGLE_API_KEY 필요)
MODEL = "openai:gpt-5.4-mini"


# ---------------------------------------------------------------------------
# 도구 — LO4에서 만든 것과 같은 형태입니다. docstring이 곧 도구 설명이며,
#        모델은 이 설명을 보고 어떤 도구를 부를지 정합니다(라우팅 기준).
# ---------------------------------------------------------------------------

@tool
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱한다."""
    return a * b


# messages 칸은 덮어쓰지 않고 누적되도록 add_messages 리듀서를 지정합니다.
# 추론·행동·관찰이 메시지로 차곡차곡 쌓여야 순환이 맥락을 이어 갈 수 있습니다.
# class는 "여러 값을 묶은 새 타입"을 정의합니다. TypedDict는 "정해진 키를 가진 딕셔너리"의 타입입니다.
class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_agent_graph():
    """모델 노드·ToolNode·조건부 엣지·되돌아오는 엣지를 배선해 Agent 그래프를 컴파일한다.

    아래 add_node·add_edge 호출 순서가 곧 그래프 부품을 한 줄씩 끼워 가는 과정입니다.
    주석의 (1)~(5) 번호를 따라가면 어떤 줄이 ReAct 루프의 어느 부분을 만드는지 보입니다.
    """
    # 도구를 모으고, bind_tools로 도구 스키마를 모델에 알려 줍니다.
    # 이제 모델은 답을 바로 내거나, "이 도구를 이런 인자로 부르라"는 도구 호출(Action) 요청을
    # 담은 AIMessage를 돌려줄 수 있습니다.
    tools = [add, multiply]
    model = init_chat_model(MODEL)
    model_with_tools = model.bind_tools(tools)

    # 모델 노드: 추론(Reason)을 담당합니다. 답을 바로 내거나, 도구 호출을 요청하는 AIMessage를 돌려줍니다.
    # 노드는 상태를 받아 "바뀐 부분만" 딕셔너리로 돌려줍니다. 여기서는 messages 칸에 새 응답을 얹습니다.
    def call_model(state: State):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    # 빌더에 부품을 하나씩 끼웁니다.
    builder = StateGraph(State)
    # (1) 모델 노드(추론)를 등록합니다.
    builder.add_node("model", call_model)
    # (2) ToolNode(행동)를 등록합니다. 직전 AIMessage의 tool_calls를 읽어 도구를 실제로 실행하고,
    #     그 결과를 ToolMessage(관찰)로 상태에 쌓아 줍니다. 손으로 짜던 for 루프를 이 부품이 대신합니다.
    builder.add_node("tools", ToolNode(tools))
    # (3) 진입점은 항상 모델(추론)부터입니다.
    builder.add_edge(START, "model")
    # (4) 조건부 엣지: tools_condition은 마지막 AIMessage에 도구 호출이 있으면 "tools"로,
    #     없으면 END로 보내는 사전 제작 분기 함수입니다. 덕분에 도구가 필요할 때만 도구로 갑니다.
    builder.add_conditional_edges("model", tools_condition)
    # (5) 되돌아오는 엣지: 도구 결과(관찰)를 들고 다시 추론으로 돌아갑니다. 이 한 줄이 ReAct 루프의 "순환"입니다.
    #     이 줄을 지우면 도구 실행 후 모델로 돌아가지 못해 순환이 끊기고, 결과를 정리해 최종 답을 만드는 단계가 사라집니다.
    builder.add_edge("tools", "model")

    # compile()이 빌더를 실행 가능한 그래프로 굳힙니다. 컴파일 전에는 invoke를 쓸 수 없습니다.
    return builder.compile()


def main() -> None:
    # 키가 없으면 호출 단계에서 실패합니다. 미리 안내하고 종료합니다.
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return

    agent = build_agent_graph()

    # 1) 도구가 필요 없는 인사말. 분기(tools_condition)가 곧장 END로 보내 도구를 거치지 않습니다.
    print("=== 도구가 필요 없는 질문 (분기가 바로 END로) ===")
    chat = agent.invoke({"messages": [{"role": "user", "content": "안녕!"}]})
    print("최종 답변:", chat["messages"][-1].content)

    # 2) 도구가 두 번 필요한 계산. add 한 번, multiply 한 번을 거쳐야 풀립니다.
    #    순환이 이어져야 마지막에 사람 문장으로 정리한 최종 답(32 포함)이 나옵니다.
    print("\n=== 도구가 두 번 필요한 질문 (추론-행동-관찰 순환) ===")
    calc = agent.invoke(
        {"messages": [{"role": "user", "content": "3 더하기 5를 4와 곱하면?"}]}
    )
    print("최종 답변:", calc["messages"][-1].content)

    # 쌓인 메시지를 처음부터 보면 순환의 흔적이 그대로 남아 있습니다.
    print("\n[누적된 메시지 흐름]")
    for m in calc["messages"]:
        # 메시지 객체면 pretty_print, dict로 오면 그대로 출력 (버전별 형태 방어)
        m.pretty_print() if hasattr(m, "pretty_print") else print(m)

    # 체크포인트:
    #   - 인사말이 도구를 거치지 않고 바로 답하면 tools_condition 분기가 동작한 것입니다.
    #   - 계산 답변에 32가 나오면 추론-행동-관찰 순환이 한 바퀴 이상 돌아간 것입니다.
    #   - 메시지 흐름에 AIMessage(도구 호출) → ToolMessage(관찰)가 번갈아 보이면 순환을 눈으로 확인한 것입니다.


if __name__ == "__main__":
    main()
