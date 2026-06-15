"""03 - create_agent 한 줄로, 01에서 손으로 배선한 그래프를 그대로 만든다.

이 예제 하나만으로 다음을 익힙니다.
  - create_agent에 모델·도구·시스템 프롬프트만 주면, 01에서 손으로 짠
    모델 노드·ToolNode·조건부 엣지·되돌아오는 순환을 내부에서 똑같이 만들어 준다.
  - 한 줄 버전의 입력·출력 형태가 수동 그래프와 같음을 확인한다
    (입력은 {"messages": [...]}, 최종 답은 result["messages"][-1].content).
  - create_agent가 돌려주는 것도 결국 컴파일된 그래프(graph)임을 본다.

01에서 모델 노드·ToolNode·tools_condition·되돌아오는 엣지를 한 줄씩 끼워 만든 그래프를,
여기서는 create_agent 한 줄로 줄여 둘을 대조합니다. 같은 루프, 줄어든 보일러플레이트입니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 06_langgraph_agent/03_create_agent.py

키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os

from dotenv import load_dotenv
# create_agent는 LangChain v1.0의 권장 진입점입니다(한 줄 Agent).
# 모델 노드·도구 노드·조건 분기·순환을 내부에서 알아서 배선합니다.
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

# MODEL은 "벤더:모델명" 문자열 하나로 모델을 지정합니다. 이 줄만 바꾸면 다른 모델을 씁니다.
MODEL = "openai:gpt-5.4-mini"


# 01과 같은 도구입니다. docstring이 곧 도구 설명이며, 모델은 이 설명을 보고 어떤 도구를 부를지 정합니다.
@tool
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱한다."""
    return a * b


def build_agent():
    """create_agent 한 줄로 ReAct Agent 그래프를 만든다 (01의 수동 배선을 자동화).

    01에서는 StateGraph에 add_node·add_conditional_edges·add_edge를 여러 줄 호출해
    그래프를 손으로 조립했습니다. create_agent는 그 과정을 내부에서 똑같이 수행합니다.
    """
    # tools: 모델이 부를 수 있는 도구 목록. 01의 bind_tools + ToolNode 역할을 한 번에 처리합니다.
    # system_prompt: 모델의 역할·규칙을 정하는 고정 지시문. 매 호출 앞에 붙습니다.
    # create_agent는 이 둘을 받아 모델 노드·ToolNode·조건 분기·순환까지 갖춘 그래프를 돌려줍니다.
    agent = create_agent(
        MODEL,
        tools=[add, multiply],
        system_prompt="너는 정확한 계산 비서다. 계산은 반드시 도구로 수행하라.",
    )
    return agent


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return

    agent = build_agent()

    # create_agent가 돌려준 것도 invoke·stream을 가진 컴파일된 그래프입니다.
    # 01의 build_agent_graph()가 돌려준 객체와 같은 종류라, 쓰는 방법도 똑같습니다.
    print("agent 타입:", type(agent).__name__, "/ invoke 보유:", hasattr(agent, "invoke"))

    # 입력 형태도 01과 같습니다: {"messages": [...]}. 프리빌트 Agent의 상태가 messages 칸을 쓰기 때문입니다.
    print("\n=== 도구가 두 번 필요한 질문 (01과 같은 답이 나오는지 대조) ===")
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "3 더하기 5를 4와 곱하면?"}]}
    )
    # 최종 답은 01과 마찬가지로 result["messages"][-1].content입니다.
    print("최종 답변:", out["messages"][-1].content)

    # 반환된 messages에는 질문·도구 호출·도구 결과·최종 답이 시간 순으로 담깁니다(01과 동일).
    print("\n[누적된 메시지 흐름 — 01의 수동 그래프와 같은 모양]")
    for m in out["messages"]:
        m.pretty_print() if hasattr(m, "pretty_print") else print(m)

    # 체크포인트:
    #   - agent가 invoke를 가진 그래프로 출력되면, create_agent가 컴파일된 그래프를 돌려준 것입니다.
    #   - 최종 답에 32가 나오면, 한 줄 버전이 01의 수동 그래프와 같은 루프를 만든 것입니다.
    #   - 메시지 흐름에 AIMessage(도구 호출) → ToolMessage(관찰)가 번갈아 보이면 같은 ReAct 루프입니다.


if __name__ == "__main__":
    main()
