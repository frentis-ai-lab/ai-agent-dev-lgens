"""02 - checkpointer(InMemorySaver) 한 줄로 단기 메모리를 켠다.

이 예제 하나만으로 다음을 익힙니다.
  - InMemorySaver를 만들어 create_agent에 checkpointer로 넘긴다 (단 한 줄 추가).
  - checkpointer가 "매 호출마다 대화 상태를 저장·복원하는 부품"임을 이해한다.
  - 01 예제와 똑같은 코드인데 checkpointer 한 줄 차이로 결과가 달라짐을 본다.

주의: checkpointer만 붙인다고 끝이 아닙니다. "어느 대화에 저장할지"를 정하는 thread_id가 필요합니다.
      이 예제는 thread_id 한 개(user-123)만 써서 멀티턴이 이어지는 것을 확인하고,
      thread_id의 분리·격리는 다음 예제(03)에서 자세히 다룹니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 07_short_memory/02_checkpointer.py

키가 없으면 안내만 출력하고 종료합니다.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
# InMemorySaver는 대화 상태(누적 메시지)를 "프로세스 메모리(RAM)"에 저장하는 checkpointer입니다.
# 가장 단순한 saver라 학습·데모에 적합합니다. 재시작하면 사라집니다(영속 저장은 06 예제에서).
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


@tool
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    # 1) checkpointer를 하나 만듭니다.
    #    checkpointer는 매 호출마다 "대화 상태(메시지 누적)"를 저장하고, 다음 호출에서 복원하는 부품입니다.
    #    InMemorySaver는 그 상태를 RAM에 둡니다. 프로세스를 끄면 사라집니다(데모용).
    checkpointer = InMemorySaver()

    # 2) 01 예제와 똑같은 에이전트인데, checkpointer 인자 한 줄만 더했습니다.
    #    이 한 줄이 단기 메모리를 켭니다.
    agent = create_agent(
        MODEL,
        tools=[add],
        system_prompt="너는 친절한 한국어 비서다.",
        checkpointer=checkpointer,  # 이 한 줄이 단기 메모리를 켭니다.
    )
    print("[에이전트]", type(agent).__name__, "(checkpointer 부착 완료)")

    # 3) thread_id를 정합니다. checkpointer만으로는 "어느 대화에 저장할지"를 모릅니다.
    #    thread_id가 그 대화방 식별자입니다. config의 정해진 자리에 넣어 호출 때마다 함께 넘깁니다.
    #    형식이 정확해야 합니다: {"configurable": {"thread_id": "..."}}
    config = {"configurable": {"thread_id": "user-123"}}

    # 4) 첫 번째 턴: 이름을 알려 줍니다. 이 대화가 thread_id=user-123에 저장됩니다.
    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "내 이름은 앤디야. 기억해 줘."}]},
        config,  # 두 번째 인자로 config를 넘기면, 그 thread_id에 상태가 저장됩니다.
    )
    print("[1턴]", r1["messages"][-1].content)  # 예: 네, 앤디님 기억하겠습니다.

    # 5) 두 번째 턴: 같은 config(같은 thread_id)로 이름을 물어봅니다.
    #    저장돼 있던 1턴 메시지가 복원되어 이번 입력 앞에 붙으므로, 모델은 앞 대화를 봅니다.
    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]},
        config,  # 01 예제와 달리, 같은 thread_id를 넘기는 것이 핵심입니다.
    )
    print("[2턴]", r2["messages"][-1].content)  # "앤디"라고 답하면 단기 메모리가 동작한 것입니다.

    # 6) 누적 메시지가 실제로 쌓였는지 확인합니다.
    #    1턴(user·ai) + 2턴(user·ai) = 4개 이상이면 상태가 이어진 것입니다.
    print("[누적 메시지 수]", len(r2["messages"]))

    # 체크포인트:
    #   - 2턴에서 "앤디"가 나오고 누적 수가 4 이상이면 멀티턴 맥락 유지에 성공한 것입니다.
    #   - 01 예제와 비교하십시오. 같은 질문인데 checkpointer 한 줄 + 같은 thread_id 차이로 결과가 달라졌습니다.


if __name__ == "__main__":
    main()
