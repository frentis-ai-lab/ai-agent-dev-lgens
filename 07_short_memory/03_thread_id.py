"""03 - thread_id로 대화의 경계를 긋는다 (같은 thread는 잇고, 다른 thread는 끊는다).

이 예제 하나만으로 다음을 익힙니다.
  - 같은 thread_id로 여러 턴을 주고받으면 맥락이 계속 이어진다.
  - thread_id를 바꾸면 앞 대화가 보이지 않는다 (버그가 아니라 의도된 격리).
  - 실제 서비스에서는 사용자 ID와 대화방 ID를 조합한 "세션 키"를 thread_id로 쓴다.

thread_id는 입력 메시지가 아니라 {"configurable": {"thread_id": ...}} 설정으로 넘깁니다.
대화의 경계를 긋는 주체는 모델이 아니라, thread_id를 정하는 우리입니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 07_short_memory/03_thread_id.py

키가 없으면 안내만 출력하고 종료합니다.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


@tool
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


# 실제 서비스에서는 thread_id를 단순 문자열 하나로 두지 않고, 의미 있는 키를 조합합니다.
# 백엔드의 "세션 키"와 같은 역할입니다: 흩어진 호출에 연속성을 부여하는 식별자입니다.
def make_thread_id(user_id: str, room_id: str) -> str:
    """사용자별·대화방별로 대화를 격리하는 합성 키를 만든다."""
    # 한 직원(user_id)이 설비 상담방(room_id=room-7)과 휴가 문의방(room-9)을 동시에 열어도
    # 서로 다른 thread_id가 만들어져 두 대화가 섞이지 않습니다.
    return f"{user_id}:{room_id}"


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    # checkpointer 하나를 만들어 에이전트에 붙입니다 (02 예제와 동일).
    agent = create_agent(
        MODEL,
        tools=[add],
        system_prompt="너는 친절한 한국어 비서다.",
        checkpointer=InMemorySaver(),
    )

    # --- 1) 같은 thread_id로 두 턴: 맥락이 이어진다 ---
    # make_thread_id로 "emp-042 직원의 room-7 상담방" 키를 만듭니다.
    config_a = {"configurable": {"thread_id": make_thread_id("emp-042", "room-7")}}
    print("[thread A]", config_a["configurable"]["thread_id"])

    agent.invoke(
        {"messages": [{"role": "user", "content": "내 이름은 앤디야. 기억해 줘."}]},
        config_a,
    )
    r_a = agent.invoke(
        {"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]},
        config_a,  # 같은 thread_id이면 앞 대화가 복원되어 이어집니다.
    )
    print("  [같은 thread 2턴]", r_a["messages"][-1].content)  # "앤디"라고 답하면 정상

    # --- 2) thread_id를 바꾸면: 맥락이 끊긴다 (의도된 격리) ---
    # 같은 직원이라도 다른 대화방(room-9)은 별개의 thread_id이므로 백지에서 출발합니다.
    config_b = {"configurable": {"thread_id": make_thread_id("emp-042", "room-9")}}
    print("[thread B]", config_b["configurable"]["thread_id"])

    r_b = agent.invoke(
        {"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]},
        config_b,  # 다른 thread_id이므로 A의 기억은 여기에 없습니다.
    )
    print("  [다른 thread]", r_b["messages"][-1].content)  # 이름을 모른다고 답하면 정상

    # 체크포인트:
    #   - thread A에서는 "앤디"를 기억하고, thread B에서는 모른다고 답하면 thread별 격리에 성공한 것입니다.
    #   - thread_id를 바꿔 기억이 사라지는 것은 버그가 아니라 의도된 격리입니다.
    #     한 사용자의 대화가 다른 창구로 새어 나가면 안 되기 때문입니다.
    #   - 어제 열어 둔 같은 상담방(같은 thread_id)에 오늘 다시 들어오면, 어제의 맥락이 그대로 이어집니다.


if __name__ == "__main__":
    main()
