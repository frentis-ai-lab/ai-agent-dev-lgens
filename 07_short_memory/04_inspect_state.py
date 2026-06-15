"""04 - 저장된 대화 상태를 직접 들여다본다 (get_state · get_state_history).

이 예제 하나만으로 다음을 익힙니다.
  - get_state로 특정 thread_id에 지금 무엇이 쌓여 있는지 스냅샷을 본다.
  - get_state_history로 턴마다 찍힌 체크포인트(상태 스냅샷)를 거슬러 본다.
  - thread별로 상태가 따로 저장된다는 사실을 눈으로 확인한다.

checkpointer는 "보이지 않는 마법"이 아닙니다. 저장소 안을 직접 조회해 디버깅·되감기에 쓸 수 있습니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 07_short_memory/04_inspect_state.py

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


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    agent = create_agent(
        MODEL,
        tools=[add],
        system_prompt="너는 친절한 한국어 비서다.",
        checkpointer=InMemorySaver(),
    )

    # 1) thread user-123에 두 턴을 쌓습니다 (조회 대상이 될 대화).
    config = {"configurable": {"thread_id": "user-123"}}
    agent.invoke({"messages": [{"role": "user", "content": "내 이름은 앤디야. 기억해 줘."}]}, config)
    agent.invoke({"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]}, config)

    # 2) get_state: 해당 thread_id에 "지금 저장된 상태 스냅샷"을 돌려줍니다(모델을 부르지 않는 조회 함수).
    #    .values는 상태 딕셔너리이고, 그 안의 "messages" 키에 누적된 메시지 리스트가 들어 있습니다.
    state = agent.get_state(config)
    # f"{...}"는 f-문자열입니다. 중괄호 안에 파이썬 식을 넣어 값을 끼워 넣습니다(여기서는 중첩 딕셔너리 접근).
    print(f"[{config['configurable']['thread_id']}의 누적 메시지] 총 {len(state.values['messages'])}개")
    # for ... in 리스트: 리스트의 원소를 하나씩 m에 담아 반복합니다.
    for m in state.values["messages"]:
        # 메시지 종류(Human/AI/Tool)와 내용 앞부분만 잘라 출력합니다.
        # content가 문자열이 아닐 수도 있어(도구 호출 등) str()로 감싸 안전하게 처리합니다.
        text = m.content if isinstance(m.content, str) else str(m.content)
        # type(m).__name__은 객체의 클래스 이름, text[:40]은 앞 40글자, repr(...)은 따옴표를 붙여 보기 좋게 출력합니다.
        print("  ", type(m).__name__, repr(text[:40]))

    # 3) get_state_history: "턴마다 찍힌 상태 스냅샷"을 거슬러 봅니다 (되감기·디버깅에 유용).
    #    제너레이터(값을 하나씩 흘려보내는 객체)를 돌려주므로 list(...)로 감싸 한 번에 모아 개수를 셉니다.
    #    턴이 늘수록 스냅샷도 늘어납니다.
    history = list(agent.get_state_history(config))
    print("[user-123의 체크포인트 수]", len(history), "(턴이 진행될 때마다 스냅샷이 쌓임)")

    # 4) 다른 thread는 거의 비어 있음을 대비로 확인합니다 (thread별로 따로 저장되므로).
    empty_config = {"configurable": {"thread_id": "user-999"}}
    empty_state = agent.get_state(empty_config)
    # .values는 아직 대화가 없으면 비어 있거나 messages 키가 없을 수 있습니다.
    # 딕셔너리.get("키", 기본값)은 키가 없어도 오류 없이 기본값(여기서는 빈 리스트)을 돌려줍니다.
    empty_messages = empty_state.values.get("messages", [])
    print("[user-999의 누적 메시지 수]", len(empty_messages), "(대화한 적 없는 thread는 0)")

    # 두 thread를 나란히 대비해 격리를 한눈에 보입니다.
    print("\n[비교] user-123은 메시지가 쌓여 있고 user-999는 비어 있습니다 → thread별로 따로 저장됩니다.")

    # 체크포인트:
    #   - user-123에는 Human·AI 메시지가 쌓여 있고, user-999는 비어 있으면 thread별 격리를 눈으로 확인한 것입니다.
    #   - 체크포인트 수가 1보다 크면, 매 턴마다 상태가 기록되고 있다는 뜻입니다.


if __name__ == "__main__":
    main()
