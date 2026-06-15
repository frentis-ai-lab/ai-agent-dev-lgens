"""07 - 단기(checkpointer) + 장기(Store)를 한 Agent에 함께 장착한다.

이 예제 하나만으로 다음을 익힙니다.
  - 한 Agent에 단기 메모리(InMemorySaver)와 장기 메모리(InMemoryStore)를 함께 단다.
  - 같은 thread에서는 단기 메모리로 직전 대화를 회상한다 (thread_id로 대화 잇기).
  - 두 메모리가 같은 그래프 안에서 어떻게 협력하는지 본다.

단기와 장기는 저장 단위·회상 방식이 다릅니다.
  - 단기(checkpointer): 한 대화의 상태를 통째로 저장했다 thread_id로 통째로 복원
  - 장기(Store):        골라낸 사실 하나를 저장했다 namespace에서 검색으로 회상
이 예제는 둘을 함께 단 Agent가 같은 thread 안에서 직전 발화를 잇는 것까지 확인합니다.
(새 thread에서도 장기 기억이 떠오르는 교차 세션 회상은 08에서 이어집니다.)

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 08_long_memory/07_short_plus_long.py

이 예제는 모델·임베딩 호출을 사용하므로 OPENAI_API_KEY가 필요합니다.
키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os  # 환경변수(OPENAI_API_KEY) 확인에 씁니다.
# Annotated[타입, 부가정보]는 타입에 동작(여기서는 메시지 누적)을 덧붙이는 표기입니다.
from typing import Annotated

# TypedDict는 "어떤 키와 타입을 갖는 dict"인지 미리 선언하는 틀입니다 (그래프 State 정의용).
from typing_extensions import TypedDict

from dotenv import load_dotenv  # .env 파일을 환경변수로 올려 줍니다.
from langchain.chat_models import init_chat_model  # "벤더:모델명" → 챗 모델 객체
from langchain.embeddings import init_embeddings  # "벤더:모델명" → 임베딩 모델 객체
from langchain.messages import SystemMessage  # 모델에게 역할·규칙을 주는 메시지
from langgraph.checkpoint.memory import InMemorySaver  # 단기 메모리(thread별 상태 저장)
from langgraph.graph import StateGraph, START, END  # 그래프 뼈대와 시작·끝 표시
from langgraph.graph.message import add_messages  # 메시지를 덮어쓰지 않고 누적하는 함수
from langgraph.store.base import BaseStore  # 노드에 주입되는 Store의 공통 타입(타입 힌트용)
from langgraph.store.memory import InMemoryStore  # 장기 메모리 저장소

load_dotenv()

MODEL = "openai:gpt-5.4-mini"
EMBED = "openai:text-embedding-3-small"
NS = ("user-123", "memories")


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_agent(store: InMemoryStore):
    """단기+장기를 함께 단 In-graph Agent를 구성한다.

    종합 실습에서는 회상 시점을 코드로 분명히 제어하는 In-graph 방식이 검증에 유리합니다.
    노드가 직접 store.search로 회상하고, 그 결과를 시스템 프롬프트에 끼워 답합니다.
    """
    model = init_chat_model(MODEL)

    def agent_node(state: State, *, store: BaseStore):
        last = state["messages"][-1].content
        recalled = store.search(NS, query=last, limit=3)

        # 회상 값은 방어적으로 읽습니다 (직접 put한 dict와 도구 저장 dict가 섞일 수 있음).
        # r.value['text'] 직접 접근은 KeyError 위험이 있어 쓰지 않습니다.
        def _text(value):
            if isinstance(value, dict):
                return value.get("text") or value.get("content") or str(value)
            return str(value)

        memory_text = "\n".join(f"- {_text(r.value)}" for r in recalled) or "(없음)"
        sys = SystemMessage(
            "너는 사용자를 기억하는 비서다. 아래는 회상한 사실이며, "
            "질문과 어긋나면 무시하라.\n" + memory_text
        )
        reply = model.invoke([sys] + state["messages"])
        return {"messages": [reply]}

    b = StateGraph(State)
    b.add_node("agent", agent_node)
    b.add_edge(START, "agent")
    b.add_edge("agent", END)

    # 단기(checkpointer) + 장기(store)를 함께 장착합니다.
    return b.compile(checkpointer=InMemorySaver(), store=store)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return

    store = InMemoryStore(
        index={"dims": 1536, "embed": init_embeddings(EMBED), "fields": ["text"]}
    )
    agent = build_agent(store)

    # 같은 thread(session-A)에서 직전 대화를 단기 메모리로 잇습니다.
    # thread_id가 같으면 checkpointer가 직전 상태를 복원해 대화가 이어집니다.
    config_a = {"configurable": {"thread_id": "session-A"}}
    print("[1번째 발화 · 같은 thread] '오늘 점심에 김치찌개를 먹었어'")
    agent.invoke(
        {"messages": [{"role": "user", "content": "오늘 점심에 김치찌개를 먹었어"}]}, config_a
    )
    print("[2번째 발화 · 같은 thread] '내가 점심에 뭐 먹었다고 했지?'")
    res_a = agent.invoke(
        {"messages": [{"role": "user", "content": "내가 점심에 뭐 먹었다고 했지?"}]}, config_a
    )
    # 같은 thread_id라 직전 발화가 단기 메모리에 남아 '김치찌개'를 회상합니다.
    print("[A 단기 회상]", res_a["messages"][-1].content)

    # 체크포인트:
    #   - 같은 thread에서 직전 발화('김치찌개')를 회상하면, 단기 메모리(checkpointer)를 이해한 것입니다.
    #   - 단기와 장기가 한 그래프에 함께 달려 있고, 노드가 그 둘을 모두 쓸 수 있음을 확인한 것입니다.


if __name__ == "__main__":
    main()
