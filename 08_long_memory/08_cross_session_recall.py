"""08 - 교차 세션 회상: 새 thread는 직전 대화를 모르지만 장기 기억은 떠올린다.

이 예제 하나만으로 다음을 익힙니다.
  - thread A에 쌓은 단기 대화는 thread B(새 세션)에서 보이지 않는다 (단기는 thread별 격리).
  - 그래도 Store에 저장한 장기 기억은 thread를 가로질러 회상된다 (교차 세션 회상).
  - "단기는 thread_id로 대화를 가르고, 장기는 namespace로 지식을 가른다"를 몸으로 확인한다.

이것이 이 챕터의 핵심 점검입니다. 같은 Agent에 단기·장기를 함께 달았을 때,
직전 점심 대화(단기)는 새 세션에서 모르면서 등산 선호(장기)는 떠올린다면,
단기와 장기의 차이를 종합적으로 이해한 것입니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 08_long_memory/08_cross_session_recall.py

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
    """단기+장기를 함께 단 In-graph Agent (07과 동일한 구성)."""
    model = init_chat_model(MODEL)

    def agent_node(state: State, *, store: BaseStore):
        last = state["messages"][-1].content
        recalled = store.search(NS, query=last, limit=3)

        # 회상 값은 방어적으로 읽습니다 (직접 put한 dict와 도구 저장 dict가 섞일 수 있음).
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

    # 1) thread A에서 단기 대화를 한 차례 쌓습니다 (점심 메뉴 → session-A의 단기 메모리에만 남음).
    config_a = {"configurable": {"thread_id": "session-A"}}
    print("=== session-A (기존 세션) ===")
    print("[A 발화] '오늘 점심에 김치찌개를 먹었어'  → session-A 단기 메모리에 저장")
    agent.invoke(
        {"messages": [{"role": "user", "content": "오늘 점심에 김치찌개를 먹었어"}]}, config_a
    )

    # 2) 새 사실 하나를 장기 메모리(Store)에 저장합니다 (thread와 무관하게 영속).
    #    Agent에 장착한 바로 그 store 객체에 직접 저장합니다 (compile 시 넘긴 것과 같은 객체).
    store.put(NS, "fact-hike", {"text": "앤디는 주말마다 등산을 간다"})
    print("[장기 저장] Store(namespace", NS, ")에 '주말 등산' 저장 → thread와 무관하게 영속")

    # 3) thread B는 완전히 새 세션입니다 (단기 메모리는 비었지만 장기 메모리는 공유).
    config_b = {"configurable": {"thread_id": "session-B"}}
    print("\n=== session-B (새 세션, thread_id가 다름) ===")
    print("[B 단기 질문] '내가 점심에 뭐 먹었다고 했지?'  (점심은 session-A의 단기에만 있음)")
    res_b1 = agent.invoke(
        {"messages": [{"role": "user", "content": "내가 점심에 뭐 먹었다고 했지?"}]}, config_b
    )
    # thread_id가 session-A와 달라, A의 단기 대화(점심)는 보이지 않습니다 → 모른다고 답합니다.
    print("[B 단기 응답]", res_b1["messages"][-1].content, "  ← 단기는 thread별 격리라 '모름'")

    # 4) 같은 새 thread에서 장기 기억을 묻습니다.
    print("\n[B 장기 질문] '주말에 내가 보통 뭐 한다고 했지?'  (등산은 장기 Store에 있음)")
    res_b2 = agent.invoke(
        {"messages": [{"role": "user", "content": "주말에 내가 보통 뭐 한다고 했지?"}]}, config_b
    )
    # 장기 메모리(Store)는 thread를 가로질러 공유되므로, 새 세션에서도 '등산'이 회상됩니다.
    print("[B 장기 응답]", res_b2["messages"][-1].content, "  ← 장기는 namespace로 공유되어 '회상'")

    print("\n=== 핵심 대비 ===")
    print("같은 새 세션(B)인데: 단기(점심)는 모르고, 장기(등산)는 떠올립니다.")
    print("→ 단기는 thread_id로 '대화'를 가르고, 장기는 namespace로 '지식'을 가릅니다.")

    # 체크포인트:
    #   - 새 thread(B)에서 직전 점심 대화를 모르면, 단기 메모리가 thread별 격리임을 이해한 것입니다.
    #   - 그런데도 등산을 회상하면, 단기와 장기 메모리의 차이를 종합적으로 이해한 것입니다.

    # 주의: InMemorySaver와 InMemoryStore는 프로세스 메모리에 저장되어 재시작 시 사라집니다.
    #      데모용이며, 운영에서는 PostgresSaver, PostgresStore 등 영속 백엔드로 교체합니다.


if __name__ == "__main__":
    main()
