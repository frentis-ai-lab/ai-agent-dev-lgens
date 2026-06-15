"""06 - 자르는 대신 요약으로 압축하고, 재시작해도 남는 영속 저장으로 전환한다.

이 예제 하나만으로 다음을 익힙니다.
  - SummarizationMiddleware로 긴 대화의 앞부분을 자동 요약해 토큰을 통제한다 (압축 보존).
  - InMemorySaver(RAM, 휘발) 대신 SqliteSaver(파일, 영속)로 갈아 끼운다.
  - 영속 saver로 바꿔도 Agent 코드는 그대로 두고 checkpointer 객체만 교체하면 됨을 확인한다.

자르기(trim, 05 예제)는 오래된 메시지를 "버립니다". 요약은 오래된 메시지를 "압축해서 보존"합니다.
SqliteSaver는 별도 패키지라, 없으면 그 부분만 자동으로 건너뜁니다 (실습이 멈추지 않습니다).

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 07_short_memory/06_summarize_persist.py

키가 없으면 안내만 출력하고 종료합니다.
영속 저장을 직접 해 보려면: uv add langgraph-checkpoint-sqlite
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
# SummarizationMiddleware는 대화가 길어지면 앞부분을 자동으로 요약문 한 덩어리로 대체하는 v1 내장 미들웨어입니다.
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


@tool
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


# ===========================================================================
# 1부. 요약 미들웨어로 긴 대화를 압축한다
# ===========================================================================

def run_summarization() -> None:
    """요약 미들웨어를 붙여, 여러 턴을 주고받아도 누적 메시지가 폭증하지 않게 한다."""
    # SummarizationMiddleware는 대화가 길어지면 자동으로 앞부분을 요약문으로 대체합니다.
    summarizer = SummarizationMiddleware(
        model=MODEL,                # 요약을 수행할 모델 (본 대화 모델과 같아도 됩니다)
        trigger=("messages", 6),    # 메시지가 6개를 넘으면 요약을 발동합니다
        keep=("messages", 4),       # 최근 4개는 원문 그대로 남기고, 그 앞을 요약합니다
    )

    # checkpointer로 누적된 대화를 요약 미들웨어가 관리하도록 함께 붙입니다.
    agent = create_agent(
        MODEL,
        tools=[add],
        system_prompt="너는 친절한 한국어 비서다.",
        checkpointer=InMemorySaver(),
        middleware=[summarizer],    # 요약 미들웨어를 에이전트 파이프라인에 끼워 넣습니다
    )
    print("[에이전트]", type(agent).__name__, "(요약 미들웨어 부착 완료)")

    config = {"configurable": {"thread_id": "summary-1"}}

    # 여러 턴을 주고받아 메시지를 쌓습니다 (trigger 임계치를 넘기기 위해).
    topics = ["내 이름은 앤디야.", "나는 서울에 살아.", "취미는 등산이야.", "방금 말한 내 정보를 요약해 줘."]
    out = None
    for t in topics:
        out = agent.invoke({"messages": [{"role": "user", "content": t}]}, config)
    print("[요약 활용 답변]", out["messages"][-1].content)  # 앞 대화를 기억해 요약하면 정상

    # 압축이 일어났는지 상태로 확인합니다. 요약 후에는 누적 메시지 수가 무한정 늘지 않습니다.
    state = agent.get_state(config)
    print("[요약 후 누적 메시지 수]", len(state.values["messages"]))

    # 체크포인트: 4턴을 주고받아도 누적 메시지가 폭증하지 않고 요약 답변이 나오면 압축이 동작하는 것입니다.


# ===========================================================================
# 2부. 재시작해도 대화가 남는 영속 저장으로 전환한다 (운영 전환)
# ===========================================================================

def run_persistent() -> None:
    """InMemorySaver(RAM, 휘발) 자리에 SqliteSaver(파일, 영속)만 끼워 넣는다."""
    # SqliteSaver는 별도 패키지입니다. 설치: uv add langgraph-checkpoint-sqlite
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        # 패키지가 없어도 실습이 멈추지 않도록 안내만 하고 건너뜁니다.
        print("[skip] langgraph-checkpoint-sqlite가 없어 영속 예제를 건너뜁니다.")
        print("       설치: uv add langgraph-checkpoint-sqlite")
        return

    db_path = "short_memory.sqlite"  # 이 파일에 대화 상태가 영속 저장됩니다

    # SqliteSaver.from_conn_string은 컨텍스트 매니저로 DB 연결을 열고 닫습니다.
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        # InMemorySaver 자리에 SqliteSaver만 끼워 넣으면 끝입니다 (나머지 Agent 코드는 그대로).
        agent = create_agent(
            MODEL,
            tools=[add],
            system_prompt="너는 친절한 한국어 비서다.",
            checkpointer=checkpointer,  # 이 인자만 영속용으로 교체합니다. 호출 코드는 손대지 않습니다.
        )

        config = {"configurable": {"thread_id": "persist-1"}}

        # 같은 db_path·thread_id로 이 스크립트를 두 번째 실행하면,
        # 아래 질문에 첫 실행 때 알려 준 이름을 기억하고 답합니다 (파일에 남아 있으므로).
        r = agent.invoke(
            {"messages": [{"role": "user", "content": "내 이름은 앤디야. 다음에도 기억해."}]},
            config,
        )
        print("[저장]", r["messages"][-1].content)

        r2 = agent.invoke(
            {"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]},
            config,
        )
        print("[복원]", r2["messages"][-1].content)

        print(f"[영속] 대화가 '{db_path}' 파일에 저장되었습니다. 재실행해도 같은 thread_id면 이어집니다.")

    # 체크포인트:
    #   - 코드를 한 번 더 실행했을 때 [복원]에서 이름을 기억하면 영속 저장에 성공한 것입니다.
    #   - 영속 파일을 비우려면 short_memory.sqlite 파일을 삭제하면 됩니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    print("=== 1부. 요약 미들웨어로 압축 ===")
    run_summarization()

    print("\n=== 2부. SqliteSaver 영속 저장 ===")
    run_persistent()


if __name__ == "__main__":
    main()
