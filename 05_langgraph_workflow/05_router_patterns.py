"""05 - 라우터 설계 세 유형: 분류 라우터·조건 분기 라우터·동적 데이터 흐름.

이 예제 하나만으로 다음을 익힙니다.
  - 분류 라우터: 모델이 입력을 분류해, 종류에 맞는 전문 노드로 보낸다.
  - 조건 분기 라우터: 상태에 담긴 수치·플래그(임계치)로 경로를 정한다.
  - 동적 데이터 흐름: 목표를 만족할 때까지 반복 횟수가 입력에 따라 달라진다.
  - 설계 요령: 분류(노드)와 라우팅(엣지)을 분리하고, 라우터 반환 타입을 좁혀 둔다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 05_langgraph_workflow/05_router_patterns.py

분류 라우터는 모델을 부르므로 OPENAI_API_KEY가 필요합니다.
조건 분기 라우터·동적 데이터 흐름은 모델 없이 동작하므로 키가 없어도 그대로 돕니다.
"""

import operator
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


# ---------------------------------------------------------------------------
# 유형 1) 분류 라우터 — 모델 판단으로 종류를 나눠 전문 노드로 보낸다 (모델 필요)
#   언제 쓰는가: 들어오는 요청의 '종류'가 여러 가지이고, 종류마다 처리 방식이 다를 때.
#   예) 고객 문의를 '환불/기술/일반'으로 나눠 각기 다른 응대 노드로 보내는 라우팅.
# ---------------------------------------------------------------------------

def run_classify_router(model) -> None:
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        category: str  # 분류 노드가 채울 결과

    def classify(state: State) -> dict:
        # 입력을 정해진 카테고리 중 하나로 분류해 상태(category)에 적어 둡니다.
        last = state["messages"][-1].content
        resp = model.invoke(
            "다음 문의를 정확히 한 단어로 분류해줘 (refund/tech/general 중 하나만): " + last
        )
        label = resp.content.strip().lower()
        # 모델이 예상 밖 단어를 내도 안전하도록 모르는 값은 general로 수렴시킵니다 (코드 안전망).
        category = label if label in {"refund", "tech", "general"} else "general"
        return {"category": category}

    def handle_refund(state: State) -> dict:
        return {"messages": [AIMessage("[환불팀] 환불 절차를 안내드리겠습니다.")]}

    def handle_tech(state: State) -> dict:
        return {"messages": [AIMessage("[기술지원팀] 증상을 확인하겠습니다.")]}

    def handle_general(state: State) -> dict:
        return {"messages": [AIMessage("[일반상담] 무엇을 도와드릴까요.")]}

    def route(state: State) -> str:
        # 라우터는 상태에 저장된 분류 결과(category)를 그대로 다음 노드 키로 사용합니다.
        # 분류(노드)와 라우팅(엣지)을 분리하면, 분류 로직과 흐름 제어를 따로 다듬을 수 있습니다.
        return state["category"]

    b = StateGraph(State)
    b.add_node("classify", classify)
    b.add_node("refund", handle_refund)
    b.add_node("tech", handle_tech)
    b.add_node("general", handle_general)
    b.add_edge(START, "classify")
    # 분류 결과 문자열을 각 전문 노드에 매핑합니다.
    b.add_conditional_edges(
        "classify", route,
        {"refund": "refund", "tech": "tech", "general": "general"},
    )
    b.add_edge("refund", END)
    b.add_edge("tech", END)
    b.add_edge("general", END)
    graph = b.compile()

    # 세 문의를 차례로 넣어, 모델이 매긴 category에 따라 서로 다른 전문 노드로 갈라지는지 봅니다.
    for q in ["환불받고 싶어요", "앱이 자꾸 꺼져요", "영업시간이 언제인가요"]:
        result = graph.invoke({"messages": [HumanMessage(q)], "category": ""})
        # category = 분류 노드가 채운 키 = 다음에 거친 전문 노드의 이름
        print(f"  [{q}]")
        print(f"     classify가 매긴 category='{result['category']}' → '{result['category']}' 노드로 분기")
        print(f"     응답: {result['messages'][-1].content}")

    # 체크포인트: 문의 내용에 따라 환불/기술/일반 노드로 각각 갈라지면 분류 라우터가 동작한 것입니다.


# ---------------------------------------------------------------------------
# 유형 2) 조건 분기 라우터 — 상태 값(임계치)으로 경로를 정한다 (모델 불필요)
#   언제 쓰는가: 모델 판단이 아니라 '상태에 담긴 수치·플래그'로 흐름이 정해질 때.
#   예) 결제 금액이 임계치를 넘으면 사람 검토로, 아니면 자동 승인으로 보내는 결정 분기.
# ---------------------------------------------------------------------------

def run_value_router() -> None:
    class State(TypedDict):
        amount: int    # 입력: 결제 금액
        decision: str  # 산출물: 처리 결정

    def evaluate(state: State) -> dict:
        # 모델 호출 없이 상태 값만으로 판단하는 통과 노드입니다 (규칙 기반 분기는 LLM이 필요 없습니다).
        return {}

    # 반환 타입을 Literal로 좁히면, 가능한 경로가 코드에 분명히 드러나 실수가 줄어듭니다.
    def route(state: State) -> Literal["auto_approve", "manual_review"]:
        # 금액이 임계치(100,000원)를 넘으면 사람 검토로, 아니면 자동 승인으로 보냅니다.
        return "manual_review" if state["amount"] >= 100_000 else "auto_approve"

    def auto_approve(state: State) -> dict:
        # f-string의 {값:,}은 1,000 단위마다 쉼표를 찍는 표기입니다.
        return {"decision": f"{state['amount']:,}원 자동 승인"}

    def manual_review(state: State) -> dict:
        return {"decision": f"{state['amount']:,}원 고액 결제 — 사람 검토 대기"}

    b = StateGraph(State)
    b.add_node("evaluate", evaluate)
    b.add_node("auto_approve", auto_approve)
    b.add_node("manual_review", manual_review)
    b.add_edge(START, "evaluate")
    b.add_conditional_edges(
        "evaluate", route,
        {"auto_approve": "auto_approve", "manual_review": "manual_review"},
    )
    b.add_edge("auto_approve", END)
    b.add_edge("manual_review", END)
    graph = b.compile()

    # 임계치(100,000원)를 기준으로 작은 금액과 큰 금액을 각각 넣어, 경로가 갈리는지 봅니다.
    for amount in [30_000, 250_000]:
        result = graph.invoke({"amount": amount, "decision": ""})
        # 임계치와 비교해 어느 노드로 갔는지 함께 보여 줍니다.
        branch = "manual_review" if amount >= 100_000 else "auto_approve"
        print(f"  [{amount:,}원] 임계치 100,000원과 비교 → '{branch}' 노드 → {result['decision']}")

    # 체크포인트: 임계치 미만은 자동 승인, 이상은 사람 검토로 갈리면 조건 분기 라우터가 동작한 것입니다.
    # 분류 라우터는 모델 판단으로, 이 조건 분기 라우터는 상태 수치로 경로를 정한다는 차이가 핵심입니다.


# ---------------------------------------------------------------------------
# 유형 3) 동적 데이터 흐름 — 목표를 만족할 때까지 반복 횟수가 달라진다 (모델 불필요)
#   언제 쓰는가: "목표를 만족할 때까지" 같은 가변 반복이 필요할 때.
#   예) 초안을 점수화하고, 기준 미달이면 다시 고쳐 쓰기를 반복하는 자기수정 루프.
#   (개념 전달이 목적이라 모델 호출 없이 점수를 모사합니다 — 흐름 구조에 집중하십시오.)
# ---------------------------------------------------------------------------

def run_dynamic_flow() -> None:
    class State(TypedDict):
        score: int                              # 현재 품질 점수 (0~100 가정)
        attempts: Annotated[int, operator.add]  # 시도 횟수 누적 (리듀서로 호출마다 +1)

    def improve(state: State) -> dict:
        # 매 반복마다 점수를 끌어올리고 시도 횟수를 1 늘립니다.
        # 실무라면 여기서 모델로 초안을 고쳐 쓰고 다시 채점합니다.
        new_score = min(state["score"] + 30, 100)
        return {"score": new_score, "attempts": 1}

    def route(state: State) -> Literal["improve", "done"]:
        # 목표(80점)에 도달하면 종료, 아니면 다시 improve로 — 반복 횟수가 입력에 따라 달라집니다.
        return "done" if state["score"] >= 80 else "improve"

    def done(state: State) -> dict:
        # 종료 노드: 최종 결과만 정리합니다 (상태 변경 없음).
        return {}

    b = StateGraph(State)
    b.add_node("improve", improve)
    b.add_node("done", done)
    b.add_edge(START, "improve")
    # improve를 한 번 돈 뒤, 점수에 따라 improve로 되돌아가거나 done으로 빠집니다 (가변 반복).
    b.add_conditional_edges("improve", route, {"improve": "improve", "done": "done"})
    b.add_edge("done", END)
    graph = b.compile()

    # 시작 점수가 낮을수록 목표 도달까지 더 많이 반복합니다 (경로가 데이터에 따라 동적으로 변함).
    for start in [70, 10]:
        result = graph.invoke({"score": start, "attempts": 0})
        # improve 노드는 매번 +30점, route는 80점 미만이면 다시 improve로 되돌립니다.
        print(f"  [시작 {start}점] improve 노드가 +30점씩 반복 (목표 80점)")
        print(f"     → 최종 {result['score']}점, improve를 {result['attempts']}회 거침")

    # 체크포인트: 시작 점수에 따라 반복 횟수가 달라지면 동적 데이터 흐름을 확인한 것입니다.
    # 변형 포인트: 목표 점수(80)나 증가폭(30)을 바꾸면 반복 횟수가 즉시 달라집니다.


def main() -> None:
    # 조건 분기 라우터·동적 데이터 흐름은 모델이 필요 없으므로 먼저 보여 줍니다.
    # 분류 라우터만 모델이 필요해, 키가 있을 때만 실행합니다.
    if os.getenv("OPENAI_API_KEY"):
        model = init_chat_model(MODEL)
        print("=== 유형 1) 분류 라우터 (모델 판단) ===")
        run_classify_router(model)
    else:
        print("=== 유형 1) 분류 라우터 (모델 판단) ===")
        print("OPENAI_API_KEY가 없어 분류 라우터는 건너뜁니다 (나머지 두 유형은 키 없이 실행됩니다).")

    print("\n=== 유형 2) 조건 분기 라우터 (상태 수치) ===")
    run_value_router()

    print("\n=== 유형 3) 동적 데이터 흐름 (가변 반복) ===")
    run_dynamic_flow()


if __name__ == "__main__":
    main()
