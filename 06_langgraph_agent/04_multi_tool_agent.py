"""04 - 도구를 여러 개 얹어, 모델이 어떤 도구를 어떤 순서로 부를지 스스로 정하게 한다.

이 예제 하나만으로 다음을 익힙니다.
  - create_agent의 tools 목록에 도구를 둘 이상 넣는다.
  - 도구가 여럿이면 모델이 질문을 보고 "어떤 도구를 어떤 순서로 부를지" 스스로 판단한다.
  - 한 도구의 결과(재고 수량)를 다음 도구(보충 판단)의 입력으로 이어 가는 다단계 흐름을 본다.

03에서 도구 두 개(add, multiply)로 ReAct 루프를 봤다면, 여기서는 업무에 가까운 도구
(재고 조회 → 보충 판단)를 얹어, 도구가 늘어나면 라우팅이 어떻게 일어나는지 관찰합니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 06_langgraph_agent/04_multi_tool_agent.py

키가 없으면 안내만 출력하고 종료합니다.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
# @tool 데코레이터로 함수를 도구로 감쌉니다. args_schema로 인자 스키마를 명시할 수도 있습니다.
from langchain.tools import tool
# BaseModel·Field는 도구 인자의 형태와 설명을 또렷이 잡아 줍니다(LO4에서 익힌 개념).
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


# ---------------------------------------------------------------------------
# 도구 1 — 재고 조회. args_schema로 인자의 의미·예시를 명시해 모델이 정확히 채우게 돕습니다.
# ---------------------------------------------------------------------------

class InventoryInput(BaseModel):
    # Field의 description은 모델에게 "이 인자에 무엇을 넣어야 하는지" 알려 주는 설명입니다.
    sku: str = Field(description="조회할 제품 코드. 예: 'BAT-21700'")
    warehouse: str = Field("ICN", description="창고 코드. 예: 'ICN'(인천), 'PUS'(부산)")


# 데모용 재고 표 (실무라면 데이터베이스·API 조회 자리입니다).
_STOCK = {("BAT-21700", "ICN"): 1240, ("BAT-21700", "PUS"): 380}


@tool("check_inventory", args_schema=InventoryInput)
def check_inventory(sku: str, warehouse: str = "ICN") -> str:
    """지정한 창고의 제품 재고 수량을 조회한다."""
    qty = _STOCK.get((sku, warehouse))
    if qty is None:
        # 빈 결과 대신 '읽을 수 있는 메시지'를 돌려줘야 모델이 없음을 이해하고 멈춥니다(06에서 자세히 다룹니다).
        return f"재고 정보 없음: sku={sku}, warehouse={warehouse}"
    return f"{warehouse} 창고의 {sku} 재고는 {qty}개입니다."


# ---------------------------------------------------------------------------
# 도구 2 — 보충 판단. 도구 1이 알려 준 수량을 입력으로 받아 임계치와 비교합니다.
# ---------------------------------------------------------------------------

@tool
def restock_threshold(qty: int, threshold: int = 500) -> str:
    """재고 수량이 임계치보다 적은지 판단해 보충 필요 여부를 알려준다."""
    return "보충 필요" if qty < threshold else "충분"


def build_agent():
    """도구 두 개(재고 조회 + 보충 판단)를 얹은 업무 Agent를 만든다.

    도구 목록만 늘리면 Agent의 능력이 확장됩니다. 그래프 배선은 03과 똑같이 한 줄입니다.
    system_prompt로 "수량은 추측하지 말고 도구로 확인하라"고 규칙을 정해, 모델이 도구를 쓰도록 유도합니다.
    """
    agent = create_agent(
        MODEL,
        tools=[check_inventory, restock_threshold],  # 도구 목록만 늘리면 능력이 확장됩니다
        system_prompt=(
            "너는 사내 재고를 관리하는 물류 비서다. "
            "재고 수량은 추측하지 말고 반드시 check_inventory 도구로 확인하라. "
            "보충이 필요한지는 restock_threshold 도구로 판단하라. "
            "마지막에는 재고 수량과 보충 여부를 한 문장으로 정리하라."
        ),
    )
    return agent


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    agent = build_agent()

    # 부산(PUS) 창고 재고는 380개로 임계치 500보다 적습니다 → 보충 필요로 결론나야 합니다.
    print("=== 도구 두 개를 순서대로 부르는 질문 (재고 조회 → 보충 판단) ===")
    res = agent.invoke(
        {"messages": [{"role": "user",
                       "content": "BAT-21700 부산(PUS) 창고 재고가 보충이 필요한지 확인해줘"}]}
    )
    print("최종 답변:", res["messages"][-1].content)

    # 누적 메시지를 보면 모델이 도구를 어떤 순서로 골라 불렀는지가 그대로 드러납니다.
    print("\n[누적된 메시지 흐름 — 모델이 고른 도구 호출 순서]")
    for m in res["messages"]:
        m.pretty_print() if hasattr(m, "pretty_print") else print(m)

    # 체크포인트:
    #   - check_inventory(PUS=380) → restock_threshold(380) 순으로 두 도구가 차례로 호출되면,
    #     모델이 도구를 스스로 골라 순서대로 엮은 것입니다.
    #   - 최종 답에 "380"과 "보충 필요"가 함께 나오면, 한 도구 결과가 다음 도구로 이어진 것입니다.
    #   - 도구를 여러 개 줬어도 그래프 배선은 03과 같은 한 줄이라는 점을 떠올리십시오.


if __name__ == "__main__":
    main()
