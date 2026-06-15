"""03 - 여러 Custom Tool을 한 모델에 붙여 질문 의도로 라우팅한다.

이 예제 하나만으로 다음을 익힙니다.
  - 서로 다른 책임을 가진 도구 두 개를 정의한다 (단일 책임 원칙).
  - 두 도구를 한 모델에 함께 붙이면, 모델이 질문을 보고 알맞은 도구를 고른다.
  - tool_calls → 도구 실행 → ToolMessage 회신의 왕복(수동 도구 루프)을 작은 함수로 묶는다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 04_custom_tool/03_multiple_tools.py

키가 없으면 안내만 출력하고 종료합니다 (도구 정의 점검은 키 없이도 됩니다).
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# v1 권장 경로입니다. langchain_core.messages에서 가져와도 동일하게 동작합니다.
from langchain.messages import HumanMessage, ToolMessage
# @tool과 ToolException은 langchain_core.tools에 있습니다 (도구 정의의 표준 경로).
from langchain_core.tools import tool, ToolException

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


# ===========================================================================
# 도구 두 개 — 책임이 서로 다릅니다 (단일 책임). 모델은 질문을 보고 둘 중 하나를 고릅니다.
# ===========================================================================

# 사내 재고를 흉내 낸 데모 데이터입니다 (실제로는 DB·API를 호출하는 자리입니다).
_STOCK = {("BAT-21700", "ICN"): 1240, ("BAT-21700", "PUS"): 380}


@tool("check_inventory")
def check_inventory(sku: str, warehouse: str = "ICN") -> str:
    """지정한 창고의 제품 재고 수량을 조회한다.
    제품 코드(sku)와 창고 코드(warehouse)로 현재 보유 수량을 반환한다.
    재고 수량을 알아야 할 때 사용한다. 예: 'BAT-21700 인천 창고 재고'."""
    qty = _STOCK.get((sku.upper(), warehouse.upper()))
    if qty is None:
        raise ToolException(f"재고 정보 없음: sku={sku}, warehouse={warehouse}")
    return f"{warehouse.upper()} 창고의 {sku.upper()} 재고는 {qty:,}개입니다."


# 데모용 환율표입니다. 단위 환산 도구를 하나 더 두어 "여러 도구 중 라우팅"을 보여 줍니다.
_FX = {"USD": 1380.0, "EUR": 1480.0, "JPY": 9.1}


@tool("convert_currency")
def convert_currency(amount: float, currency: str) -> str:
    """외화 금액을 원화(KRW)로 환산한다.
    금액(amount)과 통화 코드(currency: USD, EUR, JPY)를 받아 원화 환산액을 반환한다.
    금액을 원화로 바꿔야 할 때 사용한다. 예: '100달러 원화로'."""
    rate = _FX.get(currency.upper())
    if rate is None:
        raise ToolException(f"지원하지 않는 통화: {currency} (지원: USD, EUR, JPY)")
    return f"{amount:,.0f} {currency.upper()} = {amount * rate:,.0f} KRW"


# ===========================================================================
# 수동 도구 루프 — tool_calls가 빌 때까지 도구를 실행하고 결과를 되돌린다.
# (03 챕터에서 익힌 왕복을 여기서는 작은 함수로 묶어 재사용합니다.)
# ===========================================================================

def run_tool_loop(model_with_tools, messages: list, tool_map: dict) -> str:
    """tool_calls가 빌 때까지 도구를 실행하고 결과를 되돌린 뒤 최종 답변을 반환한다."""
    ai = model_with_tools.invoke(messages)
    messages.append(ai)
    while ai.tool_calls:                         # 부를 도구가 남아 있는 동안 반복
        for call in ai.tool_calls:
            chosen = tool_map[call["name"]]      # 요청한 이름의 도구 선택
            try:
                result = chosen.invoke(call["args"])
            except ToolException as e:
                result = str(e)                  # 실패도 결과로 전달해 모델이 회복하게 함
            # tool_call_id를 요청 id와 똑같이 맞춰야 모델이 결과를 짝지을 수 있습니다.
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        ai = model_with_tools.invoke(messages)   # 결과가 담긴 메시지로 다시 호출
        messages.append(ai)
    return ai.content


def route_among_tools(model) -> None:
    """두 도구를 한 모델에 붙이면, 질문 의도에 따라 알맞은 도구가 호출된다."""
    tools = [check_inventory, convert_currency]
    tool_map = {t.name: t for t in tools}        # 이름으로 도구를 찾기 위한 사전
    model_with_tools = model.bind_tools(tools)

    # (1) 재고 질문 → check_inventory로 라우팅되어야 합니다.
    재고답 = run_tool_loop(
        model_with_tools,
        [HumanMessage("BAT-21700 인천 창고 재고 알려줘")],
        tool_map,
    )
    print("[재고]", 재고답)                       # 예: ICN 창고의 BAT-21700 재고는 1,240개입니다.

    # (2) 환산 질문 → convert_currency로 라우팅되어야 합니다.
    환산답 = run_tool_loop(
        model_with_tools,
        [HumanMessage("100 달러는 원화로 얼마야?")],
        tool_map,
    )
    print("[환산]", 환산답)                       # 예: 100 USD = 138,000 KRW

    # 체크포인트: 질문 종류에 따라 서로 다른 도구가 호출되면 다중 도구 라우팅이 정상입니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    model = init_chat_model(MODEL)  # 강의 직전 최신 모델과 가격을 재확인하십시오.

    print("=== 여러 도구 라우팅 ===")
    route_among_tools(model)


if __name__ == "__main__":
    main()
