"""05 - ToolException으로 실패를 모델에 알려 회복(재질문)시킨다.

이 예제 하나만으로 다음을 익힙니다.
  - 도구가 실패할 때 예외로 루프를 죽이지 않고 ToolException으로 사유를 던진다.
  - 그 사유를 ToolMessage로 모델에 되돌려 '왜 실패했는지'를 알린다.
  - 모델이 값을 지어내지 않고 사용자에게 되묻거나 인자를 고쳐 재시도하게 한다.

실패도 하나의 '관찰 결과'입니다. 죽이지 말고 모델에게 돌려주면 Agent가 안정적이 됩니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 04_custom_tool/05_tool_exception_recovery.py

키가 없으면 안내만 출력하고 종료합니다 (도구의 실패 동작은 키 없이도 확인합니다).
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, ToolException

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


# ===========================================================================
# 실패할 수 있는 도구 — 없는 데이터는 ToolException으로 사유를 돌려준다.
# ===========================================================================

# 인천(ICN)·부산(PUS)만 데이터가 있습니다. 광주(GWJ)는 없으므로 ToolException이 납니다.
_STOCK = {("BAT-21700", "ICN"): 1240, ("BAT-21700", "PUS"): 380}


@tool("check_inventory")
def check_inventory(sku: str, warehouse: str = "ICN") -> str:
    """지정한 창고의 제품 재고 수량을 조회한다.
    재고 수량을 알아야 할 때 사용한다. 예: 'BAT-21700 인천 창고 재고'."""
    qty = _STOCK.get((sku.upper(), warehouse.upper()))
    if qty is None:
        # 예외를 그냥 죽이지 않고 ToolException으로 던지면, 그 메시지를 모델에 되돌릴 수 있습니다.
        raise ToolException(f"재고 정보 없음: sku={sku}, warehouse={warehouse}")
    return f"{warehouse.upper()} 창고의 {sku.upper()} 재고는 {qty:,}개입니다."


def show_tool_failure_without_llm() -> None:
    """LLM 없이도 도구의 실패 동작을 확인한다 (없는 창고 → ToolException)."""
    print("[정상] ", check_inventory.invoke({"sku": "BAT-21700", "warehouse": "ICN"}))
    try:
        check_inventory.invoke({"sku": "BAT-21700", "warehouse": "GWJ"})  # 광주 데이터 없음
    except ToolException as e:
        print("[실패] ", e)  # 예: 재고 정보 없음: sku=BAT-21700, warehouse=GWJ
    # 체크포인트: 없는 창고가 ToolException으로 사유를 돌려주면, 실패를 관찰 결과로 다룰 준비가 된 것입니다.


# ===========================================================================
# LLM 경로 — 실패 사유를 모델에 회신해 회복(재질문)시킨다.
# ===========================================================================

def recovery_loop(model) -> None:
    """도구 실패 사유를 ToolMessage로 돌려주면, 모델이 지어내지 않고 재확인을 요청한다."""
    system_prompt = SystemMessage(
        "너는 사내 재고를 조회하는 물류 비서다. "
        "재고 수량은 추측하지 말고 반드시 check_inventory 도구로 확인하라. "
        "도구가 '재고 정보 없음' 등으로 실패하면, 지어내지 말고 "
        "사용자에게 제품 코드와 창고를 다시 확인해 달라고 한국어 한 문장으로 요청하라."
    )
    model_with_tools = model.bind_tools([check_inventory])

    # 존재하지 않는 조합(광주 창고)으로 일부러 ToolException을 유발합니다.
    messages = [system_prompt, HumanMessage("BAT-21700 광주(GWJ) 창고 재고 알려줘")]
    ai = model_with_tools.invoke(messages)
    messages.append(ai)
    print("[1차 호출 요청]", ai.tool_calls)        # 모델이 GWJ로 check_inventory를 시도

    # 도구를 실행하면 ToolException이 나므로, 그 사유를 ToolMessage로 모델에 회신합니다.
    for call in ai.tool_calls:
        try:
            result = check_inventory.invoke(call["args"])
        except ToolException as e:
            result = str(e)                       # 실패 사유를 그대로 모델에 전달
            print("[도구 실패]", result)
        # tool_call_id를 요청 id와 똑같이 맞춰야 모델이 결과를 짝지을 수 있습니다.
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    # 모델이 실패를 관찰하고 사용자에게 재확인을 요청하는지 봅니다 (지어내지 않아야 합니다).
    recovered = model_with_tools.invoke(messages)
    print("[회복 응답]", recovered.content)        # "제품 코드와 창고를 다시 확인해 주십시오" 류

    # 체크포인트: 도구 실패 후 모델이 값을 지어내지 않고 재확인을 요청하면
    #            ToolException 기반 회복 루프가 정상 동작하는 것입니다.


def main() -> None:
    print("=== 도구의 실패 동작 (키 불필요) ===")
    show_tool_failure_without_llm()

    if not os.getenv("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY 미설정: LLM 회복 루프는 건너뜁니다.")
        print('  예) .env에 OPENAI_API_KEY=sk-... 입력 후 다시 실행하십시오.')
        return

    model = init_chat_model(MODEL)  # 강의 직전 최신 모델과 가격을 재확인하십시오.

    print("\n=== ToolException 회복 루프 (LLM) ===")
    recovery_loop(model)


if __name__ == "__main__":
    main()
