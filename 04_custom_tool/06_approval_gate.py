"""06 - 승인 게이트: 되돌릴 수 없는 작업을 프롬프트가 아니라 코드로 막는다.

이 예제 하나만으로 다음을 익힙니다.
  - 되돌릴 수 없는 도구(재고 차감 등)는 confirmed 플래그로 코드 안에서 막는다.
  - 프롬프트 규칙만 믿지 않는다 — 프롬프트는 경향의 유도일 뿐 강제가 아니다.
  - LLM이 confirmed 없이 부르면 코드 가드에 막히고, 그 사유로 사용자에게 확인을 요청한다.

프롬프트는 1차 방어선, 코드 가드는 2차 방어선입니다. 정확성·안전이 결정적이면 둘을 함께 둡니다.

단서(과장하지 말 것):
  confirmed는 도구 인자이므로 모델이 스스로 채울 수 있습니다. 따라서 이 예제만으로는
  완전한 강제가 아닙니다. 실서비스에서는 confirmed를 호출부(애플리케이션 코드)에서만 세팅하거나,
  LangGraph의 human-in-the-loop interrupt로 사람 승인을 받은 뒤에만 실행되게 설계합니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 04_custom_tool/06_approval_gate.py

키가 없으면 코드 가드(LLM 불필요)만 시연하고 LLM 경로는 건너뜁니다.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, ToolException

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


# ===========================================================================
# 승인 게이트가 박힌 도구 — confirmed=True일 때만 실제로 실행된다.
# ===========================================================================

@tool("adjust_inventory")
def adjust_inventory(sku: str, delta: int, confirmed: bool = False) -> str:
    """재고 수량을 delta만큼 가감한다. 되돌릴 수 없으므로 confirmed=True일 때만 실행한다."""
    # 되돌릴 수 없는 작업은 프롬프트 규칙만 믿지 말고 코드 안에서 한 번 더 막습니다 (승인 게이트).
    if not confirmed:                            # 확인 플래그가 없으면 실행 차단
        raise ToolException("재고 변경은 confirmed=True로 사용자 확인을 받은 뒤에만 가능합니다")
    # {delta:+d}는 정수에 부호를 항상 붙입니다 (예: -100 → "-100", 100 → "+100").
    return f"{sku.upper()} 재고를 {delta:+d}개 조정했습니다."


def code_guard_without_llm() -> None:
    """LLM과 무관하게, 코드 가드가 confirmed 없는 호출을 막는지 확인한다 (키 불필요)."""
    # 모델이 규칙을 어기거나 사용자가 실수로 부를 수 있으므로, 코드 안에서 막습니다.
    print("(1) confirmed 없이 직접 호출 → 코드 가드에 막혀야 정상")
    print("    입력: {'sku': 'BAT-21700', 'delta': -100}  (confirmed 누락)")
    try:
        adjust_inventory.invoke({"sku": "BAT-21700", "delta": -100})
    except ToolException as e:
        print("  가드 작동:", e)                   # 재고 변경은 confirmed=True로 ...

    # 사람이 검토한 뒤에만 confirmed=True로 실제 실행합니다.
    # (실무에서는 이 자리에서 버튼 클릭·승인 API 등 사람의 명시적 확인을 받습니다.)
    print("(2) 사람이 확인 후 confirmed=True → 이때만 실제 실행")
    print("    입력: {'sku': 'BAT-21700', 'delta': -100, 'confirmed': True}")
    print("[사용자 승인 후 실행]", adjust_inventory.invoke(
        {"sku": "BAT-21700", "delta": -100, "confirmed": True}
    ))                                            # BAT-21700 재고를 -100개 조정했습니다.

    # 체크포인트: confirmed 없는 호출이 막히고, confirmed=True일 때만 실행되면 코드 가드가 정상입니다.


def approval_gate_with_llm(model) -> None:
    """LLM 경로에서도 같은 게이트가 작동한다 (모델이 부르면 막히고, 확인을 요청)."""
    # 모델이 confirmed 없이 도구를 부르면 코드 가드에 막히고,
    # 그 사유가 모델에 회신되어 "정말 변경할까요?"라고 사용자에게 확인을 요청하게 됩니다.
    system_prompt = SystemMessage(
        "너는 사내 재고를 관리하는 물류 비서다. "
        "재고를 변경(adjust_inventory)하기 전에는 반드시 사용자에게 확인을 받아야 한다. "
        "확인 전에는 confirmed=True로 호출하지 말고, 먼저 사용자에게 변경 내용을 알리고 동의를 구하라."
    )
    model_with_tools = model.bind_tools([adjust_inventory])

    print("질문:", "BAT-21700 재고를 100개 줄여줘", "(되돌릴 수 없는 변경 요청)")
    messages = [system_prompt, HumanMessage("BAT-21700 재고를 100개 줄여줘")]
    ai = model_with_tools.invoke(messages)
    messages.append(ai)
    print("[모델 1차] tool_calls:", ai.tool_calls)
    print("[모델 1차] content  :", ai.content)     # 보통 먼저 확인을 요청하거나 confirmed 없이 시도

    # 모델이 도구를 시도했다면, 코드 게이트가 막고 그 사유를 모델에 되돌려 줍니다.
    if ai.tool_calls:
        print("[코드 게이트] 모델이 도구를 시도 → confirmed 없으면 막고 사유를 회신합니다")
    for call in ai.tool_calls:
        try:
            result = adjust_inventory.invoke(call["args"])
        except ToolException as e:
            result = str(e)                       # 승인 게이트가 막은 사유를 모델에 전달
            print("  게이트 차단:", result)
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    if ai.tool_calls:                             # 게이트에 막혔다면 모델이 사용자에게 확인을 요청합니다
        confirm_turn = model_with_tools.invoke(messages)
        print("[게이트 후 응답]", confirm_turn.content)  # "정말 100개 줄일까요?" 류

    # 사용자가 동의했다고 가정하고, 사람이 검토한 뒤에만 confirmed=True로 실제 실행합니다.
    print("[사용자 승인 후 실행] confirmed=True로 실제 변경 (사람 확인을 거친 뒤)")
    print("[사용자 승인 후 실행]", adjust_inventory.invoke(
        {"sku": "BAT-21700", "delta": -100, "confirmed": True}
    ))                                            # BAT-21700 재고를 -100개 조정했습니다.

    # 체크포인트: confirmed 없는 호출은 LLM 경로든 직접 호출이든 항상 막히고,
    #            사람의 확인(confirmed=True)을 거친 뒤에만 실행되면 승인 게이트가 정상입니다.


def main() -> None:
    print("=== 승인 게이트 코드 가드 (키 불필요) ===")
    code_guard_without_llm()

    if not os.getenv("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY 미설정: LLM 경로는 건너뜁니다 (코드 가드만 시연).")
        print('  예) .env에 OPENAI_API_KEY=sk-... 입력 후 다시 실행하십시오.')
        return

    model = init_chat_model(MODEL)  # 강의 직전 최신 모델과 가격을 재확인하십시오.

    print("\n=== 승인 게이트 (LLM 경로) ===")
    approval_gate_with_llm(model)


if __name__ == "__main__":
    main()
