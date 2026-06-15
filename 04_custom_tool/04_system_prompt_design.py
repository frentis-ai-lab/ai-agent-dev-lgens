"""04 - 시스템 프롬프트 설계 (역할·제약·예시·형식) 그리고 안티패턴.

이 예제 하나만으로 다음을 익힙니다.
  - 좋은 시스템 프롬프트의 네 요소(역할·제약·예시·출력 형식)를 한 메시지에 담는다.
  - 같은 도구·같은 질문이라도 프롬프트에 따라 도구 사용·답 형식이 달라짐을 비교한다.
  - 빈약한 프롬프트와 두 가지 안티패턴(모호한 지시·과도한 지시)을 식별한다.

도구는 모델이 '무엇을 할 수 있는가'를 정하고, 시스템 프롬프트는 '어떻게 행동하는가'를 정합니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 04_custom_tool/04_system_prompt_design.py

키가 없으면 안내만 출력하고 종료합니다.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# 시스템 프롬프트는 SystemMessage로 표현하며, 메시지 목록의 맨 앞에 둡니다.
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, ToolException

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


# ===========================================================================
# 공통 도구 — 모든 프롬프트가 같은 도구·같은 질문을 씁니다. 달라지는 것은 프롬프트뿐입니다.
# ===========================================================================

_STOCK = {("BAT-21700", "ICN"): 1240, ("BAT-21700", "PUS"): 380}


@tool("check_inventory")
def check_inventory(sku: str, warehouse: str = "ICN") -> str:
    """지정한 창고의 제품 재고 수량을 조회한다.
    재고 수량을 알아야 할 때 사용한다. 예: 'BAT-21700 인천 창고 재고'."""
    qty = _STOCK.get((sku.upper(), warehouse.upper()))
    if qty is None:
        raise ToolException(f"재고 정보 없음: sku={sku}, warehouse={warehouse}")
    return f"{warehouse.upper()} 창고의 {sku.upper()} 재고는 {qty:,}개입니다."


def run_tool_loop(model_with_tools, messages: list, tool_map: dict) -> str:
    """tool_calls가 빌 때까지 도구를 실행하고 결과를 되돌린 뒤 최종 답변을 반환한다."""
    ai = model_with_tools.invoke(messages)
    messages.append(ai)
    while ai.tool_calls:
        for call in ai.tool_calls:
            chosen = tool_map[call["name"]]
            print(f"    → 도구 호출: {call['name']} {call['args']}")  # 프롬프트가 도구를 부르게 했는지 관찰
            try:
                result = chosen.invoke(call["args"])
            except ToolException as e:
                result = str(e)
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        ai = model_with_tools.invoke(messages)
        messages.append(ai)
    return ai.content


def _ask_with_prompt(model, system_prompt: SystemMessage) -> str:
    """주어진 시스템 프롬프트로 같은 질문을 던지고 최종 답변을 돌려준다."""
    # 시스템 프롬프트는 메시지 목록의 맨 앞에 둡니다. 모델은 대화 전에 이를 먼저 읽습니다.
    tools = [check_inventory]
    tool_map = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)
    # 모든 프롬프트가 동일한 이 질문을 받습니다. 달라지는 것은 system_prompt뿐입니다.
    print("  질문:", "BAT-21700 인천 창고 재고 얼마야?")
    messages = [system_prompt, HumanMessage("BAT-21700 인천 창고 재고 얼마야?")]
    return run_tool_loop(model_with_tools, messages, tool_map)


# ===========================================================================
# (1) 좋은 시스템 프롬프트 — 네 요소를 모두 갖춘다.
# ===========================================================================

def good_prompt(model) -> None:
    """역할·제약·예시·출력 형식을 한 메시지에 명시적으로 담는다."""
    prompt = SystemMessage(
        "너는 사내 재고를 조회하는 물류 비서다. "                              # (가) 역할
        "재고 수량은 절대 추측하지 말고 반드시 check_inventory 도구로 확인하라. "  # (나) 제약·도구 사용 규칙
        "도구가 실패하면 사용자에게 제품 코드를 다시 확인해 달라고 요청하라. "      # (나) 실패 처리 규칙
        "예시) 사용자: 'BAT-21700 인천 재고?' "                                  # (다) 예시(few-shot)
        "비서: 'ICN 창고의 BAT-21700 재고는 1,240개입니다.' "
        "답변은 한국어 한 문장으로, 수량은 천 단위 쉼표와 단위(개)를 함께 표기하라."  # (라) 출력 형식
    )
    print("  프롬프트: 역할·제약·예시·출력 형식 네 요소를 모두 담음")
    print("[좋은 프롬프트]", _ask_with_prompt(model, prompt))
    # 예: ICN 창고의 BAT-21700 재고는 1,240개입니다.  (도구로 확인 + 형식까지 준수)
    # 체크포인트: 도구로 확인하고 형식(쉼표·'개')까지 지킨 한 문장이 나오면 네 요소가 작동한 것입니다.


# ===========================================================================
# (2) 빈약한 프롬프트 — 역할만 있고 제약·예시·형식이 없다.
# ===========================================================================

def weak_prompt(model) -> None:
    """규칙·예시·형식이 빠지면 도구 사용·형식이 흔들린다 (good_prompt와 비교)."""
    prompt = SystemMessage("너는 비서다.")
    print("  프롬프트:", repr("너는 비서다."), "(역할만, 제약·예시·형식 없음)")
    print("[빈약한 프롬프트]", _ask_with_prompt(model, prompt))  # 형식이 들쭉날쭉하거나 추측이 섞일 수 있음
    # 체크포인트: good_prompt와 달리 도구 사용·형식이 흔들리면, 규칙·예시·형식의 빈자리가 곧 품질 차이입니다.


# ===========================================================================
# (3) 안티패턴 ① 모호한 지시 — 측정 불가능하거나 서로 충돌한다.
# ===========================================================================

def antipattern_vague(model) -> None:
    """'알아서 잘', '최대한 자세히/간결히'처럼 측정 불가·충돌하는 지시는 기준을 못 잡게 한다."""
    prompt = SystemMessage(
        "너는 알아서 잘하는 만능 비서다. 최대한 친절하고 최대한 간결하게, "  # '자세히'와 '간결히'가 충돌
        "필요하면 도구를 쓰고 아니면 안 써도 된다."                          # 언제 쓰는지 기준이 없음
    )
    print("  프롬프트: '최대한 친절+간결'(충돌), '필요하면 도구를'(기준 없음)")
    print("[모호한 지시]", _ask_with_prompt(model, prompt))  # 도구 사용 여부·형식이 호출마다 흔들릴 수 있음
    # 체크포인트: 호출마다 결과가 흔들리면, 측정 불가능·충돌하는 지시가 왜 위험한지 이해한 것입니다.


# ===========================================================================
# (4) 안티패턴 ② 과도한 지시 — 장황한 규칙이 충돌하고 핵심이 묻힌다.
# ===========================================================================

def antipattern_overload(model) -> None:
    """규칙이 지나치게 많고 서로 충돌하면 핵심이 묻히고 일부가 무시된다."""
    prompt = SystemMessage(
        "너는 비서다. 항상 공손해라. 항상 이모지를 써라. 절대 이모지를 쓰지 마라. "  # 충돌하는 규칙
        "모든 답은 5문장 이상이어야 한다. 모든 답은 한 단어로 해라. "                # 충돌하는 규칙
        "전문 용어를 많이 써라. 초등학생도 알게 써라. 도구는 가능하면 쓰지 마라."     # 충돌·핵심 흐림
    )
    print("  프롬프트: 규칙이 너무 많고 서로 충돌(이모지 써라/쓰지 마라 등), 핵심이 묻힘")
    print("[과도한 지시]", _ask_with_prompt(model, prompt))  # 규칙 충돌로 일관성 없는 답이 나오기 쉬움
    # 체크포인트: 좋은 프롬프트만 도구로 확인하고 형식을 지키며, 나머지가 흔들리면 설계의 효과를 이해한 것입니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    model = init_chat_model(MODEL)  # 강의 직전 최신 모델과 가격을 재확인하십시오.

    print("네 가지 프롬프트가 같은 도구·같은 질문을 받습니다. 달라지는 것은 시스템 프롬프트뿐입니다.")
    print("프롬프트에 따라 도구 사용 여부와 답 형식이 어떻게 갈리는지 비교합니다.\n")
    print("=== 좋은 시스템 프롬프트 (역할·제약·예시·형식) ===")
    good_prompt(model)
    print("\n=== 빈약한 프롬프트 (역할만) ===")
    weak_prompt(model)
    print("\n=== 안티패턴 ① 모호한 지시 ===")
    antipattern_vague(model)
    print("\n=== 안티패턴 ② 과도한 지시 ===")
    antipattern_overload(model)


if __name__ == "__main__":
    main()
