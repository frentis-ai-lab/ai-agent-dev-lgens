"""04 - 한 응답에 여러 호출이 담길 때 — 전부 순회하고, 병렬을 제어한다.

이 예제 하나만으로 다음을 익힙니다.
  - 서로 무관한 질문은 한 응답에 여러 tool_calls가 담길 수 있다 (개수 확인).
  - 호출이 몇 개든 전부 순회해 실행하고, 각각에 ToolMessage를 되돌린다 (하나라도 빠뜨리면 안 됨).
  - parallel_tool_calls로 한 응답에 여러 호출을 담을지 하나만 담을지 제어한다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 03_tool_calling/04_parallel_calls.py

키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.messages import HumanMessage, ToolMessage

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


@tool
def get_weather(city: str) -> str:
    """주어진 도시의 현재 날씨를 한 줄로 알려준다."""
    # 실제로는 외부 날씨 API를 부르겠지만, 실습에서는 고정 데이터로 흉내만 냅니다.
    fake_db = {"서울": "맑음, 22도", "도쿄": "흐림, 19도", "보스턴": "비, 15도"}
    # .get(키, 기본값)은 사전에서 키를 찾되, 없으면 기본값을 돌려줍니다(KeyError를 피함).
    return fake_db.get(city, f"{city}의 날씨 정보가 없습니다.")


# 도구 이름으로 실제 도구를 찾기 위한 사전입니다.
TOOL_MAP = {get_weather.name: get_weather}


# ===========================================================================
# STEP 1 — 서로 무관한 질문은 한 응답에 여러 tool_calls가 담길 수 있다.
# ===========================================================================

def step1_count_multiple_calls(model_with_tools):
    """서로 의존하지 않는 두 질문에 호출이 몇 개 담기는지 개수를 확인한다."""
    # 서로 의존하지 않는 질문(서울 날씨, 도쿄 날씨)은 모델이 한 응답에 여러 호출을 담을 수 있습니다.
    # 03의 다단계 질문과 달리, 둘째 호출의 인자가 첫째 결과에 의존하지 않기 때문입니다.
    ai = model_with_tools.invoke([HumanMessage("서울이랑 도쿄 날씨를 둘 다 알려줘.")])

    # tool_calls가 한 개가 아니라 여러 개일 수 있습니다. len(리스트)은 항목 개수를 셉니다.
    print("[num calls]", len(ai.tool_calls))  # 예: 2 (서울 호출 + 도쿄 호출)
    for call in ai.tool_calls:
        print("  -", call["args"])  # 예: {'city': '서울'}, {'city': '도쿄'}

    # 체크포인트: 호출이 2개로 잡히면, 다음 스텝에서 "전부" 순회해야 하는 이유가 보입니다.
    return ai


# ===========================================================================
# STEP 2 — 호출이 몇 개든 전부 순회해 실행하고, 각각에 ToolMessage를 되돌린다.
# ===========================================================================

def step2_iterate_all_calls(model_with_tools, ai) -> None:
    """호출을 전부 순회해 실행하고, 각각에 대응하는 ToolMessage를 되돌린다."""
    # 대화 기록에 사용자 질문과 모델의 요청(ai)을 쌓아 둡니다.
    messages = [HumanMessage("서울이랑 도쿄 날씨를 둘 다 알려줘."), ai]

    # 핵심: 호출이 몇 개든 "전부" 순회하며 각각 실행하고, 각각에 대응하는 ToolMessage를 되돌려야 합니다.
    #   하나라도 빠뜨리면 모델은 답이 안 온 요청을 기다리며 최종 답을 못 냅니다.
    #   각 ToolMessage의 tool_call_id를 그 호출의 id와 맞춰, 어느 결과가 어느 호출의 답인지 짝지웁니다.
    for call in ai.tool_calls:
        result = TOOL_MAP[call["name"]].invoke(call["args"])
        # f"..."는 f-string으로, 문자열 안 { } 자리에 변수 값을 끼워 넣습니다.
        print(f"  - {call['args']} -> {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # 두 결과가 모두 담긴 메시지로 다시 호출하면, 모델이 두 도시 날씨를 한 답에 모읍니다.
    final = model_with_tools.invoke(messages)
    print("[final]", final.content)  # 예: 서울은 맑음 22도, 도쿄는 흐림 19도입니다.

    # 체크포인트: len(tool_calls)와 붙인 ToolMessage 개수가 같고, 두 도시 날씨가 한 답에 모이면 성공입니다.


# ===========================================================================
# STEP 3 — parallel_tool_calls로 한 응답에 여러 호출을 담을지 제어한다.
# ===========================================================================

def step3_parallel_control(model) -> None:
    """parallel_tool_calls로 한 응답에 여러 호출을 담을지(True) 하나만 담을지(False) 제어한다."""
    # 기본값은 병렬 허용(parallel_tool_calls=True)이라, 서로 무관한 호출은 한 응답에 여러 개가 담깁니다.
    parallel_on = model.bind_tools([get_weather])  # 명시하지 않으면 병렬 허용이 기본
    ai_on = parallel_on.invoke([HumanMessage("서울, 도쿄, 보스턴 날씨를 모두 알려줘.")])
    print("[parallel=on]  호출 수:", len(ai_on.tool_calls))  # 예: 3 (한 번에 세 호출)

    # parallel_tool_calls=False를 주면 모델이 한 번에 도구를 하나만 부르도록 제약합니다.
    #   호출 순서를 강제하거나, 외부 자원 경합(같은 파일 동시 수정 등)을 피하고 싶을 때 씁니다.
    parallel_off = model.bind_tools([get_weather], parallel_tool_calls=False)
    ai_off = parallel_off.invoke([HumanMessage("서울, 도쿄, 보스턴 날씨를 모두 알려줘.")])
    print("[parallel=off] 호출 수:", len(ai_off.tool_calls))  # 예: 1 (한 번에 하나씩)

    # 체크포인트: on에서는 한 응답에 여러 호출, off에서는 1개로 줄어들면 병렬 제어가 동작하는 것입니다.
    # 참고: off는 호출을 여러 라운드로 나누므로, 실제 답을 끝내려면 수동 루프(예제 03)로 여러 번 되돌려야 합니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    model = init_chat_model(MODEL)
    model_with_tools = model.bind_tools([get_weather])

    print("=== STEP 1: 여러 tool_calls 개수 확인 ===")
    ai = step1_count_multiple_calls(model_with_tools)

    print("\n=== STEP 2: 여러 호출 전부 순회 ===")
    step2_iterate_all_calls(model_with_tools, ai)

    print("\n=== STEP 3: parallel_tool_calls 제어 ===")
    step3_parallel_control(model)


if __name__ == "__main__":
    main()
