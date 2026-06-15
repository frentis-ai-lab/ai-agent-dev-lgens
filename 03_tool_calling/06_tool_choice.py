"""06 - tool_choice로 도구 사용을 강제·금지·지정한다.

이 예제 하나만으로 다음을 익힙니다.
  - tool_choice="any": 도구가 필요 없어 보이는 입력에도 반드시 하나는 부르도록 강제한다.
  - tool_choice="none": 계산 질문이어도 도구를 전혀 부르지 못하게 막는다.
  - tool_choice="<도구명>": 특정 도구 하나만 쓰도록 못박는다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 03_tool_calling/06_tool_choice.py

키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.messages import HumanMessage

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


@tool
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱한다."""
    return a * b


# 세 스텝이 공유하는 도구 목록입니다.
TOOLS = [add, multiply]


# ===========================================================================
# STEP 1 — tool_choice="any": 무엇이든 반드시 하나는 부르도록 강제한다.
# ===========================================================================

def step1_tool_choice_any(model) -> None:
    """tool_choice='any'로 도구 중 무엇이든 반드시 하나는 부르도록 강제한다."""
    # 기본값(자동)에서는 모델이 "도구가 필요한지"를 스스로 판단합니다.
    # "any"는 도구가 필요 없어 보이는 입력에도 모델이 억지로 도구를 부르려 시도하게 만듭니다.
    print("입력 질문: 그냥 인사나 해줘 (도구가 필요 없어 보이는 입력)")
    forced_any = model.bind_tools(TOOLS, tool_choice="any")
    ai_any = forced_any.invoke([HumanMessage("그냥 인사나 해줘")])
    print("[any] tool_calls:", ai_any.tool_calls)  # 비어 있지 않음 (도구를 부르려 함)
    print("관찰      : 인사 같은 입력에도 도구를 부르려 합니다(강제가 동작).")

    # 체크포인트: 인사 같은 입력에도 tool_calls가 비어 있지 않으면 강제가 동작하는 것입니다.


# ===========================================================================
# STEP 2 — tool_choice="none": 도구를 전혀 부르지 못하게 막는다.
# ===========================================================================

def step2_tool_choice_none(model) -> None:
    """tool_choice='none'으로 도구를 전혀 부르지 못하게 막는다."""
    # "none"은 계산 질문이어도 도구 없이 모델이 직접 답하도록 막습니다.
    print("입력 질문: 3 더하기 5는? (계산 질문이지만 도구 사용을 금지)")
    forbidden = model.bind_tools(TOOLS, tool_choice="none")
    ai_none = forbidden.invoke([HumanMessage("3 더하기 5는?")])
    print("[none] tool_calls:", ai_none.tool_calls)  # 예: [] (도구 호출 없음)
    print("[none] content:", ai_none.content)        # 모델이 직접 답을 작성
    print("관찰      : 계산 질문인데도 도구를 안 부르고(빈 리스트), 모델이 직접 답했습니다.")

    # 체크포인트: tool_calls가 빈 리스트이고 content에 직접 답이 들어오면 금지가 동작하는 것입니다.


# ===========================================================================
# STEP 3 — tool_choice="<도구명>": 특정 도구만 쓰도록 못박는다.
# ===========================================================================

def step3_tool_choice_specific(model) -> None:
    """tool_choice에 도구 이름을 주어 그 도구만 쓰도록 못박는다."""
    # 도구 이름을 직접 지정하면 그 도구만 강제됩니다. 곱셈을 물어도 add만 부르도록 못박는 식입니다.
    print("입력 질문: 3 곱하기 5는? (곱셈을 물어도 add만 쓰도록 못박음)")
    only_add = model.bind_tools(TOOLS, tool_choice="add")
    ai_add = only_add.invoke([HumanMessage("3 곱하기 5는?")])
    print("[add only] tool_calls:", ai_add.tool_calls)  # name이 항상 add로 고정
    print("관찰      : 곱셈을 물었는데도 name이 add로 고정됐습니다(특정 도구 강제).")

    # 체크포인트: 곱셈을 물어도 name이 add로 고정되어 나오면 특정 도구 강제를 이해한 것입니다.
    #   강제는 의도와 어긋난 도구를 부르게 만들 수 있으므로, 도구가 꼭 필요한 맥락에서만 신중히 씁니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    model = init_chat_model(MODEL)

    print("=== STEP 1: tool_choice='any' 강제 ===")
    step1_tool_choice_any(model)

    print("\n=== STEP 2: tool_choice='none' 금지 ===")
    step2_tool_choice_none(model)

    print("\n=== STEP 3: tool_choice='add' 특정 도구 지정 ===")
    step3_tool_choice_specific(model)


if __name__ == "__main__":
    main()
