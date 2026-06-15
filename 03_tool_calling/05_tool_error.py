"""05 - 도구 에러를 예외가 아니라 '관찰'로 돌려준다 (ToolException).

이 예제 하나만으로 다음을 익힙니다.
  - 도구 안에서 의도적으로 ToolException을 던져, 잘못된 입력을 명확히 알린다.
  - 예외를 잡지 않으면 루프 전체가 멈춘다는 위험을 이해한다.
  - try/except로 예외를 잡아 실패 메시지를 ToolMessage 내용으로 되돌리면, 모델이 스스로 회복한다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 03_tool_calling/05_tool_error.py

키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# ToolException은 도구 안에서 "이 호출은 잘못됐다"고 의도적으로 알릴 때 던지는 예외입니다.
from langchain_core.tools import tool, ToolException
from langchain.messages import HumanMessage, ToolMessage

load_dotenv()

MODEL = "openai:gpt-5.4-mini"

MAX_TURNS = 5


@tool
def divide(a: float, b: float) -> float:
    """첫 번째 수를 두 번째 수로 나눈다."""
    # 0으로 나누기는 사람의 입력 실수로 흔히 발생합니다.
    # raise는 "여기서 예외를 일으켜 멈추라"는 키워드입니다.
    # ToolException을 던지면 그 메시지가 (try/except로 잡혀) 모델에게 전달되어,
    # 모델이 스스로 회복(재시도·해명)할 수 있습니다.
    if b == 0:
        raise ToolException("0으로는 나눌 수 없습니다. 두 번째 수를 0이 아닌 값으로 주십시오.")
    return a / b


TOOL_MAP = {divide.name: divide}


def tool_error_recovery(model_with_tools) -> None:
    """도구 실행 에러를 try/except로 잡아 ToolMessage에 담아 모델이 회복하게 한다."""
    # 0으로 나누기를 유도하는 질문입니다. divide 도구는 이때 ToolException을 던지도록 만들어 두었습니다.
    print("입력 질문: 10을 0으로 나눈 값을 알려줘.")
    print("기대 흐름: divide 도구가 0 나누기에서 ToolException -> 오류를 관찰로 되돌림 -> 모델이 회복")
    messages = [HumanMessage("10을 0으로 나눈 값을 알려줘.")]
    ai = model_with_tools.invoke(messages)
    messages.append(ai)

    turn = 0
    while ai.tool_calls and turn < MAX_TURNS:
        turn += 1
        print(f"\n--- 루프 {turn}바퀴 ---")
        print(f"  모델이 부르라는 도구:", ai.tool_calls)
        for call in ai.tool_calls:
            # try/except는 "안쪽을 시도하다가 지정한 예외가 나면 except 블록으로 넘어가라"는 구문입니다.
            try:
                # 도구 실행이 성공하면 결과를 그대로 되돌립니다.
                result = TOOL_MAP[call["name"]].invoke(call["args"])
                content = str(result)
            except ToolException as e:
                # 핵심: 오류로 루프를 멈추지 않고, 오류 메시지를 ToolMessage 내용으로 되돌립니다.
                #   예외를 잡지 않으면 이 예외가 루프 전체를 멈춰 세웁니다(챗봇이 갑자기 죽는 셈).
                #   대신 실패도 하나의 "관찰"로 모델에 돌려주면, 모델이 "왜 실패했는지" 읽고
                #   사용자에게 설명하거나 다른 인자로 다시 시도할 수 있습니다.
                #   as e는 잡은 예외 객체를 e라는 이름으로 받아 쓰겠다는 뜻입니다.
                content = f"도구 실행 오류: {e}"
                print("  - 도구 오류를 모델에게 전달:", content)
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

        ai = model_with_tools.invoke(messages)
        messages.append(ai)

    if ai.tool_calls:
        print(f"[안전판] {MAX_TURNS}회 안에 끝나지 않아 루프를 멈췄습니다.")
        return

    # 모델은 오류 내용을 받아 "0으로 나눌 수 없다"는 취지의 답을 자연어로 정리합니다.
    print(f"\n프로그램이 예외로 죽지 않고 {turn}바퀴 만에 회복했습니다. 최종 답:")
    print("[final]", ai.content)

    # 체크포인트: 프로그램이 예외로 죽지 않고, 오류 메시지가 모델 답변에 반영되어 회복되면 성공입니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    model = init_chat_model(MODEL)
    model_with_tools = model.bind_tools([divide])

    print("=== 도구 에러 처리와 회복 (ToolException을 ToolMessage로 되돌림) ===")
    tool_error_recovery(model_with_tools)


if __name__ == "__main__":
    main()
