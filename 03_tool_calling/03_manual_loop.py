"""03 - 한 번의 왕복을 '반복'으로 일반화한다 (수동 루프 + 안전판).

이 예제 하나만으로 다음을 익힙니다.
  - 뒤 계산이 앞 결과에 의존하는 다단계 질문은 한 번의 왕복으로 끝나지 않는다.
  - tool_calls가 빌 때까지 도는 수동 루프로 여러 바퀴를 처리한다.
  - 정상 종료 조건(tool_calls가 빔)과 비정상 대비 안전판(MAX_TURNS)을 둘 다 둔다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 03_tool_calling/03_manual_loop.py

키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.messages import HumanMessage, ToolMessage

load_dotenv()

MODEL = "openai:gpt-5.4-mini"

# 루프가 끝내 끝나지 않을 때를 대비한 최대 반복 횟수입니다.
# 단순 조회는 3~5회, 복잡한 작업은 더 크게 잡되 무한정은 피합니다.
MAX_TURNS = 5


@tool
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱한다."""
    return a * b


def manual_loop(model_with_tools, tool_map) -> None:
    """다단계 질문을 tool_calls가 빌 때까지 도는 수동 루프로 처리한다."""
    # 뒤 계산이 앞 결과에 의존하는 다단계 질문입니다 (덧셈 결과에 곱셈을 해야 함).
    # 모델은 덧셈 결과를 본 뒤에야 곱셈의 인자를 채울 수 있으므로, 한 번에 두 도구를 다 부를 수 없습니다.
    messages = [HumanMessage("3 더하기 5를 한 다음 그 결과에 4를 곱하면?")]

    # 1차 호출: 모델이 답 대신 첫 도구 호출 요청을 반환합니다.
    ai = model_with_tools.invoke(messages)
    messages.append(ai)  # 대화 기록에 모델의 요청을 그대로 쌓아 둡니다
    print("[first call]", ai.tool_calls)

    # tool_calls가 빌 때까지 반복하되, 안전판(MAX_TURNS)으로 무한 반복을 막습니다.
    #   - while은 "괄호 안 조건이 참인 동안 안쪽 블록을 되풀이하라"는 반복문입니다.
    #   - and는 두 조건이 모두 참일 때만 참입니다. 여기서는 "부를 도구가 남아 있고(AND) 한도 안일 때만" 돕니다.
    #   - 정상 종료 조건: ai.tool_calls가 빔(더 부를 도구가 없으면 while 조건이 거짓이 되어 빠져나감).
    #   - 안전판: 모델이 끝내 종료를 못 정해 같은 호출을 반복할 때를 대비한 최대 반복 횟수.
    #   둘은 막는 상황이 달라 어느 하나로 대체할 수 없습니다(종료 조건만 두면 비정상 루프를, 안전판만 두면 정상 종료를 놓침).
    turn = 0
    while ai.tool_calls and turn < MAX_TURNS:
        turn += 1  # turn = turn + 1 의 줄임 표현입니다 (반복 횟수를 하나 늘림).

        # 한 응답에 호출이 여러 개일 수 있으므로, for로 전부 순회해 각각 실행합니다.
        for call in ai.tool_calls:
            chosen = tool_map[call["name"]]       # 요청한 이름의 도구를 고릅니다
            result = chosen.invoke(call["args"])  # 실제 함수를 실행합니다
            # tool_call_id를 요청 id와 똑같이 맞춰야 모델이 "이 결과가 그 요청의 답"임을 압니다.
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        # 결과가 담긴 메시지로 다시 호출합니다. 모델이 다음 도구를 제안하거나 최종 답을 냅니다.
        ai = model_with_tools.invoke(messages)
        messages.append(ai)

    # while을 빠져나온 뒤, 아직도 tool_calls가 남아 있으면 안전판에 걸린 비정상 종료입니다.
    if ai.tool_calls:
        print(f"[안전판] {MAX_TURNS}회 안에 끝나지 않아 루프를 멈췄습니다. 도구나 종료 조건을 점검하십시오.")
        # 흔한 원인은 tool_call_id 불일치입니다. 모델이 결과를 못 받은 것으로 보고 같은 도구를 다시 부릅니다.
        return

    # 더 부를 도구가 없으면(정상 종료) 모델이 최종 자연어 답변을 완성합니다.
    print("[final]", ai.content)
    # 예: 3 더하기 5는 8이고, 거기에 4를 곱하면 32입니다.

    # 체크포인트: 덧셈 호출, 결과 되돌림, 곱셈 호출, 결과 되돌림 순서를 거쳐 32가 나오면 루프가 완성된 것입니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    model = init_chat_model(MODEL)
    tools = [add, multiply]
    # {도구이름: 도구객체} 사전을 도구 리스트에서 한 번에 만듭니다(딕셔너리 컴프리헨션).
    tool_map = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    print("=== 수동 루프 (다단계 도구 호출 + MAX_TURNS 안전판) ===")
    manual_loop(model_with_tools, tool_map)


if __name__ == "__main__":
    main()
