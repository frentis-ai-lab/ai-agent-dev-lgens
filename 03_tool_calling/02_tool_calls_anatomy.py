"""02 - tool_calls의 구조를 해부하고, 한 번의 왕복을 완성한다.

이 예제 하나만으로 다음을 익힙니다.
  - 도구가 필요한 질문에 모델은 답(content)이 아니라 tool_calls를 돌려준다 (실행이 아니라 제안).
  - tool_calls 한 건을 뜯어 name·args·id 세 필드를 읽는다.
  - 모델이 요청한 도구를 코드가 손으로 실행한다.
  - 실행 결과를 ToolMessage로 되돌리고, tool_call_id로 호출과 결과를 짝지어 최종 답을 받는다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 03_tool_calling/02_tool_calls_anatomy.py

키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
# 대화는 네 종류의 메시지로 표현합니다: System·Human·AI·Tool.
# 이 예제에서 마지막 ToolMessage(도구 결과)를 더해 네 종류를 모두 채웁니다.
# v1 권장 경로입니다. langchain_core.messages에서 가져와도 동일하게 동작합니다.
from langchain.messages import HumanMessage, ToolMessage

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


# 도구 이름으로 실제 도구 객체를 찾기 위한 사전입니다.
# 모델이 보내는 요청에는 도구 "이름"만 들어 있으므로, 이름으로 실물을 찾을 길이 필요합니다.
# {도구이름: 도구객체} 형태의 딕셔너리를 미리 만들어 둡니다.
TOOL_MAP = {add.name: add, multiply.name: multiply}

# 이 예제 전체가 쓰는 하나의 질문입니다. 계산이 필요해 모델이 도구를 부르도록 유도합니다.
QUESTION = "3 더하기 5는 얼마야?"


# ===========================================================================
# STEP 1 — 도구가 필요한 질문에 모델은 답 대신 tool_calls를 돌려준다.
# ===========================================================================

def step1_observe_tool_calls(model_with_tools):
    """계산 질문을 던지면 모델이 답(content) 대신 tool_calls를 돌려주는 것을 본다."""
    # 계산이 필요한 질문을 던지면, 모델은 답을 직접 쓰지 않고 "이 도구를 이렇게 불러 달라"고 제안합니다.
    # 핵심: 모델은 도구를 실행하지 않습니다. 실행은 우리 코드의 몫입니다(제안과 실행의 분리).
    ai = model_with_tools.invoke([HumanMessage(QUESTION)])

    # 이때 content는 비어 있고(또는 짧고), 실제 의도는 tool_calls에 담깁니다.
    # repr(값)은 그 값을 따옴표까지 보이게 출력해, 빈 문자열('')도 한눈에 구분되게 합니다.
    print("[content]   ", repr(ai.content))  # 예: '' (아직 최종 답이 아니므로 비어 있음)
    print("[tool_calls]", ai.tool_calls)
    # 예: [{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_abc', 'type': 'tool_call'}]

    # 체크포인트: content가 비고 tool_calls에 add 호출 요청이 담기면, 모델이 "도구를 쓰겠다"고 판단한 것입니다.
    return ai


# ===========================================================================
# STEP 2 — tool_calls 한 건을 뜯어 name·args·id를 읽는다.
# ===========================================================================

def step2_dissect_one_tool_call(ai) -> None:
    """tool_calls 한 건의 속(name·args·id)을 뜯어 본다."""
    # tool_calls는 호출 요청들의 리스트입니다. [0]은 그중 첫 번째 항목을 꺼냅니다(번호는 0부터 셉니다).
    # 모델이 평범한 문장으로 요청을 적었다면 우리가 다시 파싱해야 하지만,
    # tool_calls는 처음부터 구조화된 객체(딕셔너리)라 파싱이 필요 없습니다.
    first = ai.tool_calls[0]

    # 각 항목은 딕셔너리이므로 ["key"]로 값을 꺼냅니다.
    print("[call name]", first["name"])  # 예: add  (부를 도구 이름)
    print("[call args]", first["args"])  # 예: {'a': 3, 'b': 5}  (채워 넣은 인자)
    print("[call id]  ", first["id"])    # 결과를 되돌릴 때 이 id로 호출과 결과를 짝지웁니다

    # 체크포인트: name·args·id 세 가지가 보이면, 다음 단계에서 무엇을 실행하고 무엇으로 짝을 맞출지 알게 된 것입니다.
    #   id는 "이 결과가 어느 호출의 답인가"를 묶는 송장 번호와 같아, 여러 호출이 한 응답에 담길 때 특히 결정적입니다.


# ===========================================================================
# STEP 3 — 모델이 요청한 도구를 코드가 손으로 실행한다.
# ===========================================================================

def step3_run_tool_call_by_hand(ai) -> None:
    """요청에 담긴 이름으로 실제 도구를 골라 args를 넣어 실행한다 (아직 모델에 안 돌려준다)."""
    # 첫 요청을 꺼내, 이름으로 도구를 고르고 args를 그대로 넣어 실제 함수를 실행합니다.
    call = ai.tool_calls[0]
    chosen = TOOL_MAP[call["name"]]       # 요청한 이름의 도구를 사전에서 고릅니다
    result = chosen.invoke(call["args"])  # 모델이 채운 args를 그대로 넣어 실행합니다
    print("[실행 결과]", result)          # 예: 8

    # 체크포인트: 모델의 요청대로 도구를 실행해 8이 나오면, 다음은 이 결과를 모델에 되돌려 줄 차례입니다.
    #   도구를 한 번 실행했다고 모델이 결과를 알아서 가져가지는 않습니다. 되돌리는 책임은 코드에 있습니다.


# ===========================================================================
# STEP 4 — 결과를 ToolMessage로 되돌려 최종 자연어 답을 받는다 (한 번의 왕복 완성).
# ===========================================================================

def step4_return_with_tool_message(model_with_tools, ai) -> None:
    """실행 결과를 ToolMessage로 모델에 되돌려, 최종 자연어 답을 받는다."""
    # 대화 기록 리스트를 만들고, 그 안에 모델의 요청(ai)을 그대로 쌓아 둡니다.
    # 모델이 무엇을 제안했는지 기록에 남아 있어야, 곧 붙일 ToolMessage가 그 제안의 답임을 모델이 압니다.
    messages = [HumanMessage(QUESTION), ai]

    # 요청을 실행한 뒤, 결과를 ToolMessage로 담습니다.
    #   content에는 결과를 문자열로(str(...)) 넣습니다.
    #   tool_call_id를 요청 id와 똑같이 맞춰야 모델이 "이 결과가 그 요청의 답"임을 압니다.
    call = ai.tool_calls[0]
    result = TOOL_MAP[call["name"]].invoke(call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # 결과가 담긴 메시지로 다시 호출하면, 모델이 비로소 자연어 최종 답을 작성합니다.
    final = model_with_tools.invoke(messages)
    print("[final]", final.content)  # 예: 3 더하기 5는 8입니다.

    # 체크포인트: 도구 실행, ToolMessage 되돌림, 재호출 순서로 자연어 답이 나오면 한 번의 왕복을 완성한 것입니다.


# ===========================================================================
# 실행 진입점 — 한 질문으로 STEP 1~4를 차례로 따라갑니다.
# ===========================================================================

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return

    # 모델을 만들고 도구를 묶습니다. 이 묶인 모델을 네 스텝이 공유합니다.
    model = init_chat_model(MODEL)
    model_with_tools = model.bind_tools([add, multiply])

    print("=== STEP 1: tool_calls 관찰 ===")
    ai = step1_observe_tool_calls(model_with_tools)

    print("\n=== STEP 2: tool_call 한 건 해부 (name·args·id) ===")
    step2_dissect_one_tool_call(ai)

    print("\n=== STEP 3: 요청한 도구 손으로 실행 ===")
    step3_run_tool_call_by_hand(ai)

    print("\n=== STEP 4: ToolMessage로 결과 되돌려 최종 답 받기 ===")
    step4_return_with_tool_message(model_with_tools, ai)


if __name__ == "__main__":
    main()
