"""02 - 메시지로 역할을 정하고, 응답을 누적해 맥락을 잇는다.

이 예제 하나만으로 다음을 익힙니다.
  - SystemMessage로 모델의 역할·형식 규칙을 먼저 고정한다.
  - HumanMessage·AIMessage를 리스트로 쌓아 멀티턴 대화를 만든다.
  - 모델은 무상태이며, 맥락이 이어지는 것은 우리가 대화 전체를 매번 다시 건네기 때문임을 확인한다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 02_langchain_core/02_messages_context.py

키가 없으면 안내만 출력하고 종료합니다.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# 대화는 네 종류의 메시지로 표현합니다: System(역할·규칙)·Human(사용자 입력)·AI(모델 응답)·Tool(도구 결과).
# 이 예제에서는 앞의 셋을 직접 씁니다. Tool은 다음 챕터(도구 호출)에서 만납니다.
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


# 이 함수는 model을 인자(괄호 안 입력값)로 받습니다. main에서 만든 모델을 건네받아 씁니다.
def role_with_system_message(model) -> None:
    """SystemMessage로 역할·형식을 고정하면 답의 태도가 바뀐다."""
    # 메시지 리스트의 맨 앞에 SystemMessage를 두면, 모델의 역할·규칙을 먼저 고정할 수 있습니다.
    # 대괄호 [ ] 안에 메시지를 쉼표로 나열하면 "메시지 리스트"가 됩니다. 줄을 바꿔 써도 됩니다.
    response = model.invoke([
        SystemMessage("너는 한 단어로만 답하는 비서다."),  # 역할·형식 규칙 (대화 맨 앞에 둠)
        HumanMessage("대한민국의 수도는?"),               # 사용자의 실제 질문
    ])
    print("[역할 고정] 한 단어 답:", response.content)  # 예: 서울

    # 비교용: 같은 질문이라도 시스템 메시지가 없으면 보통 더 길게 답합니다.
    plain = model.invoke([HumanMessage("대한민국의 수도는?")])
    print("[역할 없음] 답:", plain.content)  # 예: 대한민국의 수도는 서울입니다. ...


def multiturn_accumulation(model) -> None:
    """응답을 리스트에 누적하면 다음 턴이 앞 맥락을 이어받는다."""
    # 1턴: 첫 질문을 보냅니다. messages는 메시지를 담아 둘 리스트(변수)입니다.
    #      지금은 사용자 메시지 한 개만 담겨 있습니다.
    messages = [HumanMessage("대한민국의 수도는?")]
    first = model.invoke(messages)
    print("1턴 답:", first.content)

    # 핵심: invoke가 돌려준 결과(AIMessage)를 "그대로" 리스트에 누적합니다.
    #       .append(값)은 리스트 끝에 값을 하나 덧붙이는 메서드입니다. 원래 리스트가 바뀝니다.
    #       .content만 꺼내 새 객체로 감싸지 않습니다. 결과 객체 자체가 이미 AIMessage입니다.
    messages.append(first)

    # 2턴: 앞 맥락을 가리키는 질문을 덧붙여 다시 보냅니다.
    #      "그 도시"는 위 대화(서울)를 가리킵니다. 누적이 없으면 모델은 무엇인지 모릅니다.
    #      이제 messages에는 [질문1, 답1, 질문2] 세 개가 순서대로 쌓여 있습니다.
    messages.append(HumanMessage("그 도시의 인구는 대략 몇 명이야?"))
    second = model.invoke(messages)
    print("2턴 답(맥락 이어받음):", second.content)  # 서울 인구를 답하면 맥락 전달 성공

    # 체크포인트: "그 도시"가 서울로 이어지면 멀티턴 맥락 전달을 이해한 것입니다.
    #   - 모델은 호출 사이에 아무것도 기억하지 못합니다(무상태).
    #   - 맥락이 이어지는 것은 우리가 매 호출마다 지금까지의 대화 전체를 다시 건네기 때문입니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    # 모델을 한 번만 만들어, 아래 두 함수에 같은 모델을 인자로 건넵니다.
    model = init_chat_model(MODEL)

    # 문자열 안의 \n은 줄바꿈을 뜻하는 기호입니다. 출력 시 빈 줄 하나가 들어갑니다.
    print("=== 시스템 메시지로 역할 고정 ===")
    role_with_system_message(model)

    print("\n=== 응답 누적으로 멀티턴 맥락 잇기 ===")
    multiturn_accumulation(model)


if __name__ == "__main__":
    main()
