"""03 - 모델 파라미터(temperature·max_tokens)와 스트리밍 출력.

이 예제 하나만으로 다음을 익힙니다.
  - temperature로 답의 다양성을 조절한다 (낮으면 일관, 높으면 다양).
  - max_tokens로 답의 최대 길이를 제한한다.
  - 파라미터는 "모델을 만들 때" 지정한다.
  - invoke는 한 번에 받고, stream은 토큰이 만들어지는 대로 조각으로 받는다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 02_langchain_core/03_params_streaming.py

키가 없으면 안내만 출력하고 종료합니다.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


def temperature_demo() -> None:
    """temperature로 답의 무작위성을 조절한다."""
    # temperature는 답의 무작위성입니다. 낮으면 일관되고, 높으면 표현이 다양해집니다.
    # 파라미터는 모델을 만들 때(init_chat_model) 지정합니다.
    # 함수에 값을 넘길 때 "이름=값" 형태(키워드 인자)로 주면 어떤 설정인지 분명해집니다.
    # 0.0 같은 소수점 숫자는 실수(float) 타입입니다.
    cold = init_chat_model(MODEL, temperature=0.0)  # 분류·추출처럼 일관성이 중요할 때
    hot = init_chat_model(MODEL, temperature=1.2)   # 아이디어·표현이 다양해야 할 때

    # 같은 질문(prompt)을 두 모델에 각각 던져 결과를 비교합니다.
    # invoke에 메시지 리스트 대신 문자열 하나만 넣어도 됩니다(자동으로 HumanMessage로 감싸집니다).
    prompt = "회의 시작 인사말을 한 문장으로."
    print("[temperature=0.0]", cold.invoke(prompt).content)
    print("[temperature=1.2]", hot.invoke(prompt).content)
    # 예: 두 답의 표현 폭이 다르게 나옵니다(낮은 쪽이 더 무난·반복적).


def max_tokens_demo() -> None:
    """max_tokens로 답의 최대 길이를 제한한다."""
    # max_tokens는 생성할 수 있는 출력 토큰의 상한입니다(정수). 비용·길이를 통제할 때 씁니다.
    short = init_chat_model(MODEL, max_tokens=20)  # 짧게 끊어 받고 싶을 때
    print("[max_tokens=20]", short.invoke("LangChain을 자세히 소개해줘.").content)
    # 예: 답이 20토큰 부근에서 잘려 짧게 나옵니다.


def streaming_demo() -> None:
    """stream으로 답을 토큰 단위로 흘려 받는다."""
    model = init_chat_model(MODEL)
    # invoke는 답이 완성된 뒤 한 번에 돌려줍니다.
    # stream은 토큰이 만들어지는 대로 작은 조각(AIMessageChunk)을 차례로 내놓는 이터레이터를 돌려줍니다.
    # 최종 내용은 invoke와 같고, 받는 방식만 다릅니다(체감 응답이 빨라집니다).
    # print의 end=""는 "출력 끝에 줄바꿈을 넣지 말라", flush=True는 "버퍼에 모으지 말고 즉시 화면에 보이라"는 뜻입니다.
    print("[스트리밍] ", end="", flush=True)
    # for ... in 반복문은 오른쪽(이터레이터)이 내놓는 값들을 하나씩 꺼내 chunk에 담아 블록을 반복 실행합니다.
    # stream은 조각을 만들어지는 대로 내놓으므로, 글자가 흘러나오듯 차례로 출력됩니다.
    for chunk in model.stream("LangChain을 두 문장으로 소개해줘."):
        # 조각의 텍스트는 .text로 꺼냅니다. 줄바꿈 없이 이어 붙여 출력합니다.
        print(chunk.text, end="", flush=True)
    print()  # 인자 없이 print()만 부르면 줄바꿈 한 번(흐른 글자 뒤를 마무리)을 출력합니다.
    # 체크포인트: 글자가 흘러나오듯 출력되면 스트리밍이 동작하는 것입니다.
    #   선택 기준 — 변수에 담아 다음 처리에 넘기면 invoke, 사용자에게 실시간으로 보여 주면 stream.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    print("=== temperature ===")
    temperature_demo()

    print("\n=== max_tokens ===")
    max_tokens_demo()

    print("\n=== 스트리밍 ===")
    streaming_demo()


if __name__ == "__main__":
    main()
