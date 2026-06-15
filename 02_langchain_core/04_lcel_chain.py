"""04 - ChatPromptTemplate과 LCEL 파이프로 재사용 가능한 체인을 만든다.

이 예제 하나만으로 다음을 익힙니다.
  - ChatPromptTemplate으로 {변수} 자리를 비워 둔 프롬프트 양식을 만든다.
  - 파이프(|)로 프롬프트와 모델을 연결한다 (이것이 LCEL 체인).
  - 같은 체인에 변수만 바꿔 재사용한다.
  - 본문에 들어가는 리터럴 중괄호는 {{ }}로 이스케이프한다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 02_langchain_core/04_lcel_chain.py

키가 없으면 안내만 출력하고 종료합니다.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


# 이 함수는 -> ChatPromptTemplate 라는 타입 힌트로 "프롬프트 양식 객체를 돌려준다"고 알려 줍니다.
def build_prompt() -> ChatPromptTemplate:
    """변수 자리를 비워 둔 재사용 가능한 프롬프트 양식을 만든다 (모델 호출 전)."""
    # ChatPromptTemplate은 {변수} 자리를 비워 둔 재사용 가능한 메시지 틀입니다.
    # 중괄호 { } 안의 이름(예: {역할})은 나중에 값을 채워 넣을 빈칸입니다.
    # from_messages는 (역할, 내용) 짝의 리스트를 받습니다. ("system", ...)처럼 괄호로 묶은 두 값이 튜플(고정된 짝)입니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 {역할}이다. 쉽게 설명한다."),
        ("human", "{질문}"),
    ])

    # 값만 채우면 실제 메시지로 완성됩니다. 모델 호출 없이 "어떤 메시지가 만들어지는지"만 봅니다.
    # 중괄호 { } 그 자체는 딕셔너리(key: value 짝의 묶음)입니다. "역할"이라는 빈칸에 "교사"를 채우라는 뜻입니다.
    messages = prompt.invoke({"역할": "교사", "질문": "LCEL이 뭐야?"})
    print("[양식이 만든 메시지]")
    # 완성된 메시지들을 하나씩 꺼내 출력합니다. messages.messages는 메시지 객체들의 리스트입니다.
    for m in messages.messages:
        # f"..."는 f-string으로, 문자열 안 { } 자리에 변수 값을 끼워 넣어 줍니다.
        # m.type은 메시지 종류(system/human 등), m.content는 그 내용입니다.
        print(f"  [{m.type}] {m.content}")
    return prompt  # 만든 양식을 호출한 쪽으로 돌려줍니다(아래 run_chain에서 다시 씁니다).


# 이 함수는 인자를 두 개(model, prompt) 받습니다. 인자는 쉼표로 구분해 순서대로 전달됩니다.
def run_chain(model, prompt: ChatPromptTemplate) -> None:
    """파이프(|)로 프롬프트와 모델을 연결하고, 변수만 바꿔 재사용한다."""
    # prompt | model 은 "프롬프트로 메시지를 만들고, 그 메시지를 모델에 넣는" 흐름을 하나로 묶습니다.
    # | 기호는 LangChain에서 "앞 단계와 뒤 단계를 이어 붙이는" 연결 연산자로 쓰입니다(LCEL 파이프).
    # 파이프는 앞 단계의 출력이 다음 단계의 입력이 되도록 합성합니다(왼쪽에서 오른쪽으로 흐름).
    chain = prompt | model

    # 이제 체인을 한 번에 호출합니다. 변수 값만 딕셔너리로 넘기면 됩니다.
    # chain도 invoke를 가진 하나의 실행 단위라, 모델을 직접 부르는 것과 같은 방식으로 씁니다.
    result = chain.invoke({"역할": "교사", "질문": "LCEL을 한 문장으로 설명해줘"})
    print("\n[체인 답변]", result.content)

    # 틀(체인)은 그대로 두고 입력만 바꿔 여러 번 재사용합니다.
    print("\n[재사용 — 역할만 교체]")
    print("  [교사]", chain.invoke({"역할": "교사", "질문": "RAG가 뭐야?"}).content)
    print("  [면접관]", chain.invoke({"역할": "면접관", "질문": "RAG가 뭐야?"}).content)
    # 체크포인트: 역할만 바꿨는데 답의 톤이 달라지면 체인 재사용을 이해한 것입니다.


def escape_braces(model) -> None:
    """본문의 리터럴 중괄호는 {{ }}로 이스케이프한다."""
    # 프롬프트 본문에 진짜 중괄호를 넣고 싶으면 두 번 겹쳐 {{ }}로 적습니다.
    # 한 번만 적으면 변수 자리로 오인되어 KeyError(없는 빈칸을 찾다 나는 오류)가 납니다.
    # 작은따옴표(' ')와 큰따옴표(" ")는 둘 다 문자열입니다. 안에 "가 들어가서 바깥은 '로 감쌌습니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", '다음 JSON 형식으로만 답한다: {{"answer": "..."}}'),  # {{ }}는 리터럴 중괄호
        ("human", "{질문}"),
    ])
    chain = prompt | model
    print("[이스케이프된 양식의 답]", chain.invoke({"질문": "하늘은 무슨 색이야?"}).content)
    # 예: {"answer": "파란색"} 형태로 답합니다(중괄호가 변수로 오인되지 않음).


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    model = init_chat_model(MODEL)

    print("=== 프롬프트 양식 만들기 ===")
    prompt = build_prompt()

    print("\n=== LCEL 체인 연결과 재사용 ===")
    run_chain(model, prompt)

    print("\n=== 중괄호 이스케이프 ===")
    escape_braces(model)


if __name__ == "__main__":
    main()
