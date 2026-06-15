"""06 - 구조화 출력 심화: include_raw와 중첩 스키마.

이 예제 하나만으로 다음을 익힙니다.
  - include_raw=True로 파싱 결과(객체)와 원본 응답을 함께 받는다 (디버깅·토큰 확인용).
  - 스키마 안에 스키마를 넣어(중첩) 복잡한 데이터도 한 번에 구조화한다.

앞선 05_structured_output.py의 기본을 익힌 뒤에 보면 좋습니다.
이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 02_langchain_core/06_structured_advanced.py

키가 없으면 안내만 출력하고 종료합니다.
"""

import os
# List[str]는 "문자열들의 리스트"라는 타입 표현입니다. Optional은 05 파일과 같습니다.
from typing import Optional, List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


def include_raw(model) -> None:
    """파싱 결과와 '원본 응답'을 함께 받는다 (디버깅·토큰 확인용)."""
    # 필드가 하나뿐인 단순 스키마입니다. label에 감정 분류 결과(문자열)를 담게 합니다.
    class Sentiment(BaseModel):
        label: str = Field(description="positive, negative, neutral 중 하나")

    # include_raw=True를 주면 dict(딕셔너리: key로 값을 꺼내는 묶음)로 세 가지를 함께 받습니다.
    #   parsed         — 스키마로 파싱된 객체(Sentiment)
    #   raw            — 모델의 원본 응답 메시지(토큰 사용량 등 메타데이터 확인용)
    #   parsing_error  — 파싱 실패 시 예외, 성공이면 None
    # 파싱이 실패해도 예외로 끊기지 않고 parsing_error에 담겨, 원인을 코드로 다룰 수 있습니다.
    structured = model.with_structured_output(Sentiment, include_raw=True)
    print("받고 싶은 형태:", "Sentiment(label: str)  + include_raw=True")
    print("입력 문장:", "이번 업데이트 정말 마음에 들어요!")
    result = structured.invoke("이번 업데이트 정말 마음에 들어요!")

    # 딕셔너리는 result["키이름"] 형태로 값을 꺼냅니다(대괄호 안에 따옴표로 키를 적습니다).
    print("받은 dict의 키:", list(result.keys()))     # 예: ['raw', 'parsed', 'parsing_error']
    print("파싱 결과:", result["parsed"])             # Sentiment 객체
    print("파싱 오류:", result["parsing_error"])       # 성공 시 None
    # result["raw"]는 원본 메시지 객체이고, 그 뒤 .usage_metadata로 토큰 정보를 한 번 더 꺼냅니다.
    print("원본 토큰:", result["raw"].usage_metadata)  # 원본 메시지에서 토큰 사용량 확인
    # 체크포인트: parsed에 객체가, raw에 원본이 동시에 담기면 include_raw를 이해한 것입니다.


def nested_schema(model) -> None:
    """스키마 안에 스키마를 넣어 복잡한 데이터도 구조화한다."""
    # 중첩 스키마: 한 모델이 다른 모델·리스트를 필드로 가질 수 있습니다.
    # 먼저 안쪽에 들어갈 작은 틀(Address)을 정의합니다.
    class Address(BaseModel):
        city: str = Field(description="도시")
        country: str = Field(description="국가")

    # PersonDetail은 위 Address를 필드 타입으로 품습니다(틀 안에 틀).
    class PersonDetail(BaseModel):
        name: str
        # List[str]는 문자열 리스트입니다. default_factory=list는 "값이 없으면 빈 리스트 [ ]를 기본으로 둔다"는 설정입니다.
        # (리스트 같은 가변 기본값은 default= 대신 default_factory=로 두는 것이 파이썬의 안전한 관례입니다.)
        skills: List[str] = Field(default_factory=list, description="보유 기술 목록")
        # 타입 자리에 다른 모델(Address)을 적으면 그 모델이 통째로 한 칸이 됩니다(중첩).
        address: Optional[Address] = Field(default=None, description="거주지, 모르면 비워 둔다")

    structured = model.with_structured_output(PersonDetail)
    print("받고 싶은 형태:", "PersonDetail(name, skills: List[str], address: Address)")
    print("입력 문장:", "김철수는 파이썬과 자바를 다루고 서울(대한민국)에 산다")
    p = structured.invoke("김철수는 파이썬과 자바를 다루고 서울(대한민국)에 산다")
    print("이름:", p.name, "/ 기술:", p.skills, "/ 주소:", p.address)
    # 예: 이름: 김철수 / 기술: ['파이썬', '자바'] / 주소: city='서울' country='대한민국'
    # 체크포인트: skills가 리스트로, address가 중첩 객체로 채워지면 중첩 스키마를 이해한 것입니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    model = init_chat_model(MODEL)

    print("=== include_raw로 원본+파싱 동시 확보 ===")
    include_raw(model)

    print("\n=== 중첩 스키마 ===")
    nested_schema(model)


if __name__ == "__main__":
    main()
