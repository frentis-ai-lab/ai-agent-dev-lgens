"""05 - 답을 자유 문장이 아니라 '정해진 객체'로 받는다 (구조화 출력 기본).

이 예제 하나만으로 다음을 익힙니다.
  - 받고 싶은 데이터의 형태를 Pydantic 모델로 먼저 정의한다.
  - with_structured_output으로 모델이 그 형태에 맞춰 답하도록 제약한다.
  - Field(description=...)로 추출 정확도를 높인다.
  - Optional로 '없을 수 있는 값'을 안전하게 처리한다(지어내지 않게).

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 02_langchain_core/05_structured_output.py

키가 없으면 안내만 출력하고 종료합니다.
"""

import os
# typing은 타입을 표현하는 도구 모음입니다. Optional[X]는 "X 타입이거나 비어 있을(None) 수 있다"는 뜻입니다.
from typing import Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# pydantic은 "데이터의 형태(스키마)"를 클래스로 선언하고 검증해 주는 라이브러리입니다.
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


def structured_basic(model) -> None:
    """답을 자유 문장이 아니라 '정해진 객체'로 받는다."""
    # 받고 싶은 데이터의 형태를 Pydantic 모델로 먼저 정의합니다.
    # class는 "데이터의 틀(설계도)"을 만드는 키워드입니다. (BaseModel)은 "BaseModel의 기능을 물려받는다"는 표시입니다.
    # 안쪽의 "이름: 타입" 줄은 "이 틀은 name(문자열 str)과 age(정수 int) 필드를 가진다"는 명세입니다.
    class Person(BaseModel):
        name: str
        age: int

    # with_structured_output으로 모델이 이 형태(스키마)에 맞춰 답하도록 제약합니다.
    # 인자로 클래스 자체(Person)를 넘깁니다. ( )가 없으니 "실행"이 아니라 "그 틀을 가리킴"입니다.
    # 내부적으로는 스키마를 모델에 함께 건네 출력을 그 형태로 강제하고, 응답을 Person으로 파싱합니다.
    structured = model.with_structured_output(Person)
    person = structured.invoke("홍길동은 30살이다")

    # 반환값이 곧 Person 객체이므로 문자열 파싱 없이 필드에 바로 접근합니다(person.name처럼).
    print("이름:", person.name, "/ 나이:", person.age, "/ 나이 타입:", type(person.age).__name__)
    # 예: 이름: 홍길동 / 나이: 30 / 나이 타입: int  (문자열 "30"이 아니라 정수)
    # 체크포인트: age가 정수로 나오면 구조화 출력에 성공한 것입니다.


def field_descriptions(model) -> None:
    """Field에 설명을 달면 추출 정확도가 올라간다."""
    # description은 모델에게 전달되는 지시입니다. 구체적일수록 원하는 값을 정확히 뽑습니다.
    # "필드: 타입 = Field(...)" 형태로, 필드에 추가 설명·기본값을 붙일 수 있습니다.
    # Field(description="...")의 설명 글은 모델에게 "이 칸에는 이런 값을 넣어라"는 안내로 함께 전달됩니다.
    class Product(BaseModel):
        name: str = Field(description="제품의 이름")
        price: int = Field(description="가격, 숫자만(원 단위, 쉼표 없이)")

    structured = model.with_structured_output(Product)
    p = structured.invoke("이 노트북은 1,250,000원입니다")
    print("제품:", p.name, "/ 가격:", p.price)  # 예: 제품: 노트북 / 가격: 1250000
    # 체크포인트: price가 쉼표 없는 정수로 나오면 description 지시가 먹힌 것입니다.


def optional_fields(model) -> None:
    """없을 수 있는 값은 Optional로 두어 모델이 지어내지 않게 한다."""
    # Optional[int]는 "정수이거나 None(값 없음)일 수 있다"는 뜻입니다.
    # default=None은 "입력에 값이 없으면 기본으로 None을 넣어 둔다"는 설정입니다.
    class Person(BaseModel):
        name: str = Field(description="사람의 이름")
        age: Optional[int] = Field(default=None, description="만 나이, 모르면 비워 둔다")

    structured = model.with_structured_output(Person)
    p = structured.invoke("내 이름은 앤디야")  # 나이 정보가 없는 입력
    print("이름:", p.name, "/ 나이:", p.age)  # 예: 이름: 앤디 / 나이: None (없는 값을 지어내지 않음)
    # 체크포인트: 나이가 None으로 나오면 안전하게 동작하는 것입니다.
    #   필수 필드로 두면 모델이 값을 채우려 지어낼 수 있으므로, 빈 수 있는 값은 Optional로 둡니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    model = init_chat_model(MODEL)

    print("=== 구조화 출력 기본 ===")
    structured_basic(model)

    print("\n=== Field 설명으로 정확도 높이기 ===")
    field_descriptions(model)

    print("\n=== Optional로 없는 값 안전 처리 ===")
    optional_fields(model)


if __name__ == "__main__":
    main()
