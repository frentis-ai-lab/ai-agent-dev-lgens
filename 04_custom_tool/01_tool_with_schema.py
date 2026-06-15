"""01 - @tool과 Pydantic args_schema로 도구를 정의하고 입구에서 검증한다.

이 예제 하나만으로 다음을 익힙니다.
  - @tool로 일반 파이썬 함수를 LangChain 도구로 바꾼다.
  - 모델이 보는 것은 함수 본문이 아니라 이름·설명·인자 스키마 세 가지뿐임을 확인한다.
  - args_schema(Pydantic BaseModel)로 인자의 타입·기본값·의미를 명세한다.
  - field_validator로 업무 규칙(형식·정규화)을 도구 '입구'에서 강제한다.

이 파일은 자기완결입니다. 다른 파일이나 main에 의존하지 않으며, 아래 한 줄로 단독 실행됩니다.
  uv run python 04_custom_tool/01_tool_with_schema.py

이 예제는 LLM을 호출하지 않으므로 API 키 없이도 처음부터 끝까지 동작합니다
(도구의 정체와 입구 검증은 모델 없이 코드만으로 확인할 수 있습니다).
"""

# from A import B 는 "A라는 모듈에서 B만 골라 가져온다"는 뜻입니다.
# @tool과 ToolException은 langchain_core.tools에 있습니다 (도구 정의의 표준 경로).
from langchain_core.tools import tool, ToolException
# pydantic은 "데이터의 형태(스키마)"를 클래스로 선언하고 검증해 주는 라이브러리입니다.
from pydantic import BaseModel, Field, field_validator


# ===========================================================================
# 1) 입력 스키마 — 인자의 타입·기본값·의미를 한곳에 모은다.
# ===========================================================================

# class는 "데이터의 틀(설계도)"을 만드는 키워드입니다. (BaseModel)은 "BaseModel의 기능을 물려받는다"는 표시입니다.
# args_schema를 함수와 분리해 두면, 검증 규칙을 한곳에 모아 재사용·테스트하기 쉽습니다.
class InventoryInput(BaseModel):
    # "이름: 타입 = Field(...)" 형태로, 필드에 설명·기본값·제약을 붙입니다.
    # Field의 description은 모델에게 전달되는 "이 인자가 무엇인지"에 대한 지시입니다.
    sku: str = Field(min_length=1, description="조회할 제품 코드. 예: 'BAT-21700'")
    # default="ICN"은 "입력에 값이 없으면 기본으로 ICN을 넣는다"는 설정입니다.
    warehouse: str = Field(default="ICN", description="창고 코드. 기본값은 인천(ICN)")

    # @field_validator는 "특정 필드가 들어올 때 이 함수로 검사·가공하라"는 표시입니다.
    # 도구 본문에 닿기 전에 입구에서 실행되어, 형식이 틀리면 여기서 막습니다.
    @field_validator("sku")
    @classmethod
    def sku_정규화(cls, v: str) -> str:
        v = v.strip().upper()                    # 앞뒤 공백 제거 후 대문자로 정규화
        if not v.startswith("BAT-"):             # 업무 규칙: 제품 코드는 BAT-로 시작
            raise ValueError("제품 코드는 'BAT-'로 시작해야 합니다")
        return v                                 # 검사를 통과한 값(정규화된 값)을 돌려줍니다


# 사내 재고를 흉내 낸 데모 데이터입니다 (실제로는 DB·API를 호출하는 자리입니다).
_STOCK = {("BAT-21700", "ICN"): 1240, ("BAT-21700", "PUS"): 380}


# ===========================================================================
# 2) 도구 정의 — @tool로 함수를 도구로 바꾸고 args_schema를 붙인다.
# ===========================================================================

# @tool("이름", args_schema=...)은 일반 함수를 LangChain 도구로 바꿉니다.
# 첫 인자는 도구 이름이고, args_schema로 위에서 만든 입력 스키마를 연결합니다.
@tool("check_inventory", args_schema=InventoryInput)
def check_inventory(sku: str, warehouse: str = "ICN") -> str:
    """지정한 창고의 제품 재고 수량을 조회한다.
    제품 코드(sku)와 창고 코드(warehouse)로 현재 보유 수량을 반환한다.
    재고 수량을 알아야 할 때 사용한다. 예: 'BAT-21700 인천 창고 재고'."""  # docstring=description
    # .get(키)는 사전에서 값을 찾되, 없으면 None을 돌려줍니다.
    qty = _STOCK.get((sku, warehouse))
    if qty is None:
        # 형식은 맞지만 데이터가 없는 경우: ToolException으로 회신해 모델이 인자를 고쳐 재시도하게 합니다.
        raise ToolException(f"재고 정보 없음: sku={sku}, warehouse={warehouse}")
    # 반환값은 모델에게 다시 들어가므로, 사람이 읽기 좋은 간결한 문자열로 돌려줍니다.
    return f"{warehouse} 창고의 {sku} 재고는 {qty:,}개입니다."


# ===========================================================================
# 3) 확인 — 도구의 정체를 보고, 입구 검증이 작동하는지 확인한다.
# ===========================================================================

def show_tool_anatomy() -> None:
    """모델은 함수 본문이 아니라 이름·설명·인자 스키마 세 가지만 본다."""
    # 모델은 이 세 가지(이름·설명·인자 스키마)만 보고 "이 도구를 부를지"를 판단합니다.
    print("[name]       ", check_inventory.name)         # 예: check_inventory
    print("[description]", check_inventory.description)  # docstring 내용 (라우팅 근거가 됩니다)
    print("[args]       ", check_inventory.args)         # {'sku': {...}, 'warehouse': {...}} 스키마
    # 체크포인트: name·description·args가 모두 출력되면, 도구가 모델에 어떻게 보이는지 이해한 것입니다.


def show_entry_validation() -> None:
    """args_schema로 입력 형식을 도구 '입구'에서 강제한다."""
    # (1) 형식이 틀리면 본문 로직에 닿기 전에 입구(스키마)에서 막힙니다.
    try:
        check_inventory.invoke({"sku": "xyz"})           # BAT-로 시작하지 않으므로 차단
        print("[검증] (차단되어야 정상인데 통과했습니다)")
    except Exception as e:
        print("[검증 차단]", e)                          # 예: 제품 코드는 'BAT-'로 시작해야 합니다

    # (2) 정상 입력은 정규화를 거쳐 통과합니다 (소문자·공백이 들어와도 대문자로 정리됩니다).
    print("[정상 호출]", check_inventory.invoke({"sku": " bat-21700 ", "warehouse": "ICN"}))
    # 예: ICN 창고의 BAT-21700 재고는 1,240개입니다.

    # (3) 형식은 맞지만 데이터가 없으면 ToolException으로 사유를 돌려줍니다.
    try:
        check_inventory.invoke({"sku": "BAT-21700", "warehouse": "GWJ"})  # 광주 창고 데이터 없음
    except ToolException as e:
        print("[데이터 없음]", e)                        # 예: 재고 정보 없음: sku=BAT-21700, warehouse=GWJ

    # 체크포인트: 잘못된 sku가 입구에서 막히고, 정상 입력만 통과하며,
    #            없는 데이터는 ToolException으로 회신되면 입구 검증을 이해한 것입니다.


def main() -> None:
    print("=== 도구의 정체 (name·description·args) ===")
    show_tool_anatomy()
    print("\n=== args_schema 입구 검증 ===")
    show_entry_validation()


# 아래 한 줄은 "이 파일을 직접 실행했을 때만 main()을 부른다"는 파이썬의 표준 관용구입니다.
if __name__ == "__main__":
    main()
