"""테스트 공통 설정.

실제 모델 호출 없이 'API 계약(import 경로·객체 구성·그래프 컴파일·메모리 동작)'만 검증합니다.
모델 객체 구성 단계에서 키 부재로 막히지 않도록 더미 키를 넣습니다(실제 호출은 하지 않음).
실제 LLM 호출이 필요한 검증은 강의 직전에 키를 넣고 노트북을 1회 실행해 확인합니다.
"""
import os

# 실제 호출이 아니라 객체 구성만 하므로 더미 키로 충분합니다.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-not-used")
os.environ.setdefault("GOOGLE_API_KEY", "test-dummy-not-used")
