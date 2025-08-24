"""
Gemini AI 통합 서비스 - google-genai 안전 모드
- response_schema 미사용(추후 재도입)
- contents=[prompt] 호출
- 멀티파트 안전 추출 + JSON 가드 + 백업 파서
- 제로-컨피그 우선, 문제 키 자동 배제 전략
"""

from typing import List, Dict, Any, Optional
import json
import asyncio
import re
import traceback

from pydantic import BaseModel, Field  # 선택: 파싱 후 검증용
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from app.core.config import settings

from google import genai
from google.genai import types as gat

logger = structlog.get_logger()


# ---------- (선택) Pydantic 모델: 파싱 후 유효성 검사용 ----------
class Question(BaseModel):
    question_text: str
    category: str
    difficulty: str
    good_answer: str

class QuestionsPayload(BaseModel):
    questions: List[Question] = Field(default_factory=list)

class ActionItem(BaseModel):
    step: int
    action: str
    duration: str
    resources: List[str] = Field(default_factory=list)

class ResourceItem(BaseModel):
    type: str
    title: str
    url: Optional[str] = None
    description: Optional[str] = None

class LearningPath(BaseModel):
    title: str
    description: str
    priority: str
    estimated_duration: str
    difficulty: str
    action_items: List[ActionItem] = Field(default_factory=list)
    recommended_resources: List[ResourceItem] = Field(default_factory=list)
    expected_outcomes: List[str] = Field(default_factory=list)

class LearningPathsPayload(BaseModel):
    learning_paths: List[LearningPath] = Field(default_factory=list)


class GeminiService:
    """
    response_schema를 쓰지 않는 안전 모드:
      - config 없이 우선 호출(제로-컨피그)
      - 실패 시 최소 옵션부터 하나씩 추가하며 문제 키 자동 배제
      - 멀티파트/비단순 응답 안전 추출 + JSON 가드 + 백업 파서
    """

    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)

        # 추가 옵션 후보(문제 키가 있으면 자동 제외)
        self._cfg_steps = [
            {"max_output_tokens": 2048},
            {"temperature": 0.7},
            {"top_p": 0.8},
            {"top_k": 40},
            {"response_mime_type": "application/json"},
        ]

    # -------- 공용 호출기: 제로-컨피그 → 점진 옵션 추가 --------
    async def _safe_call(self, prompt: str) -> Any:
        # 1) 제로-컨피그
        try:
            return await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-2.5-flash",
                contents=[prompt],  # 리스트로 고정
            )
        except Exception as e:
            logger.warning("Gemini zero-config 실패, 최소 옵션으로 시도",
                           err_type=type(e).__name__, err=str(e), tb=traceback.format_exc())

        # 2) 최소 옵션부터 하나씩 추가(문제 키 자동 배제)
        cfg_kwargs: Dict[str, Any] = {}
        last_error: Optional[Exception] = None

        for addition in self._cfg_steps:
            cfg_kwargs.update(addition)
            try:
                gen_cfg = gat.GenerateContentConfig(**cfg_kwargs)
            except TypeError as e:
                logger.warning("GenerateContentConfig TypeError → 해당 키 제외",
                               bad_keys=list(addition.keys()), err=str(e))
                for k in addition.keys():
                    cfg_kwargs.pop(k, None)
                last_error = e
                continue

            try:
                resp = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model="gemini-2.5-flash",
                    contents=[prompt],
                    config=gen_cfg,
                )
                logger.info("Gemini 호출 성공(옵션 조합)", cfg_kwargs=cfg_kwargs)
                return resp
            except Exception as e:
                logger.warning("Gemini 호출 실패(옵션 조합) → 해당 키 제외",
                               bad_keys=list(addition.keys()),
                               err_type=type(e).__name__, err=str(e))
                for k in addition.keys():
                    cfg_kwargs.pop(k, None)
                last_error = e

        raise last_error or RuntimeError("Gemini 호출 실패: 모든 config 조합 실패")

    # ---------- Public APIs ----------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_interview_questions(
        self,
        career_summary: str,
        job_functions: str,
        technical_skills: str,
    ) -> List[Dict[str, Any]]:
        prompt = self._create_interview_questions_prompt(career_summary, job_functions, technical_skills)

        resp = await self._safe_call(prompt)

        text = self._extract_response_text(resp)
        logger.info("면접 질문 생성 완료", text_length=len(text), text_preview=text[:200])

        cleaned = self._clean_json_response(text)
        if not isinstance(cleaned, str) or not cleaned or cleaned[0] not in "{[":
            logger.warning("JSON 시작 토큰 없음. 백업 파서 사용",
                           genai_json_ok=0, genai_backup_parse_used=1, sample=cleaned[:160])
            return self._parse_text_response_to_questions(text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("면접 질문 JSON 파싱 실패", error=str(e),
                         response_text_length=len(text),
                         response_text_head=cleaned[:500],
                         response_text_tail=cleaned[-500:],
                         genai_json_ok=0, genai_backup_parse_used=1)
            return self._parse_text_response_to_questions(text)

        # (선택) Pydantic 검증
        try:
            payload = QuestionsPayload.model_validate(data)
            logger.info("면접 질문 파싱 완료",
                        question_count=len(payload.questions),
                        genai_json_ok=1, genai_backup_parse_used=0)
            return [q.model_dump() for q in payload.questions]
        except Exception as e:
            logger.warning("면접 질문 Pydantic 검증 실패 - 원시 JSON 사용",
                           err=str(e), genai_json_ok=1, genai_backup_parse_used=0)
            return data.get("questions", [])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_learning_path(
        self,
        career_summary: str,
        job_functions: str,
        technical_skills: str,
    ) -> List[Dict[str, Any]]:
        prompt = self._create_learning_path_prompt(career_summary, job_functions, technical_skills)

        resp = await self._safe_call(prompt)

        text = self._extract_response_text(resp)
        logger.info("학습 경로 생성 완료", text_length=len(text), text_preview=text[:200])

        cleaned = self._clean_json_response(text)
        if not isinstance(cleaned, str) or not cleaned or cleaned[0] not in "{[":
            logger.warning("JSON 시작 토큰 없음. 백업 파서 사용",
                           genai_json_ok=0, genai_backup_parse_used=1, sample=cleaned[:160])
            return self._parse_text_response_to_learning_paths(text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("학습 경로 JSON 파싱 실패", error=str(e),
                         response_text_length=len(text),
                         response_text_head=cleaned[:500],
                         response_text_tail=cleaned[-500:],
                         genai_json_ok=0, genai_backup_parse_used=1)
            return self._parse_text_response_to_learning_paths(text)

        # (선택) Pydantic 검증
        try:
            payload = LearningPathsPayload.model_validate(data)
            logger.info("학습 경로 파싱 완료",
                        path_count=len(payload.learning_paths),
                        genai_json_ok=1, genai_backup_parse_used=0)
            return [p.model_dump() for p in payload.learning_paths]
        except Exception as e:
            logger.warning("학습 경로 Pydantic 검증 실패 - 원시 JSON 사용",
                           err=str(e), genai_json_ok=1, genai_backup_parse_used=0)
            return data.get("learning_paths", [])

    async def health_check(self) -> Dict[str, Any]:
        """Gemini AI 서비스 헬스 체크"""
        try:
            # 간단한 테스트 프롬프트로 연결 확인
            test_prompt = "Hello, this is a health check. Please respond with 'OK'."
            response = await self._safe_call(test_prompt)
            
            if response and hasattr(response, 'text'):
                return {
                    "status": "healthy",
                    "service": "Gemini AI",
                    "response_preview": response.text[:100],
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            else:
                return {
                    "status": "unhealthy",
                    "service": "Gemini AI",
                    "error": "Invalid response format",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
                
        except Exception as e:
            logger.error("Gemini AI 헬스 체크 실패", error=str(e))
            return {
                "status": "unhealthy",
                "service": "Gemini AI",
                "error": str(e),
                "timestamp": "2024-01-01T00:00:00Z"
            }

    # ---------- Prompt Builders (JSON-only 강제 규칙 포함) ----------
    def _create_interview_questions_prompt(self, career_summary: str, job_functions: str, technical_skills: str) -> str:
        return f"""
당신은 경험이 풍부한 기술 면접관이다. 아래 구직자 정보를 바탕으로 실제 면접에서 사용할 수 있는 **개인 맞춤형 심층 질문 5개**를 생성하라.

[구직자 정보]
- 경력 요약: {career_summary}
- 수행 직무: {job_functions}
- 보유 기술 스킬: {technical_skills}

[개인화 절차 — 내부 수행, 출력 금지]
1) 위 구직자 정보에서 **앵커 용어**를 6~12개 추출한다. (예: Framework/Lib/DB/Infra/Observability/도메인/전략/지표/배포방식)
2) 각 질문의 본문에 **최소 1개 이상의 앵커 용어를 그대로 포함**한다.
3) 다섯 문항 전체에서 **서로 다른 앵커 용어를 최소 4개 이상** 사용해 중복을 피한다.
4) 난이도 배치: 초급 1개(비교형), 중급 2개, 고급 2개(아래 고정 슬롯 포함).

[주제 커버리지 — 고정 슬롯]
- 초급 1개: **비교 선택형**(동일 맥락의 두 기술/방식 비교: 예. REST vs 메시지 큐, Spring Boot vs Django 등).
- 고급 1개: **정합성/트랜잭션/멱등성/재처리/분산 락/Outbox/Saga** 중 하나를 반드시 다룬다.
- 고급 1개: **관찰성(SLO, p95/p99, 오류율, 분산 추적, 로그 상관, 대시보드, 알림 규칙)**을 반드시 다룬다.
- 나머지 2개: 프로젝트경험·협업경험·성장가능성 축에서 **서로 다른 주제**로 채운다.

[질문 품질 규칙]
- **비개인화·상투어 금지**: “최신 기술”, “효율적으로” 같은 뜬말 금지. **조건·지표·상황**을 요구하라(예: p99 300ms, 오류율 0.5%, 롤백 시간 5분 등).
- 실제 면접 구두 질문으로 자연스럽게 말할 수 있는 한국어 문장이며, **반드시 물음표(?)로 끝난다**.
- 특정 회사명/PII/비속어 금지. 너무 추상적인 이론 문제 출제 금지(지원자 경험 기반으로 답할 수 있어야 함).

[출력 형식]
반드시 **유효한 JSON만** 출력한다. 코드블록 금지. 주석/설명 금지. 후행 쉼표 금지.
키 순서 유지. 문자열은 모두 따옴표 사용. 배열은 비어 있지 않게 생성.
형식:
{{
  "questions": [
    {{
      "question_text": "문장 끝이 반드시 물음표(?)로 끝나는 한국어 질문",
      "category": "기술역량 | 문제해결 | 협업경험 | 프로젝트경험 | 성장가능성 중 하나",
      "difficulty": "초급 | 중급 | 고급 중 하나",
      "good_answer": "지원자 경험 맥락이 들어간 1~2문장 예시(키워드 나열 금지, 앵커 용어 1개 이상 포함)"
    }}
  ]
}}

[강제 규칙]
- 총 5문항: **초급 1 · 중급 2 · 고급 2** 정확히 준수.
- 다섯 카테고리(기술역량/문제해결/협업경험/프로젝트경험/성장가능성)를 모두 포함한다(각 1개씩).
- 질문 간 **주제 중복 금지**(서로 다른 역량·상황 평가).
- "category"와 "difficulty"는 허용값만 사용한다.

[자체 검증 체크리스트 — 출력 금지]
- 각 질문에 앵커 용어가 최소 1개 포함되었는가?
- 다섯 문항에서 서로 다른 앵커 용어가 **4개 이상** 쓰였는가?
- 초급 문항이 **비교형**인가?
- 고급 문항에 **정합성 축** 1개, **관찰성 축** 1개가 각각 포함되었는가?
- 카테고리 5종이 **모두 1개씩** 충족되었는가?
- 모든 질문이 물음표로 끝나는가? "good_answer"가 **문장형 예시**로 작성되었는가?

[출력 예시 — 값은 예시일 뿐이며 반드시 위 구직자 정보에 맞게 재생성할 것]
{{
  "questions": [
    {{
      "question_text": "Spring Boot와 Django를 같은 MSA에서 운용할 때 특정 도메인을 각 프레임워크로 분리한 기준과, 팀/성능/배포 체인 관점의 트레이드오프는 무엇이었습니까?",
      "category": "기술역량",
      "difficulty": "초급",
      "good_answer": "도메인 경계·개발 생산성·빌드/배포 파이프라인·관측성 지원성을 근거로 선택했고, Spring Boot는 APM·생태계 강점, Django는 개발 속도·ORM 편의 강점을 사례와 함께 설명한다."
    }}
  ]
}}
"""

    def _create_learning_path_prompt(self, career_summary: str, job_functions: str, technical_skills: str) -> str:
        return f"""
당신은 경험이 풍부한 기술 멘토이다. 아래 구직자 정보를 바탕으로 **개인 맞춤형 학습 경로 3개**를 생성하라.

[구직자 정보]
- 경력 요약: {career_summary}
- 수행 직무: {job_functions}
- 보유 기술 스킬: {technical_skills}

[개인화 절차 — 내부 수행, 출력 금지]
1) 위 구직자 정보에서 **앵커 용어**를 6~12개 추출한다. (예: Framework/Lib/DB/Infra/Observability/도메인/전략/지표/배포방식)
2) 각 학습 경로에 **최소 1개 이상의 앵커 용어를 그대로 포함**한다.
3) 세 경로 전체에서 **서로 다른 앵커 용어를 최소 4개 이상** 사용해 중복을 피한다.
4) 난이도 배치: 초급 1개(기초 강화), 중급 1개(실무 적용), 고급 1개(전문성 향상).

[주제 커버리지 — 고정 슬롯]
- 초급 1개: **기초 역량 강화**(현재 보유 기술의 깊이 있는 이해와 실무 적용).
- 중급 1개: **실무 적용 확장**(새로운 기술 스택과 도구 습득, 프로젝트 경험).
- 고급 1개: **전문성 및 아키텍처**(시스템 설계, 성능 최적화, 운영 자동화).

[학습 경로 품질 규칙]
- **비개인화·상투어 금지**: "최신 기술", "효율적으로" 같은 뜬말 금지. **구체적인 기술명·도구명·방법론**을 요구하라(예: Spring Boot 3.x, Docker Compose, Prometheus + Grafana 등).
- 실제 학습 가능한 구체적인 경로이며, **단계별 액션 아이템**을 포함한다.
- 특정 회사명/PII/비속어 금지. 너무 추상적인 이론 학습 금지(실무 적용 가능한 실습 중심).

[출력 형식]
반드시 **유효한 JSON만** 출력한다. 코드블록 금지. 주석/설명 금지. 후행 쉼표 금지.
키 순서 유지. 문자열은 모두 따옴표 사용. 배열은 비어 있지 않게 생성.
형식:
{{
  "learning_paths": [
    {{
      "title": "학습 경로 제목 (앵커 용어 포함)",
      "description": "왜 이 학습이 필요한지, 어떤 성장을 가져올지 상세 설명",
      "priority": "높음 | 중간 | 낮음",
      "estimated_duration": "구체적인 기간 (예: 2-3개월, 6개월)",
      "difficulty": "초급 | 중급 | 고급",
      "action_items": [
        {{
          "step": 1,
          "action": "구체적인 실천 방안 (앵커 용어 포함)",
          "duration": "소요 시간 (예: 2주, 1개월)",
          "resources": ["필요한 리소스 목록"]
        }}
      ],
      "recommended_resources": [
        {{
          "type": "도서 | 온라인강의 | 프로젝트 | 블로그 | 컨퍼런스 | 공식문서",
          "title": "리소스 제목",
          "url": "URL (선택사항)",
          "description": "리소스 설명"
        }}
      ],
      "expected_outcomes": ["측정 가능한 기대 효과 1", "측정 가능한 기대 효과 2"]
    }}
  ]
}}

[강제 규칙]
- 총 3개 학습 경로: **초급 1 · 중급 1 · 고급 1** 정확히 준수.
- 세 경로는 **서로 다른 기술 영역**을 다룬다(중복 금지).
- 각 경로에 **최소 1개 이상의 앵커 용어**가 포함되어야 함.
- 실천 방안은 **구체적이고 실행 가능**해야 함.
- 추천 리소스는 **실제 존재하는 것들**로 제시.
- 기대 효과는 **측정 가능하고 구체적**이어야 함.

[자체 검증 체크리스트 — 출력 금지]
- 각 학습 경로에 앵커 용어가 최소 1개 포함되었는가?
- 세 경로에서 서로 다른 앵커 용어가 **4개 이상** 쓰였는가?
- 초급 경로가 **기초 역량 강화**에 집중하는가?
- 중급 경로가 **실무 적용 확장**을 다루는가?
- 고급 경로가 **전문성 및 아키텍처**를 포함하는가?
- 각 경로의 실천 방안이 **구체적이고 실행 가능**한가?
- 추천 리소스가 **실제 존재하는 것들**인가?
- 기대 효과가 **측정 가능하고 구체적**인가?

[출력 예시 — 값은 예시일 뿐이며 반드시 위 구직자 정보에 맞게 재생성할 것]
{{
  "learning_paths": [
    {{
      "title": "Spring Boot 3.x 심화 및 실무 적용",
      "description": "현재 보유한 Spring Boot 기초 지식을 바탕으로 최신 버전의 고급 기능과 실무 패턴을 학습하여 프로덕션 환경에서 안정적인 서비스를 구축할 수 있는 역량을 기른다.",
      "priority": "높음",
      "estimated_duration": "3-4개월",
      "difficulty": "중급",
      "action_items": [
        {{
          "step": 1,
          "action": "Spring Boot 3.x 신규 기능 학습 (Native Image, Virtual Threads)",
          "duration": "1개월",
          "resources": ["Spring Boot 3.x 공식 가이드", "Spring Boot 3.x 실전 가이드 도서"]
        }}
      ],
      "recommended_resources": [
        {{
          "type": "도서",
          "title": "Spring Boot 3.x 실전 가이드",
          "description": "Spring Boot 3.x의 신규 기능과 실무 적용 사례"
        }}
      ],
      "expected_outcomes": ["Spring Boot 3.x 고급 기능 활용 능력", "프로덕션 환경 최적화 역량"]
    }}
  ]
}}
"""
    # ---------- Extraction / Cleaning ----------
    def _extract_response_text(self, response) -> str:
        """
        항상 str 반환.
        1) 구조화 출력이면 response.text
        2) 멀티파트: candidates[].content.parts[].text 수집
        3) 최후수단: to_dict() 직렬화
        """
        t = getattr(response, "text", None)
        if isinstance(t, str) and t.strip():
            return t.strip()

        texts: List[str] = []
        for cand in getattr(response, "candidates", []) or []:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                pt = getattr(part, "text", None)
                if isinstance(pt, str) and pt.strip():
                    texts.append(pt.strip())

        if texts:
            return "\n".join(texts)

        if hasattr(response, "to_dict"):
            try:
                return json.dumps(response.to_dict(), ensure_ascii=False)
            except Exception:
                pass

        return ""

    def _clean_json_response(self, s: str) -> str:
        """
        코드블록/잡설 제거 + 첫/마지막 JSON 토큰으로 슬라이스
        """
        if not isinstance(s, str):
            return ""
        # 코드블록 제거
        s = re.sub(r"```json\s*(.*?)\s*```", r"\1", s, flags=re.DOTALL | re.IGNORECASE)
        s = re.sub(r"```\s*(.*?)\s*```", r"\1", s, flags=re.DOTALL)
        # BOM/제로폭 등 정리
        s = s.replace("\ufeff", "").replace("\u200b", "").strip()

        # 첫 { 와 마지막 }를 기준으로 잘라 JSON 본문만 추출
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1].strip()

        return s

    # ---------- Fallback Parsers ----------
    def _parse_text_response_to_questions(self, text: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ("?" in line) or ("질문" in line) or ("Question" in line):
                out.append({
                    "question_text": line,
                    "category": "기술역량",
                    "difficulty": "중급",
                    "good_answer": "자동 파싱"
                })
            if len(out) >= 5:
                break
        return out

    def _parse_text_response_to_learning_paths(self, text: str) -> List[Dict[str, Any]]:
        paths: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        def new_path(title: str) -> Dict[str, Any]:
            return {
                "title": title[:120],
                "description": "자동 파싱된 학습 경로",
                "priority": "중간",
                "estimated_duration": "1-3개월",
                "difficulty": "중급",
                "action_items": [{"step": 1, "action": title[:120], "duration": "1주", "resources": []}],
                "recommended_resources": [],
                "expected_outcomes": []
            }

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if ("학습" in line) or ("경로" in line) or ("Learning" in line) or ("Path" in line):
                if current:
                    paths.append(current)
                current = new_path(line)

        if current:
            paths.append(current)

        return paths[:5]


# Singleton
gemini_service = GeminiService()