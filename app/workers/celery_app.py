"""
Celery 애플리케이션 및 비동기 작업 정의
"""
from celery import Celery
from celery.result import AsyncResult
from typing import Dict, Any
import asyncio
import time
import structlog
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.resume import ResumeProfile, InterviewQuestion, LearningPath
from app.services.gemini_service import gemini_service

logger = structlog.get_logger()

# Celery 애플리케이션 생성
celery_app = Celery(
    "jobkorea_ai_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.celery_app"]
)

# Celery 설정
celery_app.conf.update(
    task_serializer=settings.celery_task_serializer,
    accept_content=settings.celery_accept_content,
    result_serializer=settings.celery_result_serializer,
    timezone=settings.celery_timezone,
    enable_utc=settings.celery_enable_utc,
    task_track_started=settings.celery_task_track_started,
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
    task_acks_late=settings.celery_task_acks_late,
    worker_max_tasks_per_child=settings.celery_worker_max_tasks_per_child,
    task_routes={
        "app.workers.celery_app.process_resume_async": {
            "queue": "resume_processing"
        }
    },
    task_default_queue="default",
    task_default_exchange="default",
    task_default_exchange_type="direct",
    task_default_routing_key="default",
    result_expires=3600,  # 1시간 후 결과 만료
    worker_send_task_events=True,
    task_send_sent_event=True
)

# 비동기 데이터베이스 설정
async_engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_reset_on_return='commit'
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


def parse_resume_text(resume_text: str) -> dict:
    """이력서 텍스트를 구조화된 데이터로 파싱"""
    parsed_data = {
        "career_years": 0,
        "job_roles": [],
        "technical_skills": [],
        "keywords": []
    }
    
    text_lower = resume_text.lower()
    
    # 경력 연차 추출
    if "년차" in text_lower:
        for word in resume_text.split():
            if "년차" in word:
                try:
                    years = int(''.join(filter(str.isdigit, word)))
                    parsed_data["career_years"] = years
                    break
                except ValueError:
                    pass
    
    # 기술 스킬 키워드 추출
    tech_keywords = [
        "python", "java", "javascript", "typescript", "react", "vue", "angular",
        "spring", "django", "flask", "node.js", "express", "nestjs",
        "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "aws", "gcp", "azure", "docker", "kubernetes", "jenkins",
        "git", "github", "gitlab", "jira", "confluence"
    ]
    
    for keyword in tech_keywords:
        if keyword in text_lower:
            parsed_data["technical_skills"].append(keyword)
    
    # 직무 키워드 추출
    job_keywords = [
        "백엔드", "프론트엔드", "풀스택", "데이터", "devops", "ml", "ai",
        "개발자", "엔지니어", "아키텍트", "리드", "시니어", "주니어"
    ]
    
    for keyword in job_keywords:
        if keyword in text_lower:
            parsed_data["job_roles"].append(keyword)
    
    return parsed_data


@celery_app.task(bind=True, name="app.workers.celery_app.process_resume_async")
def process_resume_async(self, resume_text: str, session_id: str, client_ip: str) -> Dict[str, Any]:
    """
    이력서 분석 비동기 처리
    """
    start_time = time.time()
    
    logger.info(
        "비동기 이력서 분석 시작",
        task_id=self.request.id,
        session_id=session_id,
        resume_length=len(resume_text)
    )
    
    try:
        # 진행률 업데이트
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "이력서 분석 시작"})
        
        # 이력서 텍스트 파싱
        parsed_data = parse_resume_text(resume_text)
        
        # 비동기 이벤트 루프 처리 개선
        try:
            # 기존 이벤트 루프가 있는지 확인
            try:
                loop = asyncio.get_running_loop()
                # 이미 실행 중인 루프가 있으면 새로 생성하지 않음
                if loop.is_running():
                    # 동기적으로 실행 (권장하지 않지만 호환성을 위해)
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            _process_resume_with_db(
                                self, 
                                resume_text, 
                                parsed_data, 
                                session_id, 
                                client_ip
                            )
                        )
                        result = future.result(timeout=300)  # 5분 타임아웃
                else:
                    result = asyncio.run(
                        _process_resume_with_db(
                            self, 
                            resume_text, 
                            parsed_data, 
                            session_id, 
                            client_ip
                        )
                    )
            except RuntimeError:
                # 이벤트 루프가 실행 중이지 않은 경우
                result = asyncio.run(
                    _process_resume_with_db(
                        self, 
                        resume_text, 
                        parsed_data, 
                        session_id, 
                        client_ip
                    )
                )
            
            processing_time = time.time() - start_time
            
            logger.info(
                "비동기 이력서 분석 완료",
                task_id=self.request.id,
                session_id=session_id,
                processing_time=processing_time
            )
            
            return result
            
        except asyncio.TimeoutError:
            raise Exception("작업 처리 시간 초과")
        except Exception as e:
            logger.error(
                "비동기 처리 중 오류",
                task_id=self.request.id,
                session_id=session_id,
                error=str(e)
            )
            raise
            
    except Exception as e:
        processing_time = time.time() - start_time
        
        # 예외 타입 정보를 포함한 상세 로깅
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "error_module": type(e).__module__,
            "session_id": session_id,
            "processing_time": processing_time
        }
        
        logger.error(
            "비동기 이력서 분석 실패",
            **error_info
        )
        
        # Celery에 예외 정보 전달 (타입 정보 포함)
        raise type(e)(f"{type(e).__name__}: {str(e)}") from e


async def _process_resume_with_db(
    task, resume_text: str, parsed_data: dict, session_id: str, client_ip: str
) -> Dict[str, Any]:
    """데이터베이스와 함께 이력서 처리"""
    
    async with AsyncSessionLocal() as db:
        try:
            # 이력서 프로필 저장
            resume_profile = ResumeProfile(
                career_summary=resume_text[:500],
                job_functions=", ".join(parsed_data.get("job_roles", [])),
                technical_skills=", ".join(parsed_data.get("technical_skills", [])),
                original_input=resume_text,
                parsed_data=parsed_data,
                session_id=session_id,
                user_ip=client_ip
            )
            
            db.add(resume_profile)
            await db.flush()  # ID 생성을 위해 flush
            
            task.update_state(
                state="PROGRESS",
                meta={"progress": 30, "status": "이력서 프로필 저장 완료"}
            )
            
            # AI 서비스 호출
            ai_start_time = time.time()
            
            task.update_state(
                state="PROGRESS",
                meta={"progress": 40, "status": "면접 질문 생성 중"}
            )
            
            try:
                questions_data = await gemini_service.generate_interview_questions(
                    resume_profile.career_summary,
                    resume_profile.job_functions,
                    resume_profile.technical_skills
                )
            except Exception as e:
                logger.error(
                    "면접 질문 생성 실패",
                    task_id=task.request.id,
                    session_id=session_id,
                    error=str(e)
                )
                raise Exception(f"면접 질문 생성 실패: {str(e)}")
            
            task.update_state(
                state="PROGRESS",
                meta={"progress": 60, "status": "학습 경로 생성 중"}
            )
            
            try:
                paths_data = await gemini_service.generate_learning_path(
                    resume_profile.career_summary,
                    resume_profile.job_functions,
                    resume_profile.technical_skills
                )
            except Exception as e:
                logger.error(
                    "학습 경로 생성 실패",
                    task_id=task.request.id,
                    session_id=session_id,
                    error=str(e)
                )
                raise Exception(f"학습 경로 생성 실패: {str(e)}")
            
            ai_processing_time = time.time() - ai_start_time
            
            task.update_state(
                state="PROGRESS",
                meta={"progress": 80, "status": "결과 저장 중"}
            )
            
            # 면접 질문 저장
            interview_questions = []
            for q_data in questions_data:
                question = InterviewQuestion(
                    resume_profile_id=resume_profile.id,
                    session_id=session_id,
                    question_text=q_data.get("question_text", ""),
                    question_category=q_data.get("category", "기술역량"),
                    difficulty_level=q_data.get("difficulty", "중급"),
                    ai_model_used="gemini-pro",
                    prompt_version="v1.0",
                    generation_metadata=q_data
                )
                db.add(question)
                interview_questions.append({
                    "question_text": q_data.get("question_text", ""),
                    "category": q_data.get("category", "기술역량"),
                    "difficulty": q_data.get("difficulty", "중급"),
                    "reasoning": q_data.get("reasoning", "")
                })
            
            # 학습 경로 저장
            learning_paths = []
            for p_data in paths_data:
                learning_path = LearningPath(
                    resume_profile_id=resume_profile.id,
                    session_id=session_id,
                    path_title=p_data.get("title", ""),
                    path_description=p_data.get("description", ""),
                    priority_level=p_data.get("priority", "중간"),
                    action_items=p_data.get("action_items", []),
                    estimated_duration=p_data.get("estimated_duration", ""),
                    difficulty_level=p_data.get("difficulty", "중급"),
                    recommended_resources=p_data.get("recommended_resources", []),
                    ai_model_used="gemini-pro",
                    prompt_version="v1.0",
                    generation_metadata=p_data
                )
                db.add(learning_path)
                
                # 응답 형식으로 변환
                learning_paths.append({
                    "title": p_data.get("title", ""),
                    "description": p_data.get("description", ""),
                    "priority": p_data.get("priority", "중간"),
                    "estimated_duration": p_data.get("estimated_duration", ""),
                    "difficulty": p_data.get("difficulty", "중급"),
                    "action_items": p_data.get("action_items", []),
                    "recommended_resources": p_data.get("recommended_resources", []),
                    "expected_outcomes": p_data.get("expected_outcomes", [])
                })
            
            await db.commit()
            
            task.update_state(
                state="PROGRESS",
                meta={"progress": 100, "status": "분석 완료"}
            )
            
            processing_time = time.time() - ai_start_time
            
            logger.info(
                "비동기 이력서 분석 완료",
                task_id=task.request.id,
                session_id=session_id,
                resume_profile_id=resume_profile.id,
                processing_time=processing_time,
                ai_processing_time=ai_processing_time
            )
            
            # 결과 반환
            return {
                "session_id": session_id,
                "resume_profile_id": resume_profile.id,
                "interview_questions": interview_questions,
                "learning_paths": learning_paths,
                "created_at": resume_profile.created_at.isoformat(),
                "processing_time": processing_time,
                "ai_processing_time": ai_processing_time
            }
            
        except Exception as e:
            await db.rollback()
            logger.error(
                "데이터베이스 처리 중 오류",
                task_id=task.request.id,
                session_id=session_id,
                error=str(e)
            )
            # 작업 상태를 실패로 업데이트
            task.update_state(
                state="FAILURE",
                meta={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "progress": 0
                }
            )
            raise


@celery_app.task(bind=True, name="app.workers.celery_app.health_check")
def health_check(self):
    """워커 헬스 체크"""
    return {
        "status": "healthy",
        "worker_id": self.request.id,
        "timestamp": time.time()
    }


# 작업 상태 조회 함수
def get_task_result(task_id: str) -> Dict[str, Any]:
    """작업 결과 조회"""
    result = AsyncResult(task_id, app=celery_app)
    
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result,
        "traceback": result.traceback
    }


# Celery 시그널 핸들러
from celery.signals import task_prerun, task_postrun, task_failure

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """작업 시작 전 로깅"""
    logger.info(
        "Celery 작업 시작",
        task_id=task_id,
        task_name=task.name if task else "unknown"
    )


@task_postrun.connect
def task_postrun_handler(
    sender=None, task_id=None, task=None, args=None, kwargs=None, 
    retval=None, state=None, **kwds
):
    """작업 완료 후 로깅"""
    logger.info(
        "Celery 작업 완료",
        task_id=task_id,
        task_name=task.name if task else "unknown",
        state=state
    )


@task_failure.connect
def task_failure_handler(
    sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds
):
    """작업 실패 시 로깅"""
    logger.error(
        "Celery 작업 실패",
        task_id=task_id,
        exception=str(exception),
        traceback=traceback
    )
