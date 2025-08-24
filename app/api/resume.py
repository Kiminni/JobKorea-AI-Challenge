"""
이력서 관련 API 엔드포인트
"""
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json
import uuid
import time
from datetime import datetime
from typing import Optional
import structlog

from app.db.base import get_db
from app.models.resume import ResumeProfile, InterviewQuestion, LearningPath, ApiMetrics
from app.services.gemini_service import gemini_service
from app.api.schemas import (
    ResumeInputRequest, 
    ResumeAnalysisResponse, 
    AsyncJobResponse,
    JobStatusResponse,
    InterviewQuestionResponse,
    LearningPathResponse,
    ActionItemResponse,
    LearningResourceResponse
)
from app.workers.celery_app import process_resume_async
from app.core.config import settings

logger = structlog.get_logger()
router = APIRouter()


def get_session_id(request: Request) -> str:
    """요청에서 세션 ID 추출"""
    return request.headers.get("X-Session-ID", str(uuid.uuid4()))


def get_client_ip(request: Request) -> str:
    """클라이언트 IP 추출"""
    forwarded_ip = request.headers.get("X-Forwarded-For")
    if forwarded_ip:
        return forwarded_ip.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def log_api_metrics(
    db: AsyncSession,
    endpoint: str,
    method: str,
    session_id: str,
    user_ip: str,
    response_time_ms: int,
    status_code: int,
    ai_processing_time_ms: Optional[int] = None,
    ai_token_usage: Optional[dict] = None,
    error_message: Optional[str] = None,
    error_type: Optional[str] = None
):
    """API 메트릭스 로깅"""
    try:
        metrics = ApiMetrics(
            endpoint=endpoint,
            method=method,
            session_id=session_id,
            user_ip=user_ip,
            response_time_ms=response_time_ms,
            status_code=status_code,
            ai_processing_time_ms=ai_processing_time_ms,
            ai_token_usage=ai_token_usage,
            error_message=error_message,
            error_type=error_type
        )
        db.add(metrics)
        await db.commit()
    except Exception as e:
        logger.error("메트릭스 로깅 실패", error=str(e))


def parse_resume_text(resume_text: str) -> dict:
    """이력서 텍스트를 구조화된 데이터로 파싱"""
    # 간단한 키워드 기반 파싱
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
    
    # 기술 스킬 추출 (일반적인 개발 기술들)
    tech_keywords = [
        "spring", "boot", "java", "python", "javascript", "react", "vue", "angular",
        "node.js", "express", "django", "flask", "fastapi", "aws", "azure", "gcp",
        "docker", "kubernetes", "jenkins", "git", "mysql", "postgresql", "mongodb",
        "redis", "elasticsearch", "kafka", "rabbitmq", "msa", "microservices"
    ]
    
    for tech in tech_keywords:
        if tech in text_lower:
            parsed_data["technical_skills"].append(tech)
    
    # 직무 역할 추출
    job_keywords = ["백엔드", "프론트엔드", "풀스택", "devops", "데이터", "ai", "ml"]
    for job in job_keywords:
        if job in resume_text:
            parsed_data["job_roles"].append(job)
    
    # 일반 키워드 추출
    words = resume_text.split()
    for word in words:
        if len(word) > 2 and word not in parsed_data["technical_skills"]:
            parsed_data["keywords"].append(word)
    
    return parsed_data


@router.post(
    "/analyze", 
    response_model=ResumeAnalysisResponse,
    summary="이력서 분석",
    description="""
    이력서 정보를 분석하여 맞춤형 면접 질문과 학습 경로를 생성합니다.
    
    Gemini AI를 직접 호출하여 즉시 결과를 반환합니다.
    
    **처리 과정**:
    1. 이력서 텍스트 분석 및 구조화
    2. AI를 통한 면접 질문 생성 (5개)
    3. 맞춤형 학습 경로 및 실천 방안 생성
    4. 결과 데이터베이스 저장
    
    **응답 시간**: 일반적으로 10-30초 (AI 처리 시간에 따라 다름)
    """,
    response_description="이력서 분석 결과 (면접 질문 5개 + 학습 경로)",
    responses={
        200: {"description": "분석 완료"},
        400: {"description": "잘못된 요청 (이력서 텍스트 검증 실패)"},
        422: {"description": "요청 데이터 검증 실패"},
        500: {"description": "서버 내부 오류"}
    }
)
async def analyze_resume_sync(
    request: Request,
    resume_request: ResumeInputRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    start_time = time.time()
    session_id = get_session_id(request)
    client_ip = get_client_ip(request)
    
    logger.info(
        "이력서 분석 요청",
        session_id=session_id,
        resume_length=len(resume_request.resume_text)
    )
    
    try:
        # 이력서 텍스트 파싱
        parsed_data = parse_resume_text(resume_request.resume_text)
        
        # 이력서 프로필 저장
        resume_profile = ResumeProfile(
            career_summary=resume_request.resume_text[:500],
            job_functions=", ".join(parsed_data.get("job_roles", [])),
            technical_skills=", ".join(parsed_data.get("technical_skills", [])),
            original_input=resume_request.resume_text,
            parsed_data=parsed_data,
            session_id=session_id,
            user_ip=client_ip
        )
        
        db.add(resume_profile)
        await db.flush()  # ID 생성을 위해 flush
        
        ai_start_time = time.time()
        
        # AI 서비스 호출 (병렬 처리)
        import asyncio
        questions_task = gemini_service.generate_interview_questions(
            resume_profile.career_summary,
            resume_profile.job_functions,
            resume_profile.technical_skills
        )
        
        paths_task = gemini_service.generate_learning_path(
            resume_profile.career_summary,
            resume_profile.job_functions,
            resume_profile.technical_skills
        )
        
        questions_data, paths_data = await asyncio.gather(
            questions_task, paths_task
        )
        
        ai_processing_time = int((time.time() - ai_start_time) * 1000)
        
        # 면접 질문 저장
        interview_questions = []
        for q_data in questions_data:
            question = InterviewQuestion(
                resume_profile_id=resume_profile.id,
                session_id=session_id,
                question_text=q_data.get("question_text", ""),
                question_category=q_data.get("category", "기술역량"),
                difficulty_level=q_data.get("difficulty", "중급"),
                ai_model_used="gemini-2.5-flash",
                prompt_version="v1.0",
                generation_metadata=q_data
            )
            db.add(question)
            interview_questions.append(InterviewQuestionResponse(
                question_text=q_data.get("question_text", ""),
                category=q_data.get("category", "기술역량"),
                difficulty=q_data.get("difficulty", "중급"),
                reasoning=q_data.get("reasoning", "")
            ))
        
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
                ai_model_used="gemini-2.5-flash",
                prompt_version="v1.0",
                generation_metadata=p_data
            )
            db.add(learning_path)
            
            # Action Items 변환
            action_items = []
            for item in p_data.get("action_items", []):
                action_items.append(ActionItemResponse(
                    step=item.get("step", 1),
                    action=item.get("action", ""),
                    duration=item.get("duration", ""),
                    resources=item.get("resources", [])
                ))
            
            # Recommended Resources 변환
            resources = []
            for resource in p_data.get("recommended_resources", []):
                resources.append(LearningResourceResponse(
                    type=resource.get("type", ""),
                    title=resource.get("title", ""),
                    url=resource.get("url"),
                    description=resource.get("description", "")
                ))
            
            learning_paths.append(LearningPathResponse(
                title=p_data.get("title", ""),
                description=p_data.get("description", ""),
                priority=p_data.get("priority", "중간"),
                estimated_duration=p_data.get("estimated_duration", ""),
                difficulty=p_data.get("difficulty", "중급"),
                action_items=action_items,
                recommended_resources=resources,
                expected_outcomes=p_data.get("expected_outcomes", [])
            ))
        
        # 데이터베이스에 저장
        await db.commit()
        
        response_time = int((time.time() - start_time) * 1000)
        
        # 성공 메트릭스 로깅
        background_tasks.add_task(
            log_api_metrics,
            db,
            "/api/v1/resume/analyze",
            "POST",
            session_id,
            client_ip,
            response_time,
            200,
            ai_processing_time
        )
        
        logger.info(
            "이력서 분석 완료  ",
            session_id=session_id,
            resume_id=resume_profile.id,
            questions_count=len(interview_questions),
            paths_count=len(learning_paths),
            ai_processing_time=ai_processing_time
        )
        
        return ResumeAnalysisResponse(
            session_id=session_id,
            resume_profile_id=resume_profile.id,
            interview_questions=interview_questions,
            learning_paths=learning_paths,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        response_time = int((time.time() - start_time) * 1000)
        
        # 오류 메트릭스 로깅
        background_tasks.add_task(
            log_api_metrics,
            db,
            "/api/v1/resume/analyze",
            "POST",
            session_id,
            client_ip,
            response_time,
            500,
            error_message=str(e),
            error_type=type(e).__name__
        )
        
        logger.error(
            "이력서 분석 실패",
            error=str(e),
            error_type=type(e).__name__,
            session_id=session_id
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"이력서 분석 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/analyze-async", 
    response_model=AsyncJobResponse,
    summary="이력서 분석",
    description="""
    이력서 정보를 분석하여 맞춤형 면접 질문과 학습 경로를 생성합니다.
    
    **Async-lane**: Celery 워커를 통해 백그라운드에서 처리합니다.
    
    **처리 과정**:
    1. 이력서 분석 작업을 Celery 큐에 추가
    2. 즉시 job_id 반환
    3. 백그라운드에서 AI 분석 수행
    4. `/job-status/{job_id}`로 결과 확인
    
    **장점**: 
    - 긴 대기 시간 없음
    - 대용량 처리 가능
    - 실시간 상태 확인
    
    **사용법**:
    1. 이 엔드포인트로 요청 전송
    2. job_id 받기
    3. 상태 조회 엔드포인트로 결과 확인
    """,
    response_description="비동기 작업 ID 및 상태 정보",
    tags=["resume"],
    responses={
        202: {"description": "작업이 큐에 추가됨"},
        400: {"description": "잘못된 요청"},
        422: {"description": "요청 데이터 검증 실패"},
        500: {"description": "서버 내부 오류"}
    }
)
async def analyze_resume_async(
    request: Request,
    resume_request: ResumeInputRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    start_time = time.time()
    session_id = get_session_id(request)
    client_ip = get_client_ip(request)
    
    logger.info(
        "이력서 분석 요청",
        session_id=session_id,
        resume_length=len(resume_request.resume_text)
    )
    
    try:
        # Celery 작업 큐에 추가
        task = process_resume_async.delay(
            resume_text=resume_request.resume_text,
            session_id=session_id,
            client_ip=client_ip
        )
        
        response_time = int((time.time() - start_time) * 1000)
        
        # 메트릭스 로깅
        background_tasks.add_task(
            log_api_metrics,
            db,
            "/api/v1/resume/analyze-async",
            "POST",
            session_id,
            client_ip,
            response_time,
            202
        )
        
        logger.info(
            "비동기 작업 큐에 추가됨",
            session_id=session_id,
            job_id=task.id
        )
        
        return AsyncJobResponse(
            job_id=task.id,
            status="PENDING",
            message="이력서 분석 작업이 큐에 추가되었습니다.",
            estimated_completion_time=30
        )
        
    except Exception as e:
        response_time = int((time.time() - start_time) * 1000)
        
        # 오류 메트릭스 로깅
        background_tasks.add_task(
            log_api_metrics,
            db,
            "/api/v1/resume/analyze-async",
            "POST",
            session_id,
            client_ip,
            response_time,
            500,
            error_message=str(e),
            error_type=type(e).__name__
        )
        
        logger.error(
            "비동기 작업 큐 추가 실패",
            session_id=session_id,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"비동기 작업 큐 추가 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/resume/job-status/{job_id}",
    response_model=JobStatusResponse,
    summary="비동기 작업 상태 조회",
    description="""
    비동기 이력서 분석 작업의 현재 상태를 확인합니다.
    
    **상태 값**:
    - `PENDING`: 작업이 큐에 대기 중
    - `PROCESSING`: 작업 처리 중
    - `COMPLETED`: 작업 완료
    - `FAILED`: 작업 실패
    
    **응답 정보**:
    - 진행률 (0-100)
    - 결과 데이터 (완료 시)
    - 오류 메시지 (실패 시)
    """,
    response_description="작업 상태 및 결과 정보",
    tags=["resume"],
    responses={
        200: {"description": "작업 상태 조회 성공"},
        404: {"description": "작업을 찾을 수 없음"},
        500: {"description": "서버 내부 오류"}
    }
)
async def get_job_status(
    job_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    start_time = time.time()
    session_id = get_session_id(request)
    client_ip = get_client_ip(request)
    
    logger.info(
        "작업 상태 조회 요청",
        session_id=session_id,
        job_id=job_id
    )
    
    try:
        # Celery 작업 상태 확인
        from celery.result import AsyncResult
        task_result = AsyncResult(job_id)
        
        response_time = int((time.time() - start_time) * 1000)
        
        # 메트릭스 로깅
        background_tasks.add_task(
            log_api_metrics,
            db,
            f"/api/v1/resume/job-status/{job_id}",
            "GET",
            session_id,
            client_ip,
            response_time,
            200
        )
        
        if task_result.ready():
            if task_result.successful():
                result_data = task_result.result
                return JobStatusResponse(
                    job_id=job_id,
                    status="COMPLETED",
                    progress=100,
                    result=result_data,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
            else:
                # 실패한 작업의 경우 오류 정보 추출
                error_info = task_result.info if hasattr(task_result, 'info') else {}
                error_message = str(error_info) if error_info else "작업 처리 중 오류가 발생했습니다"
                
                return JobStatusResponse(
                    job_id=job_id,
                    status="FAILED",
                    error_message=error_message,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
        else:
            # 진행 중인 작업의 경우 진행률 추정
            progress = 0
            status_message = "작업 처리 중"
            
            if hasattr(task_result, 'info') and isinstance(task_result.info, dict):
                progress = task_result.info.get('progress', 0)
                status_message = task_result.info.get('status', '작업 처리 중')
            
            return JobStatusResponse(
                job_id=job_id,
                status="PROCESSING",
                progress=progress,
                message=status_message,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
    except Exception as e:
        response_time = int((time.time() - start_time) * 1000)
        
        # 오류 메트릭스 로깅
        background_tasks.add_task(
            log_api_metrics,
            db,
            f"/api/v1/resume/job-status/{job_id}",
            "GET",
            session_id,
            client_ip,
            response_time,
            500,
            error_message=str(e),
            error_type=type(e).__name__
        )
        
        logger.error(
            "작업 상태 조회 실패",
            session_id=session_id,
            job_id=job_id,
            error=str(e),
            error_type=type(e).__name__
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"작업 상태 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/health",
    summary="서비스 헬스 체크",
    description="이력서 분석 서비스의 전반적인 상태를 확인합니다.",
    tags=["resume"]
)
async def health_check():
    """서비스 헬스 체크"""
    try:
        # Celery 워커 상태 확인
        from app.workers.celery_app import celery_app
        
        # Redis 연결 상태 확인
        redis_client = celery_app.connection()
        redis_client.ensure_connection(max_retries=3)
        
        # 워커 상태 확인
        active_workers = celery_app.control.inspect().active()
        registered_tasks = celery_app.control.inspect().registered()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "celery": {
                "broker_connected": True,
                "active_workers": len(active_workers) if active_workers else 0,
                "registered_tasks": len(registered_tasks) if registered_tasks else 0
            },
            "services": {
                "database": "connected",
                "redis": "connected",
                "celery_worker": "running"
            }
        }
    except Exception as e:
        logger.error("헬스 체크 실패", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "services": {
                "database": "unknown",
                "redis": "unknown",
                "celery_worker": "unknown"
            }
        }


@router.get(
    "/workers/status",
    summary="Celery 워커 상태 조회",
    description="현재 실행 중인 Celery 워커들의 상태를 확인합니다.",
    tags=["resume"]
)
async def get_workers_status():
    """Celery 워커 상태 조회"""
    try:
        from app.workers.celery_app import celery_app
        
        # 워커 정보 수집
        inspect = celery_app.control.inspect()
        
        active_workers = inspect.active()
        registered_tasks = inspect.registered()
        stats = inspect.stats()
        
        workers_info = []
        if stats:
            for worker_name, worker_stats in stats.items():
                worker_info = {
                    "name": worker_name,
                    "status": "online",
                    "active_tasks": len(active_workers.get(worker_name, [])) if active_workers else 0,
                    "registered_tasks": len(registered_tasks.get(worker_name, [])) if registered_tasks else 0,
                    "stats": worker_stats
                }
                workers_info.append(worker_info)
        
        return {
            "status": "success",
            "workers_count": len(workers_info),
            "workers": workers_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("워커 상태 조회 실패", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"워커 상태 조회 중 오류가 발생했습니다: {str(e)}"
        )
