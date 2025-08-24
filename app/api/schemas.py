"""
API 요청/응답 스키마 정의
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class ResumeInputRequest(BaseModel):
    """이력서 정보 입력 요청
    
    구직자의 경력, 직무, 기술 스킬 등 핵심 정보를 텍스트로 입력받습니다.
    AI가 이 정보를 분석하여 맞춤형 면접 질문과 학습 경로를 생성합니다.
    """
    resume_text: str = Field(
        ..., 
        min_length=10, 
        max_length=2000,
        description="이력서 핵심 정보 텍스트 (경력, 직무, 기술 스킬 등)",
        example="3년차 백엔드 개발자, Spring Boot/MSA/Python 기반 커머스 서비스 개발, AWS EC2 운영 경험",
        examples=[
            "3년차 백엔드 개발자, Spring Boot/MSA/Python 기반 커머스 서비스 개발, AWS EC2 운영 경험",
            "신입 프론트엔드 개발자, React/TypeScript/Next.js, 토이 프로젝트 3개 완성",
            "5년차 DevOps 엔지니어, Docker/Kubernetes/AWS, CI/CD 파이프라인 구축 경험"
        ]
    )
    
    @validator('resume_text')
    def validate_resume_text(cls, v):
        if not v.strip():
            raise ValueError('이력서 텍스트는 비어있을 수 없습니다.')
        return v.strip()


class InterviewQuestionResponse(BaseModel):
    """면접 질문 응답"""
    question_text: str = Field(..., description="면접 질문 내용")
    category: str = Field(..., description="질문 카테고리")
    difficulty: str = Field(..., description="난이도")
    reasoning: Optional[str] = Field(None, description="질문 선택 이유")


class ActionItemResponse(BaseModel):
    """실천 방안 응답"""
    step: int = Field(..., description="단계")
    action: str = Field(..., description="실천 방안")
    duration: str = Field(..., description="소요 시간")
    resources: List[str] = Field(default=[], description="필요 리소스")


class LearningResourceResponse(BaseModel):
    """학습 리소스 응답"""
    type: str = Field(..., description="리소스 타입")
    title: str = Field(..., description="리소스 제목")
    url: Optional[str] = Field(None, description="리소스 URL")
    description: str = Field(..., description="리소스 설명")


class LearningPathResponse(BaseModel):
    """학습 경로 응답"""
    title: str = Field(..., description="학습 경로 제목")
    description: str = Field(..., description="학습 경로 설명")
    priority: str = Field(..., description="우선순위")
    estimated_duration: str = Field(..., description="예상 소요 기간")
    difficulty: str = Field(..., description="난이도")
    action_items: List[ActionItemResponse] = Field(..., description="실천 방안 목록")
    recommended_resources: List[LearningResourceResponse] = Field(..., description="추천 리소스")
    expected_outcomes: List[str] = Field(default=[], description="기대 효과")


class ResumeAnalysisResponse(BaseModel):
    """이력서 분석 전체 응답"""
    session_id: str = Field(..., description="세션 ID")
    resume_profile_id: int = Field(..., description="이력서 프로필 ID")
    interview_questions: List[InterviewQuestionResponse] = Field(..., description="면접 질문 목록")
    learning_paths: List[LearningPathResponse] = Field(..., description="학습 경로 목록")
    created_at: datetime = Field(..., description="생성 시간")


class AsyncJobResponse(BaseModel):
    """비동기 작업 응답"""
    job_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="작업 상태")
    message: str = Field(..., description="상태 메시지")
    estimated_completion_time: Optional[int] = Field(None, description="예상 완료 시간(초)")


class JobStatusResponse(BaseModel):
    """작업 상태 조회 응답"""
    job_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="작업 상태")
    progress: Optional[int] = Field(None, description="진행률(%)")
    result: Optional[ResumeAnalysisResponse] = Field(None, description="완료된 결과")
    error_message: Optional[str] = Field(None, description="오류 메시지")
    created_at: datetime = Field(..., description="작업 생성 시간")
    updated_at: Optional[datetime] = Field(None, description="마지막 업데이트 시간")


class ErrorResponse(BaseModel):
    """오류 응답"""
    error: str = Field(..., description="오류 메시지")
    status_code: int = Field(..., description="HTTP 상태 코드")
    details: Optional[Dict[str, Any]] = Field(None, description="상세 오류 정보")
