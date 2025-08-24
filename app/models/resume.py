"""
이력서 관련 데이터베이스 모델
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Boolean, Index
from sqlalchemy.sql import func
from app.db.base import Base


class ResumeProfile(Base):
    """구직자 이력서 프로필"""
    __tablename__ = "resume_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 기본 정보
    career_summary = Column(Text, nullable=False, comment="경력 요약")
    job_functions = Column(Text, nullable=False, comment="수행 직무")
    technical_skills = Column(Text, nullable=False, comment="보유 기술 스킬")
    
    # 원본 입력 텍스트
    original_input = Column(Text, nullable=False, comment="원본 입력 텍스트")
    
    # 파싱된 구조화 데이터
    parsed_data = Column(JSON, comment="파싱된 구조화 데이터")
    
    # 메타데이터
    session_id = Column(String(255), index=True, comment="세션 ID")
    user_ip = Column(String(45), comment="사용자 IP")
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 인덱스
    __table_args__ = (
        Index('idx_resume_session_created', 'session_id', 'created_at'),
        Index('idx_resume_created_at', 'created_at'),
    )


class InterviewQuestion(Base):
    """생성된 면접 질문"""
    __tablename__ = "interview_questions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 연관 정보
    resume_profile_id = Column(Integer, nullable=False, comment="이력서 프로필 ID")
    session_id = Column(String(255), index=True, comment="세션 ID")
    
    # 질문 내용
    question_text = Column(Text, nullable=False, comment="면접 질문 내용")
    question_category = Column(String(100), comment="질문 카테고리 (기술/경험/상황)")
    difficulty_level = Column(String(50), comment="난이도 (초급/중급/고급)")
    
    # AI 생성 관련
    ai_model_used = Column(String(50), comment="사용된 AI 모델")
    prompt_version = Column(String(50), comment="프롬프트 버전")
    generation_metadata = Column(JSON, comment="생성 메타데이터")
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 인덱스
    __table_args__ = (
        Index('idx_interview_resume_profile', 'resume_profile_id'),
        Index('idx_interview_session_created', 'session_id', 'created_at'),
    )


class LearningPath(Base):
    """맞춤형 학습 경로"""
    __tablename__ = "learning_paths"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 연관 정보
    resume_profile_id = Column(Integer, nullable=False, comment="이력서 프로필 ID")
    session_id = Column(String(255), index=True, comment="세션 ID")
    
    # 학습 경로 내용
    path_title = Column(String(255), nullable=False, comment="학습 경로 제목")
    path_description = Column(Text, comment="학습 경로 설명")
    priority_level = Column(String(50), comment="우선순위 (높음/중간/낮음)")
    
    # 구체적 실천 방안
    action_items = Column(JSON, nullable=False, comment="구체적 실천 방안 목록")
    estimated_duration = Column(String(50), comment="예상 소요 기간")
    difficulty_level = Column(String(50), comment="난이도")
    
    # 추천 리소스
    recommended_resources = Column(JSON, comment="추천 학습 리소스")
    
    # AI 생성 관련
    ai_model_used = Column(String(50), comment="사용된 AI 모델")
    prompt_version = Column(String(50), comment="프롬프트 버전")
    generation_metadata = Column(JSON, comment="생성 메타데이터")
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 인덱스
    __table_args__ = (
        Index('idx_learning_resume_profile', 'resume_profile_id'),
        Index('idx_learning_session_created', 'session_id', 'created_at'),
    )


class ApiMetrics(Base):
    """API 사용 메트릭스"""
    __tablename__ = "api_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 요청 정보
    endpoint = Column(String(255), nullable=False, comment="API 엔드포인트")
    method = Column(String(10), nullable=False, comment="HTTP 메서드")
    session_id = Column(String(255), index=True, comment="세션 ID")
    user_ip = Column(String(45), comment="사용자 IP")
    
    # 성능 메트릭
    response_time_ms = Column(Integer, comment="응답 시간 (밀리초)")
    status_code = Column(Integer, comment="HTTP 상태 코드")
    
    # AI 관련 메트릭
    ai_processing_time_ms = Column(Integer, comment="AI 처리 시간 (밀리초)")
    ai_token_usage = Column(JSON, comment="AI 토큰 사용량")
    
    # 에러 정보
    error_message = Column(Text, comment="에러 메시지")
    error_type = Column(String(100), comment="에러 타입")
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 인덱스
    __table_args__ = (
        Index('idx_metrics_endpoint_created', 'endpoint', 'created_at'),
        Index('idx_metrics_session_created', 'session_id', 'created_at'),
        Index('idx_metrics_created_at', 'created_at'),
    )

