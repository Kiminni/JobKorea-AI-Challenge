#!/usr/bin/env python3
"""
Celery 워커 테스트 스크립트
"""
import os
import sys
import time
import asyncio
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.workers.celery_app import celery_app, health_check
from app.core.config import settings


def test_celery_connection():
    """Celery 연결 테스트"""
    print("🔌 Celery 연결 테스트 중...")
    
    try:
        # Redis 연결 확인
        redis_client = celery_app.connection()
        redis_client.ensure_connection(max_retries=3)
        print("✅ Redis 연결 성공")
        
        # 워커 상태 확인
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        
        if stats:
            print(f"✅ {len(stats)} 개의 워커가 실행 중")
            for worker_name in stats.keys():
                print(f"   - {worker_name}")
        else:
            print("⚠️  실행 중인 워커가 없습니다")
            
        return True
        
    except Exception as e:
        print(f"❌ Celery 연결 실패: {e}")
        return False


def test_celery_task():
    """Celery 태스크 테스트"""
    print("\n🧪 Celery 태스크 테스트 중...")
    
    try:
        # 헬스 체크 태스크 실행
        task = health_check.delay()
        print(f"✅ 태스크가 큐에 추가됨: {task.id}")
        
        # 결과 대기
        print("⏳ 태스크 완료 대기 중...")
        result = task.get(timeout=10)
        
        print(f"✅ 태스크 완료: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 태스크 실행 실패: {e}")
        return False


def test_resume_processing():
    """이력서 처리 태스크 테스트"""
    print("\n📝 이력서 처리 태스크 테스트 중...")
    
    try:
        from app.workers.celery_app import process_resume_async
        
        # 테스트 이력서 데이터
        test_resume = """
        백엔드 개발자 3년차
        주요 기술: Java, Spring Boot, MySQL, Redis
        경력: 웹 서비스 개발, API 설계, 데이터베이스 최적화
        """
        
        # 태스크 실행
        task = process_resume_async.delay(
            resume_text=test_resume,
            session_id="test-session-123",
            client_ip="127.0.0.1"
        )
        
        print(f"✅ 이력서 처리 태스크가 큐에 추가됨: {task.id}")
        
        # 상태 확인
        print("⏳ 태스크 상태 확인 중...")
        for i in range(10):  # 최대 10번 확인
            if task.ready():
                if task.successful():
                    result = task.result
                    print(f"✅ 이력서 처리 완료: {result}")
                    return True
                else:
                    print(f"❌ 이력서 처리 실패: {task.info}")
                    return False
            
            print(f"   진행률: {i*10}%")
            time.sleep(2)
        
        print("⚠️  태스크가 시간 내에 완료되지 않았습니다")
        return False
        
    except Exception as e:
        print(f"❌ 이력서 처리 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🚀 Celery 워커 테스트 시작\n")
    
    # 환경 변수 설정 확인
    print("📋 환경 변수 확인:")
    print(f"   CELERY_BROKER_URL: {os.getenv('CELERY_BROKER_URL', '설정되지 않음')}")
    print(f"   CELERY_RESULT_BACKEND: {os.getenv('CELERY_RESULT_BACKEND', '설정되지 않음')}")
    print(f"   DATABASE_URL: {os.getenv('DATABASE_URL', '설정되지 않음')}")
    print()
    
    # 테스트 실행
    tests = [
        ("Celery 연결", test_celery_connection),
        ("Celery 태스크", test_celery_task),
        ("이력서 처리", test_resume_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n📊 테스트 결과 요약:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"전체: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트가 통과했습니다!")
        return 0
    else:
        print("⚠️  일부 테스트가 실패했습니다.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
