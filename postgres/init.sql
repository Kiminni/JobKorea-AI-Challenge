-- PostgreSQL 초기화 스크립트
-- 데이터베이스와 사용자는 docker-compose.yml에서 생성됨

-- 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- 인덱스 성능 향상을 위한 설정
SET random_page_cost = 1.1;
SET effective_cache_size = '256MB';

-- 타임존 설정
SET timezone = 'Asia/Seoul';

-- 로그 설정
ALTER SYSTEM SET log_statement = 'mod';
ALTER SYSTEM SET log_min_duration_statement = 1000;

SELECT pg_reload_conf();

