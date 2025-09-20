-- PostgreSQL Database Initialization Script
-- Sets up database with proper permissions and indexes

-- Create database (if not exists)
SELECT 'CREATE DATABASE resume_checker'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'resume_checker')\gexec

-- Connect to the database
\c resume_checker;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- For JSON indexing

-- Set timezone
SET timezone = 'UTC';

-- Create indexes after tables are created (handled by SQLAlchemy)
-- Additional performance indexes will be added here

-- Grant permissions to application user (if different from postgres)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- Create full-text search indexes (after table creation)
-- These will be added via migration or manual setup
