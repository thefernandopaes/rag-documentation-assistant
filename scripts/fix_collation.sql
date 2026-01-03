-- Fix PostgreSQL collation version mismatch warning
-- Run this in your database query tool or CLI
ALTER DATABASE railway REFRESH COLLATION VERSION;
