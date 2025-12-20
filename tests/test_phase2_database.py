"""
Phase 2 Validation Test - Async Database Layer

Tests:
1. Database connection
2. Async model CRUD operations
3. Session management
4. Schema compatibility
"""

import asyncio
import sys
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError


async def test_database_connection():
    """Test async database connection."""
    print("=" * 60)
    print("TEST 1: Database Connection")
    print("=" * 60)

    try:
        from database_async import check_db_connection, get_async_database_uri

        uri = get_async_database_uri()
        print(f"[INFO] Database URI: {uri[:50]}...")

        is_connected = await check_db_connection()

        if is_connected:
            print("[PASS] Database connection successful")
            return True
        else:
            print("[FAIL] Database connection failed")
            return False

    except Exception as e:
        print(f"[FAIL] Connection test failed: {e}")
        return False


async def test_async_models():
    """Test async model imports and structure."""
    print("\n" + "=" * 60)
    print("TEST 2: Async Models")
    print("=" * 60)

    try:
        from models_async import Conversation, DocumentChunk, RateLimit, Base

        models = [
            ("Conversation", Conversation),
            ("DocumentChunk", DocumentChunk),
            ("RateLimit", RateLimit),
        ]

        for model_name, model_class in models:
            # Check tablename
            tablename = model_class.__tablename__
            print(f"[PASS] {model_name}: table='{tablename}'")

            # Check if has to_dict method
            if hasattr(model_class, 'to_dict'):
                print(f"[PASS] {model_name}: has to_dict() method")

        print("[PASS] All models imported successfully")
        return True

    except Exception as e:
        print(f"[FAIL] Model import failed: {e}")
        return False


async def test_crud_operations():
    """Test CRUD operations with async models."""
    print("\n" + "=" * 60)
    print("TEST 3: CRUD Operations")
    print("=" * 60)

    try:
        from database_async import AsyncSessionLocal, init_async_db
        from models_async import Conversation, DocumentChunk, RateLimit
        import uuid

        # Initialize database (create tables if needed)
        await init_async_db()
        print("[PASS] Database tables initialized")

        async with AsyncSessionLocal() as session:
            # CREATE: Insert a test conversation
            test_conversation = Conversation(
                id=str(uuid.uuid4()),
                session_id=str(uuid.uuid4()),
                user_query="Test query for Phase 2",
                ai_response="Test response from async database",
                sources='[{"url": "https://test.com", "title": "Test"}]',
                response_time=1.23,
                created_at=datetime.utcnow(),
                feedback=None
            )
            session.add(test_conversation)
            await session.commit()
            print(f"[PASS] CREATE: Conversation inserted (id={test_conversation.id[:8]}...)")

            # READ: Query the conversation
            result = await session.execute(
                select(Conversation).where(Conversation.id == test_conversation.id)
            )
            fetched = result.scalar_one_or_none()

            if fetched and fetched.user_query == "Test query for Phase 2":
                print(f"[PASS] READ: Conversation fetched successfully")
            else:
                print("[FAIL] READ: Conversation not found or data mismatch")
                return False

            # UPDATE: Update feedback
            fetched.feedback = 1
            await session.commit()
            print("[PASS] UPDATE: Feedback updated")

            # Verify update
            await session.refresh(fetched)
            if fetched.feedback == 1:
                print("[PASS] UPDATE: Feedback verified")
            else:
                print("[FAIL] UPDATE: Feedback not updated")
                return False

            # COUNT: Test aggregation
            count_result = await session.execute(
                select(func.count(Conversation.id))
            )
            total_count = count_result.scalar()
            print(f"[PASS] COUNT: Total conversations = {total_count}")

            # DELETE: Clean up test data
            await session.delete(fetched)
            await session.commit()
            print("[PASS] DELETE: Test conversation removed")

        return True

    except SQLAlchemyError as e:
        print(f"[FAIL] Database error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] CRUD test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_session_management():
    """Test async session context management."""
    print("\n" + "=" * 60)
    print("TEST 4: Session Management")
    print("=" * 60)

    try:
        from database_async import get_async_db

        # Test async generator
        async for session in get_async_db():
            print("[PASS] Async session created via dependency")

            # Verify session is AsyncSession
            from sqlalchemy.ext.asyncio import AsyncSession
            if isinstance(session, AsyncSession):
                print("[PASS] Session is AsyncSession instance")
            else:
                print("[FAIL] Session is not AsyncSession")
                return False

            # Test query
            from models_async import Conversation
            result = await session.execute(select(Conversation).limit(1))
            print("[PASS] Query executed successfully")

            break  # Exit generator

        return True

    except Exception as e:
        print(f"[FAIL] Session management test failed: {e}")
        return False


async def test_alembic_compatibility():
    """Test compatibility with Alembic migrations."""
    print("\n" + "=" * 60)
    print("TEST 5: Alembic Compatibility")
    print("=" * 60)

    try:
        from models_async import Conversation, DocumentChunk, RateLimit

        # Expected table names (from sync models)
        expected_tables = {
            "Conversation": "conversation",
            "DocumentChunk": "document_chunk",
            "RateLimit": "rate_limit",
        }

        # Verify async models use same table names
        models_to_check = [
            ("Conversation", Conversation, expected_tables["Conversation"]),
            ("DocumentChunk", DocumentChunk, expected_tables["DocumentChunk"]),
            ("RateLimit", RateLimit, expected_tables["RateLimit"]),
        ]

        for model_name, async_model, expected_table in models_to_check:
            actual_table = async_model.__tablename__

            if actual_table == expected_table:
                print(f"[PASS] {model_name}: table name matches ('{actual_table}')")
            else:
                print(f"[FAIL] {model_name}: table mismatch ({actual_table} != {expected_table})")
                return False

        print("[PASS] All table names compatible with existing schema")
        print("[INFO] Alembic migrations will work with async models")
        print("[INFO] Both sync and async models can coexist during migration")
        return True

    except Exception as e:
        print(f"[FAIL] Alembic compatibility check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all Phase 2 validation tests."""
    print("\n")
    print("+" + "=" * 58 + "+")
    print("|" + " " * 10 + "PHASE 2 VALIDATION TEST SUITE" + " " * 18 + "|")
    print("|" + " " * 10 + "Async Database Layer" + " " * 28 + "|")
    print("+" + "=" * 58 + "+")

    results = []

    # Run tests
    results.append(("Database Connection", await test_database_connection()))
    results.append(("Async Models", await test_async_models()))
    results.append(("CRUD Operations", await test_crud_operations()))
    results.append(("Session Management", await test_session_management()))
    results.append(("Alembic Compatibility", await test_alembic_compatibility()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for _, result in results if result)

    for test_name, result in results:
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {test_name}")

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)

    # Cleanup
    try:
        from database_async import dispose_async_db
        await dispose_async_db()
        print("\n[INFO] Database connections closed")
    except Exception as e:
        print(f"\n[WARN] Cleanup warning: {e}")

    if passed == total:
        print("\n*** Phase 2 COMPLETE - All validations passed! ***")
        print("Ready to proceed to Phase 3: Async RAG Engine")
        return 0
    else:
        print(f"\nWARNING: {total - passed} test(s) failed - review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
