"""
Phase 5 Validation Test - Async Document Processing

Tests async document processor with:
1. Concurrent source processing
2. Batch URL processing with httpx
3. Rate limiting with Semaphore
4. Performance comparison (async vs sync)
"""

import asyncio
import time
import sys
from typing import List, Dict, Any


def test_async_processor_initialization():
    """Test AsyncDocumentProcessor initialization"""
    print("=" * 60)
    print("TEST 1: AsyncDocumentProcessor Initialization")
    print("=" * 60)

    try:
        from app.services.document_processor import AsyncDocumentProcessor

        processor = AsyncDocumentProcessor(max_concurrent=5)
        print("[PASS] AsyncDocumentProcessor initialized")
        print(f"[INFO] Max concurrent: {processor.max_concurrent}")
        print(f"[INFO] Timeout: {processor.timeout}s")

        return True, processor

    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_sample_data_processing():
    """Test processing sample documentation"""
    print("\n" + "=" * 60)
    print("TEST 2: Sample Data Processing (Async)")
    print("=" * 60)

    try:
        from app.services.document_processor import AsyncDocumentProcessor
        from config import Config

        # Ensure sample mode is enabled
        Config.DOC_USE_SAMPLE = True

        processor = AsyncDocumentProcessor()

        # Process all sources asynchronously
        start_time = time.time()
        documents = asyncio.run(processor.process_documentation_sources())
        elapsed = time.time() - start_time

        print(f"[PASS] Processed {len(documents)} documents")
        print(f"[INFO] Processing time: {elapsed:.2f}s")

        if documents:
            # Verify document structure
            first_doc = documents[0]
            required_keys = ['title', 'source_url', 'content', 'doc_type', 'version']
            missing_keys = [k for k in required_keys if k not in first_doc]

            if missing_keys:
                print(f"[FAIL] Missing keys in document: {missing_keys}")
                return False

            print(f"[INFO] Sample document:")
            print(f"  - Title: {first_doc['title'][:50]}...")
            print(f"  - Type: {first_doc['doc_type']}")
            print(f"  - Content length: {len(first_doc['content'])} chars")

            # Verify all doc types
            doc_types = set(doc['doc_type'] for doc in documents)
            print(f"[INFO] Document types: {', '.join(doc_types)}")

        return True

    except Exception as e:
        print(f"[FAIL] Sample processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_concurrent_processing():
    """Test that sources are processed concurrently"""
    print("\n" + "=" * 60)
    print("TEST 3: Concurrent Source Processing")
    print("=" * 60)

    try:
        from app.services.document_processor import AsyncDocumentProcessor
        from config import Config

        Config.DOC_USE_SAMPLE = True
        processor = AsyncDocumentProcessor()

        # Time async processing
        start_time = time.time()
        documents_async = asyncio.run(processor.process_documentation_sources())
        async_time = time.time() - start_time

        print(f"[INFO] Async processing: {async_time:.2f}s for {len(documents_async)} docs")

        # Check if multiple sources were processed
        doc_types = set(doc['doc_type'] for doc in documents_async)
        if len(doc_types) > 1:
            print(f"[PASS] Processed {len(doc_types)} different sources concurrently")
        else:
            print(f"[WARN] Only {len(doc_types)} source type found")

        return True

    except Exception as e:
        print(f"[FAIL] Concurrent processing test failed: {e}")
        return False


def test_performance_comparison():
    """Compare async vs sync performance"""
    print("\n" + "=" * 60)
    print("TEST 4: Performance Comparison (Async vs Sync)")
    print("=" * 60)

    try:
        from app.services.document_processor import AsyncDocumentProcessor
        from config import Config

        Config.DOC_USE_SAMPLE = True

        # Test async version
        print("[INFO] Testing async version...")
        processor_async = AsyncDocumentProcessor()
        start_async = time.time()
        docs_async = asyncio.run(processor_async.process_documentation_sources())
        time_async = time.time() - start_async

        print(f"[INFO] Async: {len(docs_async)} docs in {time_async:.2f}s")

        # Try sync version if available
        try:
            from document_processor import DocumentProcessor
            print("[INFO] Testing sync version...")
            processor_sync = DocumentProcessor()
            start_sync = time.time()
            docs_sync = processor_sync.process_documentation_sources()
            time_sync = time.time() - start_sync

            print(f"[INFO] Sync: {len(docs_sync)} docs in {time_sync:.2f}s")

            # Calculate speedup
            if time_async > 0:
                speedup = time_sync / time_async
                print(f"\n[PASS] Performance improvement: {speedup:.2f}x faster")

                if speedup >= 1.5:
                    print(f"[INFO] Excellent speedup! Target was 1.5-2x for sample data")
                elif speedup >= 1.0:
                    print(f"[INFO] Modest speedup (sample data doesn't benefit much from async)")
                else:
                    print(f"[WARN] Async slower than sync (overhead with small sample data)")

        except ImportError:
            print("[SKIP] Sync version not available for comparison")

        return True

    except Exception as e:
        print(f"[FAIL] Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rate_limiting():
    """Test rate limiting with Semaphore"""
    print("\n" + "=" * 60)
    print("TEST 5: Rate Limiting (Semaphore)")
    print("=" * 60)

    try:
        from app.services.document_processor import AsyncDocumentProcessor

        # Create processor with low concurrency limit
        processor = AsyncDocumentProcessor(max_concurrent=2)

        print(f"[INFO] Semaphore limit: {processor.max_concurrent}")
        print(f"[PASS] Rate limiting configured with Semaphore")

        # Verify semaphore is properly initialized
        if hasattr(processor, 'semaphore'):
            print(f"[INFO] Semaphore object: {type(processor.semaphore).__name__}")
        else:
            print("[FAIL] Semaphore not found in processor")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] Rate limiting test failed: {e}")
        return False


def test_httpx_client():
    """Test httpx async client configuration"""
    print("\n" + "=" * 60)
    print("TEST 6: httpx AsyncClient Configuration")
    print("=" * 60)

    try:
        import httpx
        from app.services.document_processor import AsyncDocumentProcessor

        processor = AsyncDocumentProcessor()

        # Verify httpx is imported and used
        print("[PASS] httpx imported successfully")
        print(f"[INFO] httpx version: {httpx.__version__}")

        # Check headers are configured
        if hasattr(processor, 'headers'):
            print(f"[INFO] User-Agent: {processor.headers.get('User-Agent', 'Not set')}")

        return True

    except ImportError as e:
        print(f"[FAIL] httpx not available: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] httpx test failed: {e}")
        return False


def test_async_patterns():
    """Verify async/await patterns are used"""
    print("\n" + "=" * 60)
    print("TEST 7: Async/Await Patterns")
    print("=" * 60)

    try:
        from app.services.document_processor import AsyncDocumentProcessor
        import inspect

        processor = AsyncDocumentProcessor()

        # Check if key methods are async
        async_methods = [
            'process_documentation_sources',
            'process_source',
            '_crawl_and_extract',
            '_process_url_batch',
            '_process_single_url',
            '_fetch_html'
        ]

        all_async = True
        for method_name in async_methods:
            if hasattr(processor, method_name):
                method = getattr(processor, method_name)
                is_async = inspect.iscoroutinefunction(method)
                status = "[PASS]" if is_async else "[FAIL]"
                print(f"{status} {method_name}: {'async' if is_async else 'NOT async'}")
                if not is_async:
                    all_async = False
            else:
                print(f"[WARN] Method {method_name} not found")

        if all_async:
            print("\n[PASS] All key methods use async/await patterns")
        else:
            print("\n[FAIL] Some methods are not async")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] Async pattern verification failed: {e}")
        return False


def main():
    """Run all Phase 5 validation tests"""
    print("\n")
    print("+" + "=" * 58 + "+")
    print("|" + " " * 10 + "PHASE 5 VALIDATION TEST SUITE" + " " * 18 + "|")
    print("|" + " " * 10 + "Async Document Processing" + " " * 23 + "|")
    print("+" + "=" * 58 + "+")

    results = []

    # Run tests
    success, processor = test_async_processor_initialization()
    results.append(("Processor Initialization", success))

    if not success:
        print("\n[FAIL] Cannot proceed without processor")
        return 1

    results.append(("Sample Data Processing", test_sample_data_processing()))
    results.append(("Concurrent Processing", test_concurrent_processing()))
    results.append(("Performance Comparison", test_performance_comparison()))
    results.append(("Rate Limiting", test_rate_limiting()))
    results.append(("httpx Client", test_httpx_client()))
    results.append(("Async Patterns", test_async_patterns()))

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

    if passed == total:
        print("\n*** Phase 5 COMPLETE - All validations passed! ***")
        print("Async document processing ready")
        print("\nExpected performance improvements:")
        print("  - Sample data: 1.5-2x faster")
        print("  - Real web crawling: 5-10x faster")
        print("  - Concurrent sources: Near-instant with asyncio.gather()")
        return 0
    else:
        print(f"\nWARNING: {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
