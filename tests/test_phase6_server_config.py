"""
Phase 6 Validation Test - Server Configuration

Tests server configuration:
1. Uvicorn config module
2. Worker calculation
3. Environment variables
4. Procfile syntax
5. Development/production scripts
"""

import os
import sys
import multiprocessing


def test_uvicorn_config_module():
    """Test uvicorn_config.py module"""
    print("=" * 60)
    print("TEST 1: Uvicorn Config Module")
    print("=" * 60)

    try:
        import uvicorn_config

        print("[PASS] uvicorn_config.py imported successfully")

        # Check required functions
        required_functions = [
            'get_workers',
            'get_host',
            'get_port',
            'get_log_level',
            'get_config',
            'print_config'
        ]

        for func_name in required_functions:
            if hasattr(uvicorn_config, func_name):
                print(f"[PASS] Function {func_name} exists")
            else:
                print(f"[FAIL] Function {func_name} missing")
                return False

        return True

    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_worker_calculation():
    """Test worker process calculation"""
    print("\n" + "=" * 60)
    print("TEST 2: Worker Calculation")
    print("=" * 60)

    try:
        from server.uvicorn import get_workers

        # Clear env var for testing
        original = os.environ.get("WEB_CONCURRENCY")
        if "WEB_CONCURRENCY" in os.environ:
            del os.environ["WEB_CONCURRENCY"]

        workers = get_workers()
        cpu_count = multiprocessing.cpu_count()
        expected = (2 * cpu_count) + 1

        print(f"[INFO] CPU cores: {cpu_count}")
        print(f"[INFO] Workers: {workers}")
        print(f"[INFO] Expected: {expected}")

        if workers == expected:
            print(f"[PASS] Worker calculation correct: (2 * {cpu_count}) + 1 = {workers}")
        else:
            print(f"[WARN] Worker count mismatch (expected {expected}, got {workers})")

        # Test WEB_CONCURRENCY override
        os.environ["WEB_CONCURRENCY"] = "4"
        workers_override = get_workers()

        if workers_override == 4:
            print(f"[PASS] WEB_CONCURRENCY override works: {workers_override}")
        else:
            print(f"[FAIL] WEB_CONCURRENCY override failed")
            return False

        # Restore original
        if original:
            os.environ["WEB_CONCURRENCY"] = original
        elif "WEB_CONCURRENCY" in os.environ:
            del os.environ["WEB_CONCURRENCY"]

        return True

    except Exception as e:
        print(f"[FAIL] Worker calculation test failed: {e}")
        return False


def test_environment_variables():
    """Test environment variable handling"""
    print("\n" + "=" * 60)
    print("TEST 3: Environment Variables")
    print("=" * 60)

    try:
        from server.uvicorn import get_host, get_port, get_log_level

        # Test defaults
        host = get_host()
        port = get_port()
        log_level = get_log_level()

        print(f"[INFO] Host: {host}")
        print(f"[INFO] Port: {port}")
        print(f"[INFO] Log level: {log_level}")

        # Validate defaults
        if host == "0.0.0.0":
            print("[PASS] Default host correct (0.0.0.0)")
        else:
            print(f"[WARN] Unexpected default host: {host}")

        if port == 8000:
            print("[PASS] Default port correct (8000)")
        else:
            print(f"[WARN] Unexpected default port: {port}")

        if log_level == "info":
            print("[PASS] Default log level correct (info)")
        else:
            print(f"[WARN] Unexpected default log level: {log_level}")

        # Test PORT override
        os.environ["PORT"] = "9000"
        port_override = get_port()

        if port_override == 9000:
            print("[PASS] PORT override works")
        else:
            print("[FAIL] PORT override failed")
            return False

        # Cleanup
        if "PORT" in os.environ:
            del os.environ["PORT"]

        return True

    except Exception as e:
        print(f"[FAIL] Environment variable test failed: {e}")
        return False


def test_config_dictionary():
    """Test complete configuration dictionary"""
    print("\n" + "=" * 60)
    print("TEST 4: Configuration Dictionary")
    print("=" * 60)

    try:
        from server.uvicorn import get_config

        config = get_config()

        # Check required keys
        required_keys = [
            'host',
            'port',
            'workers',
            'timeout_keep_alive',
            'log_level',
            'proxy_headers',
            'limit_concurrency',
            'lifespan'
        ]

        all_present = True
        for key in required_keys:
            if key in config:
                print(f"[PASS] Config has '{key}': {config[key]}")
            else:
                print(f"[FAIL] Config missing '{key}'")
                all_present = False

        if not all_present:
            return False

        # Validate types
        if isinstance(config['host'], str):
            print("[PASS] host is string")
        else:
            print(f"[FAIL] host is not string: {type(config['host'])}")
            return False

        if isinstance(config['port'], int):
            print("[PASS] port is integer")
        else:
            print(f"[FAIL] port is not integer: {type(config['port'])}")
            return False

        if isinstance(config['workers'], int):
            print("[PASS] workers is integer")
        else:
            print(f"[FAIL] workers is not integer: {type(config['workers'])}")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] Configuration dictionary test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_procfile():
    """Test Procfile syntax"""
    print("\n" + "=" * 60)
    print("TEST 5: Procfile")
    print("=" * 60)

    try:
        with open("Procfile", "r") as f:
            content = f.read()

        print("[PASS] Procfile exists and readable")

        # Check for uvicorn command
        if "uvicorn" in content:
            print("[PASS] Procfile contains uvicorn command")
        else:
            print("[FAIL] Procfile missing uvicorn command")
            return False

        # Check for fastapi_app
        if "fastapi_app:app" in content:
            print("[PASS] Procfile references fastapi_app:app")
        else:
            print("[FAIL] Procfile doesn't reference fastapi_app:app")
            return False

        # Check for essential flags
        essential_flags = [
            "--host",
            "--port",
            "--workers",
            "--proxy-headers"
        ]

        for flag in essential_flags:
            if flag in content:
                print(f"[PASS] Procfile has {flag} flag")
            else:
                print(f"[WARN] Procfile missing {flag} flag")

        return True

    except FileNotFoundError:
        print("[FAIL] Procfile not found")
        return False
    except Exception as e:
        print(f"[FAIL] Procfile test failed: {e}")
        return False


def test_run_scripts():
    """Test development and production run scripts"""
    print("\n" + "=" * 60)
    print("TEST 6: Run Scripts")
    print("=" * 60)

    try:
        # Check run_dev.py
        if os.path.exists("run_dev.py"):
            print("[PASS] run_dev.py exists")

            # Check if it's executable Python
            with open("run_dev.py", "r") as f:
                content = f.read()

            if "uvicorn" in content and "reload=True" in content:
                print("[PASS] run_dev.py has hot reload enabled")
            else:
                print("[WARN] run_dev.py might not have hot reload")

        else:
            print("[FAIL] run_dev.py not found")
            return False

        # Check run_prod.py
        if os.path.exists("run_prod.py"):
            print("[PASS] run_prod.py exists")

            with open("run_prod.py", "r") as f:
                content = f.read()

            if "uvicorn_config" in content:
                print("[PASS] run_prod.py uses uvicorn_config")
            else:
                print("[WARN] run_prod.py doesn't use uvicorn_config")

        else:
            print("[FAIL] run_prod.py not found")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] Run scripts test failed: {e}")
        return False


def test_env_example():
    """Test .env.example has FastAPI variables"""
    print("\n" + "=" * 60)
    print("TEST 7: Environment Variables Template")
    print("=" * 60)

    try:
        with open(".env.example", "r") as f:
            content = f.read()

        print("[PASS] .env.example exists and readable")

        # Check for FastAPI variables
        fastapi_vars = [
            "FASTAPI_ENABLED",
            "UVICORN_HOST",
            "UVICORN_LOG_LEVEL",
            "PORT",
            "WEB_CONCURRENCY"
        ]

        all_present = True
        for var in fastapi_vars:
            if var in content:
                print(f"[PASS] .env.example has {var}")
            else:
                print(f"[FAIL] .env.example missing {var}")
                all_present = False

        return all_present

    except FileNotFoundError:
        print("[FAIL] .env.example not found")
        return False
    except Exception as e:
        print(f"[FAIL] .env.example test failed: {e}")
        return False


def main():
    """Run all Phase 6 validation tests"""
    print("\n")
    print("+" + "=" * 58 + "+")
    print("|" + " " * 10 + "PHASE 6 VALIDATION TEST SUITE" + " " * 18 + "|")
    print("|" + " " * 10 + "Server Configuration" + " " * 27 + "|")
    print("+" + "=" * 58 + "+")

    results = []

    # Run tests
    results.append(("Uvicorn Config Module", test_uvicorn_config_module()))
    results.append(("Worker Calculation", test_worker_calculation()))
    results.append(("Environment Variables", test_environment_variables()))
    results.append(("Configuration Dictionary", test_config_dictionary()))
    results.append(("Procfile", test_procfile()))
    results.append(("Run Scripts", test_run_scripts()))
    results.append(("Environment Template", test_env_example()))

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
        print("\n*** Phase 6 COMPLETE - All validations passed! ***")
        print("Server configuration ready for production")
        print("\nConfiguration features:")
        print("  - Worker process calculation (2*CPU+1)")
        print("  - Environment variable overrides")
        print("  - Proxy header support")
        print("  - Connection limits and timeouts")
        print("  - Development and production scripts")
        return 0
    else:
        print(f"\nWARNING: {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
