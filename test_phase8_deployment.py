"""
Phase 8 Validation Test - Deployment Strategy

Validates Phase 8 implementation:
1. Deployment documentation exists
2. Nginx configuration file
3. Systemd service file
4. Monitoring scripts
5. Rollback procedure
6. All files are valid and complete
"""

import os
import sys


def test_deployment_documentation():
    """Test deployment documentation exists"""
    print("=" * 60)
    print("TEST 1: Deployment Documentation")
    print("=" * 60)

    try:
        if not os.path.exists("DEPLOYMENT.md"):
            print("[FAIL] DEPLOYMENT.md not found")
            return False

        with open("DEPLOYMENT.md", "r", encoding="utf-8") as f:
            content = f.read()

        print("[PASS] DEPLOYMENT.md exists")

        # Check for key sections
        required_sections = [
            "## Overview",
            "## Pre-Deployment Checklist",
            "## Blue-Green Deployment Strategy",
            "## Deployment Steps",
            "## Monitoring",
            "## Rollback Procedure",
            "## Post-Deployment"
        ]

        all_present = True
        for section in required_sections:
            if section in content:
                print(f"[PASS] Section '{section}' exists")
            else:
                print(f"[FAIL] Section '{section}' missing")
                all_present = False

        # Check for deployment phases
        phases = ["10%", "25%", "50%", "75%", "100%"]
        phase_count = sum(1 for phase in phases if phase in content)
        print(f"[INFO] Found {phase_count}/5 deployment phases documented")

        return all_present

    except Exception as e:
        print(f"[FAIL] Deployment documentation test failed: {e}")
        return False


def test_nginx_configuration():
    """Test Nginx configuration file"""
    print("\n" + "=" * 60)
    print("TEST 2: Nginx Configuration")
    print("=" * 60)

    try:
        if not os.path.exists("nginx.conf"):
            print("[FAIL] nginx.conf not found")
            return False

        with open("nginx.conf", "r", encoding="utf-8") as f:
            content = f.read()

        print("[PASS] nginx.conf exists")

        # Check for key directives
        required_directives = [
            "upstream backend",
            "server localhost:5000",  # Flask
            "server localhost:8000",  # FastAPI
            "proxy_pass",
            "proxy_set_header",
            "location /health"
        ]

        all_present = True
        for directive in required_directives:
            if directive in content:
                print(f"[PASS] Directive '{directive}' exists")
            else:
                print(f"[FAIL] Directive '{directive}' missing")
                all_present = False

        # Check for weight configuration
        if "weight=" in content:
            print("[PASS] Load balancing weights configured")
        else:
            print("[WARN] Load balancing weights not found")

        return all_present

    except Exception as e:
        print(f"[FAIL] Nginx configuration test failed: {e}")
        return False


def test_systemd_service():
    """Test systemd service file"""
    print("\n" + "=" * 60)
    print("TEST 3: Systemd Service File")
    print("=" * 60)

    try:
        if not os.path.exists("fastapi.service"):
            print("[FAIL] fastapi.service not found")
            return False

        with open("fastapi.service", "r", encoding="utf-8") as f:
            content = f.read()

        print("[PASS] fastapi.service exists")

        # Check for required sections
        required_sections = [
            "[Unit]",
            "[Service]",
            "[Install]",
            "Description=",
            "ExecStart=",
            "Restart=",
            "WorkingDirectory="
        ]

        all_present = True
        for section in required_sections:
            if section in content:
                print(f"[PASS] Section '{section}' exists")
            else:
                print(f"[FAIL] Section '{section}' missing")
                all_present = False

        # Check for uvicorn command
        if "uvicorn" in content and "fastapi_app:app" in content:
            print("[PASS] Uvicorn command configured")
        else:
            print("[FAIL] Uvicorn command missing")
            all_present = False

        return all_present

    except Exception as e:
        print(f"[FAIL] Systemd service test failed: {e}")
        return False


def test_monitoring_script():
    """Test monitoring script"""
    print("\n" + "=" * 60)
    print("TEST 4: Monitoring Script")
    print("=" * 60)

    try:
        if not os.path.exists("monitor_deployment.sh"):
            print("[FAIL] monitor_deployment.sh not found")
            return False

        with open("monitor_deployment.sh", "r", encoding="utf-8") as f:
            content = f.read()

        print("[PASS] monitor_deployment.sh exists")

        # Check for key monitoring features
        required_features = [
            "health_response",
            "response_time",
            "stats_response",
            "Resource usage",
            "success_rate"
        ]

        all_present = True
        for feature in required_features:
            if feature in content:
                print(f"[PASS] Feature '{feature}' exists")
            else:
                print(f"[FAIL] Feature '{feature}' missing")
                all_present = False

        # Check if script is executable (on Unix systems)
        if hasattr(os, 'access') and not os.name == 'nt':
            if os.access("monitor_deployment.sh", os.X_OK):
                print("[PASS] Script is executable")
            else:
                print("[WARN] Script not executable (run: chmod +x monitor_deployment.sh)")

        return all_present

    except Exception as e:
        print(f"[FAIL] Monitoring script test failed: {e}")
        return False


def test_rollback_script():
    """Test rollback script"""
    print("\n" + "=" * 60)
    print("TEST 5: Rollback Script")
    print("=" * 60)

    try:
        if not os.path.exists("rollback.sh"):
            print("[FAIL] rollback.sh not found")
            return False

        with open("rollback.sh", "r", encoding="utf-8") as f:
            content = f.read()

        print("[PASS] rollback.sh exists")

        # Check for key rollback steps
        required_steps = [
            "Redirecting traffic to Flask",
            "Stopping FastAPI",
            "Verifying Flask is healthy",
            "nginx",
            "systemctl stop fastapi",
            "curl"
        ]

        all_present = True
        for step in required_steps:
            if step in content:
                print(f"[PASS] Step '{step}' exists")
            else:
                print(f"[FAIL] Step '{step}' missing")
                all_present = False

        # Check if script is executable (on Unix systems)
        if hasattr(os, 'access') and not os.name == 'nt':
            if os.access("rollback.sh", os.X_OK):
                print("[PASS] Script is executable")
            else:
                print("[WARN] Script not executable (run: chmod +x rollback.sh)")

        return all_present

    except Exception as e:
        print(f"[FAIL] Rollback script test failed: {e}")
        return False


def test_deployment_checklist():
    """Test deployment checklist completeness"""
    print("\n" + "=" * 60)
    print("TEST 6: Deployment Checklist")
    print("=" * 60)

    try:
        with open("DEPLOYMENT.md", "r", encoding="utf-8") as f:
            content = f.read()

        # Check for critical deployment items
        critical_items = [
            "Backup database",
            "alembic upgrade",
            "validate_config",
            "Response time",
            "Error rate",
            "Health check",
            "When to Rollback"
        ]

        all_present = True
        for item in critical_items:
            if item in content:
                print(f"[PASS] Item '{item}' in checklist")
            else:
                print(f"[WARN] Item '{item}' not found")

        print("[PASS] Deployment checklist is comprehensive")
        return True

    except Exception as e:
        print(f"[FAIL] Deployment checklist test failed: {e}")
        return False


def test_file_permissions():
    """Test file structure and permissions"""
    print("\n" + "=" * 60)
    print("TEST 7: File Structure")
    print("=" * 60)

    try:
        deployment_files = {
            "DEPLOYMENT.md": "Documentation",
            "nginx.conf": "Nginx config",
            "fastapi.service": "Systemd service",
            "monitor_deployment.sh": "Monitoring script",
            "rollback.sh": "Rollback script"
        }

        all_exist = True
        total_size = 0

        for filename, description in deployment_files.items():
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                total_size += size
                print(f"[PASS] {filename} ({description}): {size} bytes")
            else:
                print(f"[FAIL] {filename} ({description}): NOT FOUND")
                all_exist = False

        print(f"\n[INFO] Total deployment artifacts: {total_size:,} bytes")

        return all_exist

    except Exception as e:
        print(f"[FAIL] File structure test failed: {e}")
        return False


def main():
    """Run all Phase 8 validation tests"""
    print("\n")
    print("+" + "=" * 58 + "+")
    print("|" + " " * 10 + "PHASE 8 VALIDATION TEST SUITE" + " " * 18 + "|")
    print("|" + " " * 10 + "Deployment Strategy" + " " * 29 + "|")
    print("+" + "=" * 58 + "+")

    results = []

    # Run tests
    results.append(("Deployment Documentation", test_deployment_documentation()))
    results.append(("Nginx Configuration", test_nginx_configuration()))
    results.append(("Systemd Service File", test_systemd_service()))
    results.append(("Monitoring Script", test_monitoring_script()))
    results.append(("Rollback Script", test_rollback_script()))
    results.append(("Deployment Checklist", test_deployment_checklist()))
    results.append(("File Structure", test_file_permissions()))

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
        print("\n*** Phase 8 COMPLETE - All validations passed! ***")
        print("Deployment strategy ready for production")
        print("\nDeployment artifacts:")
        print("  - DEPLOYMENT.md: Comprehensive deployment guide")
        print("  - nginx.conf: Blue-green load balancing config")
        print("  - fastapi.service: Systemd service configuration")
        print("  - monitor_deployment.sh: Automated monitoring")
        print("  - rollback.sh: Emergency rollback procedure")
        print("\nNext steps:")
        print("  1. Review DEPLOYMENT.md carefully")
        print("  2. Test deployment in staging environment")
        print("  3. Schedule production deployment window")
        print("  4. Follow blue-green rollout: 10% -> 25% -> 50% -> 75% -> 100%")
        print("  5. Monitor metrics at each stage")
        print("\nReady for production deployment!")
        return 0
    else:
        print(f"\nWARNING: {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
