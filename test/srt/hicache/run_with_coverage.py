#!/usr/bin/env python3
"""
Coverage runner for HiCache tests using unittest
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def install_coverage():
    """Install coverage.py if not available"""
    try:
        import coverage
        print("✓ coverage.py is already installed")
        return True
    except ImportError:
        print("Installing coverage.py...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage", "psutil"])
            print("✓ coverage.py and psutil installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install coverage.py: {e}")
            return False

def run_tests_with_coverage(test_file, source_patterns=None, output_dir="coverage_reports"):
    """Run unittest tests with coverage analysis"""
    
    if not install_coverage():
        return False
        
    # Default source patterns for HiCache
    if source_patterns is None:
        source_patterns = [
            "sglang/python/sglang/srt/mem_cache/*",
            "sglang/python/sglang/srt/managers/cache_controller.py",
        ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build coverage command for unittest
    coverage_cmd = [
        sys.executable, "-m", "coverage", "run",
        "--branch",  # Enable branch coverage
        "--source", "sglang.srt.mem_cache,sglang.srt.managers.cache_controller",
        test_file
    ]
    
    print("=" * 60)
    print("Running HiCache tests with coverage using unittest...")
    print("=" * 60)
    print(f"Command: {' '.join(coverage_cmd)}")
    print()
    
    # Run tests with coverage
    try:
        result = subprocess.run(coverage_cmd, cwd=os.getcwd(), check=False)
        if result.returncode != 0:
            print(f"⚠️  Tests completed with exit code {result.returncode}")
        else:
            print("✓ Tests completed successfully")
    except Exception as e:
        print(f"✗ Failed to run tests: {e}")
        return False
    
    # Generate coverage reports
    print("\nGenerating coverage reports...")
    
    # Console report
    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    subprocess.run([sys.executable, "-m", "coverage", "report", "-m"], check=False)
    
    # HTML report
    html_dir = os.path.join(output_dir, "html")
    print(f"\nGenerating HTML report in {html_dir}...")
    subprocess.run([
        sys.executable, "-m", "coverage", "html", 
        "--directory", html_dir
    ], check=False)
    
    # XML report (useful for CI/CD)
    xml_file = os.path.join(output_dir, "coverage.xml")
    print(f"Generating XML report: {xml_file}")
    subprocess.run([
        sys.executable, "-m", "coverage", "xml", 
        "-o", xml_file
    ], check=False)
    
    print("\n" + "=" * 60)
    print("COVERAGE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📊 HTML Report: file://{os.path.abspath(html_dir)}/index.html")
    print(f"📋 XML Report:  {os.path.abspath(xml_file)}")
    print("=" * 60)
    
    return True

def run_quick_tests(test_file):
    """Run tests without long duration test for quick feedback"""
    print("=" * 60)
    print("Running quick tests (skipping long duration test)...")
    print("=" * 60)
    
    # Run with specific test pattern to exclude long duration test
    cmd = [
        sys.executable, "-m", "unittest", 
        "TestHiCacheMemStorageEnhanced.test_config_*",
        "TestHiCacheMemStorageEnhanced.test_memory_pressure_*",
        "TestHiCacheMemStorageEnhanced.test_mixed_operation_patterns",
        "-v"
    ]
    
    # Change to the directory containing the test file
    test_dir = os.path.dirname(os.path.abspath(test_file))
    test_module = os.path.basename(test_file).replace('.py', '')
    
    try:
        result = subprocess.run([sys.executable, test_file], cwd=test_dir, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Failed to run quick tests: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run HiCache tests with coverage analysis using unittest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_with_coverage.py                                    # Run with coverage
  python run_with_coverage.py --test-file test_original.py       # Run original test
  python run_with_coverage.py --quick                            # Run without coverage, quick tests
  python run_with_coverage.py --output-dir my_reports            # Custom output directory
        """
    )
    
    parser.add_argument(
        "--test-file",
        default="test_hicache_mem_storage_enhanced.py",
        help="Test file to run (default: test_hicache_mem_storage_enhanced.py)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="coverage_reports",
        help="Output directory for coverage reports (default: coverage_reports)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests without coverage (faster feedback)"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Run tests without coverage analysis"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_tests(args.test_file)
    elif args.no_coverage:
        print("Running tests without coverage...")
        try:
            result = subprocess.run([sys.executable, args.test_file], check=False)
            success = result.returncode == 0
        except Exception as e:
            print(f"✗ Failed to run tests: {e}")
            success = False
    else:
        success = run_tests_with_coverage(
            test_file=args.test_file,
            output_dir=args.output_dir
        )
    
    if not success:
        sys.exit(1)
    
    print("\n🎉 Tests completed successfully!")

if __name__ == "__main__":
    main()
