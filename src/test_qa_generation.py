# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple test for QA generation functionality.
Tests the basic structure and data flow without requiring LLM calls.
"""

import json
import tempfile
from pathlib import Path
from qa_generation import (
    _normalize_aggregate,
    _extract_code_block,
    QAGenerationError,
    generate_qa_pairs,
    single_stage_qa_pair
)


def create_sample_metrics_json():
    """Create a sample metrics JSON file for testing."""
    sample_data = {
        "design_name": "test_design",
        "stages": [
            {
                "stage": "synthesis",
                "occurrence": 1,
                "design_name": "test_design",
                "sub_design_name": "test_sub",
                "metrics_file": "test_metrics.json",
                "metrics": {
                    "area": {"total": 1000.5, "unit": "um^2"},
                    "timing": {
                        "wns": -0.5,
                        "tns": -10.2,
                        "whs": 0.0,
                        "ths": 0.0
                    },
                    "power": {
                        "total": 50.0,
                        "dynamic": 40.0,
                        "leakage": 10.0,
                        "unit": "mW"
                    },
                    "violations": {
                        "drv": 5,
                        "short": 0,
                        "spacing": 2
                    }
                }
            },
            {
                "stage": "floorplan",
                "occurrence": 1,
                "design_name": "test_design",
                "sub_design_name": "test_sub",
                "metrics_file": "test_metrics_fp.json",
                "metrics": {
                    "area": {"total": 1200.8, "unit": "um^2"},
                    "utilization": 0.65,
                    "core_area": 2000.0,
                    "die_area": 2500.0
                }
            },
            {
                "stage": "placement",
                "occurrence": 1,
                "design_name": "test_design",
                "sub_design_name": "test_sub",
                "metrics_file": "test_metrics_place.json",
                "metrics": {
                    "timing": {
                        "wns": -0.3,
                        "tns": -5.1
                    },
                    "density": 0.70,
                    "wirelength": 50000
                }
            }
        ]
    }
    return sample_data


def test_normalize_aggregate():
    """Test the normalize aggregate function."""
    print("\n" + "="*80)
    print("TEST: _normalize_aggregate")
    print("="*80)
    
    # Test with valid data
    valid_data = {"design_name": "test", "stages": []}
    try:
        result = _normalize_aggregate(valid_data)
        print("✓ Valid data normalized successfully")
        assert result == valid_data
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test with invalid data (no stages)
    try:
        _normalize_aggregate({"design_name": "test"})
        print("✗ Should have raised QAGenerationError for missing stages")
        return False
    except QAGenerationError:
        print("✓ Correctly raised error for missing stages")
    
    # Test with non-dict
    try:
        _normalize_aggregate([])
        print("✗ Should have raised QAGenerationError for non-dict")
        return False
    except QAGenerationError:
        print("✓ Correctly raised error for non-dict input")
    
    return True


def test_extract_code_block():
    """Test code block extraction."""
    print("\n" + "="*80)
    print("TEST: _extract_code_block")
    print("="*80)
    
    # Test Python markdown block
    python_block = """Here's the code:
```python
x = 1
y = 2
```
Done."""
    result = _extract_code_block(python_block)
    assert result == "x = 1\ny = 2", f"Expected 'x = 1\\ny = 2', got '{result}'"
    print("✓ Python markdown block extracted correctly")
    
    # Test generic markdown block
    generic_block = """```
a = 3
b = 4
```"""
    result = _extract_code_block(generic_block)
    assert result == "a = 3\nb = 4"
    print("✓ Generic markdown block extracted correctly")
    
    # Test plain code (no markdown)
    plain_code = "answer = 'test'"
    result = _extract_code_block(plain_code)
    assert result == "answer = 'test'"
    print("✓ Plain code extracted correctly")
    
    return True


def test_sample_metrics_structure():
    """Test that sample metrics are properly structured."""
    print("\n" + "="*80)
    print("TEST: Sample Metrics Structure")
    print("="*80)
    
    sample_data = create_sample_metrics_json()
    
    # Check structure
    assert "design_name" in sample_data
    assert "stages" in sample_data
    assert len(sample_data["stages"]) == 3
    print(f"✓ Sample data has {len(sample_data['stages'])} stages")
    
    # Check each stage has required fields
    for i, stage in enumerate(sample_data["stages"]):
        assert "stage" in stage, f"Stage {i} missing 'stage' field"
        assert "occurrence" in stage, f"Stage {i} missing 'occurrence' field"
        assert "metrics" in stage, f"Stage {i} missing 'metrics' field"
        print(f"✓ Stage {i} ({stage['stage']}) has required fields")
    
    return True


def test_create_temp_json_file():
    """Test creating a temporary JSON file with metrics."""
    print("\n" + "="*80)
    print("TEST: Create Temporary JSON File")
    print("="*80)
    
    sample_data = create_sample_metrics_json()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f, indent=2)
        temp_path = f.name
    
    try:
        # Read it back
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == sample_data
        print(f"✓ Created and verified temp file: {temp_path}")
        
        # Test normalization on the loaded data
        normalized = _normalize_aggregate(loaded_data)
        print(f"✓ Successfully normalized loaded data")
        print(f"  Design: {normalized['design_name']}")
        print(f"  Stages: {len(normalized['stages'])}")
        
        return True, temp_path
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False, temp_path


def test_generate_qa_pairs_real(metrics_file: str = None):
    """
    Real integration test with actual LLM calls.
    Only run this manually with: python test_qa_generation.py --real <metrics_file>
    """
    if not metrics_file:
        print("\n⚠️  Skipping real LLM test (requires --real flag with metrics file)")
        return None
    
    print("\n" + "="*80)
    print("INTEGRATION TEST: generate_qa_pairs (REAL LLM CALLS)")
    print("="*80)
    print(f"⚠️  This test will make actual LLM API calls!")
    print(f"📁 Using metrics file: {metrics_file}")
    
    # Check if file exists
    if not Path(metrics_file).exists():
        print(f"\n❌ ERROR: File not found: {metrics_file}")
        print(f"\nℹ️  The metrics file should be a JSON file with the structure:")
        print(f"   - *_all_stages_metrics.json from your EDA flow")
        print(f"\n💡 To create a sample file for testing, run:")
        print(f"   python test_qa_generation.py")
        print(f"   (This will create a temporary test file and show you the path)")
        return False
    
    try:
        # DIRECT CALL to generate_qa_pairs with REAL LLM
        print("\n🔹 Calling: generate_qa_pairs(metrics_file, n=1, seed=42, temperature=0.7)")
        result = generate_qa_pairs(metrics_file, n=1, seed=42, temperature=0.7)
        
        print(f"\n✓ Generated {len(result['questions'])} QA pair(s)")
        
        # Display the generated QA pair
        for q in result['questions']:
            print(f"\n{'─'*80}")
            print(f"ID: {q['id']}")
            print(f"Stage: {q['stage']} (occurrence {q['occurrence']})")
            
            # Display QA pairs if they exist
            if 'qa_pairs' in q:
                print(f"\nGenerated {len(q['qa_pairs'])} QA pairs:")
                for i, qa in enumerate(q['qa_pairs'], 1):
                    print(f"\n  [{i}] Question:")
                    print(f"      {qa.get('question', 'N/A')}")
                    print(f"\n      Answer:")
                    answer_text = str(qa.get('answer', 'N/A'))
                    # Truncate long answers
                    if len(answer_text) > 500:
                        print(f"      {answer_text[:500]}...")
                    else:
                        print(f"      {answer_text}")
            
            # Show extraction code
            if 'extraction_code' in q:
                print(f"\n{'─'*40}")
                print(f"Extraction Code (preview):")
                print(f"{q['extraction_code'][:300]}...")
        
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(real_test_file: str = None):
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING QA GENERATION TESTS")
    print("="*80)
    
    results = []
    
    # Test 1: Normalize aggregate
    results.append(("Normalize Aggregate", test_normalize_aggregate()))
    
    # Test 2: Extract code block
    results.append(("Extract Code Block", test_extract_code_block()))
    
    # Test 3: Sample metrics structure
    results.append(("Sample Metrics Structure", test_sample_metrics_structure()))
    
    # Test 4: Create temp JSON
    success, temp_path = test_create_temp_json_file()
    results.append(("Create Temp JSON", success))
    
    # Test 5: Real integration test (optional)
    if real_test_file:
        real_result = test_generate_qa_pairs_real(real_test_file)
        if real_result is not None:
            results.append(("Generate QA Pairs (REAL)", real_result))
            print(f"Real result: {real_result}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if success and temp_path:
        print(f"\n📁 Sample metrics file created at: {temp_path}")
        print(f"\n💡 Usage Examples:")
        print(f"   1. Test with sample data:")
        print(f"      python test_qa_generation.py")
        print(f"\n   2. Generate QA with sample file:")
        print(f"      python qa_generation.py {temp_path} -n 1")
        print(f"\n   3. Run real integration test:")
        print(f"      python test_qa_generation.py --real {temp_path}")
    
    return passed == total


if __name__ == "__main__":
    import sys
    
    # Check for --real flag
    real_file = None
    if "--real" in sys.argv:
        idx = sys.argv.index("--real")
        if idx + 1 < len(sys.argv):
            real_file = sys.argv[idx + 1]
            print(f"🔥 Running with REAL LLM calls using: {real_file}")
        else:
            print("❌ --real flag requires a metrics file path")
            print("\n💡 Usage:")
            print("   python test_qa_generation.py --real <path_to_metrics_file.json>")
            print("\n   Example:")
            print("   python test_qa_generation.py --real /path/to/design_all_stages_metrics.json")
            sys.exit(1)
    
    success = run_all_tests(real_test_file=real_file)
    exit(0 if success else 1)

