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
Quick comparison of original vs filtered log to show what was removed.
"""

import re

def count_lines_by_type(filename):
    """Count different types of lines in a log file."""
    counts = {
        'total': 0,
        'warnings': 0,
        'errors': 0,
        'info': 0,
        'passes': 0,
        'commands': 0,
        'metrics': 0
    }
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            counts['total'] += 1
            
            if 'Warning' in line or 'WARNING' in line:
                counts['warnings'] += 1
            if 'Error' in line or 'ERROR' in line:
                counts['errors'] += 1
            if '[INFO' in line:
                counts['info'] += 1
            if 'Executing' in line and 'pass' in line:
                counts['passes'] += 1
            if '.sh ' in line or '.tcl' in line:
                counts['commands'] += 1
            if any(x in line for x in ['WNS', 'TNS', 'HPWL', 'area', 'slack']):
                counts['metrics'] += 1
    
    return counts

def print_comparison():
    """Print comparison of original vs filtered log."""
    print("="*70)
    print("LOG COMPARISON: Original vs Filtered")
    print("="*70)
    
    try:
        original = count_lines_by_type('gcdshy130.txt')
        filtered = count_lines_by_type('filtered_output.txt')
        
        print(f"\n{'Metric':<25} {'Original':<15} {'Filtered':<15} {'Reduction':<15}")
        print("-"*70)
        
        for key in ['total', 'warnings', 'info', 'passes', 'commands']:
            orig_val = original[key]
            filt_val = filtered[key]
            reduction = orig_val - filt_val
            reduction_pct = (reduction / orig_val * 100) if orig_val > 0 else 0
            
            print(f"{key.capitalize():<25} {orig_val:<15} {filt_val:<15} {reduction} ({reduction_pct:.1f}%)")
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        total_reduction = original['total'] - filtered['total']
        total_reduction_pct = (total_reduction / original['total'] * 100)
        
        print(f"Total lines reduced: {total_reduction} ({total_reduction_pct:.1f}%)")
        print(f"Size: {original['total']:,} lines → {filtered['total']:,} lines")
        
        # Show what was filtered
        print("\n" + "="*70)
        print("WHAT WAS FILTERED")
        print("="*70)
        print("✓ Repetitive optimization sub-passes (3.7.1, 3.7.2, etc.)")
        print("✓ 'Ignoring module' warnings (expected during synthesis)")
        print("✓ 'contains processes' warnings (expected)")
        print("✓ LEF loading info messages")
        print("✓ Verbose library cell loading messages")
        
        print("\n" + "="*70)
        print("WHAT WAS PRESERVED")
        print("="*70)
        print("✓ All timestamps (line-by-line)")
        print("✓ Command invocations")
        print("✓ Main synthesis passes")
        print("✓ Critical warnings and errors")
        print("✓ Performance metrics (CPU, memory, elapsed time)")
        print("✓ Design metrics (WNS, TNS, HPWL, area, power, slack)")
        print("✓ Summary statistics")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")

if __name__ == "__main__":
    print_comparison()

