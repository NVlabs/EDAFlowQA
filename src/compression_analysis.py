#!/usr/bin/env python3
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
Compression Analysis Script
Compares the compression rate of concatenated extracted logs vs the original log file
"""

import os


def get_file_size_bytes(filepath):
    """Get file size in bytes"""
    return os.path.getsize(filepath)


def get_file_size_kb(filepath):
    """Get file size in KB"""
    return get_file_size_bytes(filepath) / 1024


def count_lines(filepath):
    """Count the number of lines in a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def main():
    # Define the files
    extracted_files = [
        'floorplan_extract.txt',
        'placement_extract.txt',
        'cts_extract.txt',
        'route_extract.txt',
        'final_extract.txt'
    ]
    original_file = 'gcdshy130.txt'
    
    print("=" * 80)
    print("COMPRESSION ANALYSIS: Extracted Logs vs Original Log")
    print("=" * 80)
    print()
    
    # Check if all files exist
    missing_files = []
    for f in extracted_files + [original_file]:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        print(f"ERROR: The following files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    # Analyze each extracted file
    print("EXTRACTED FILES:")
    print("-" * 80)
    total_extracted_size = 0
    total_extracted_lines = 0
    
    for fname in extracted_files:
        size_bytes = get_file_size_bytes(fname)
        size_kb = get_file_size_kb(fname)
        lines = count_lines(fname)
        total_extracted_size += size_bytes
        total_extracted_lines += lines
        
        print(f"{fname:30s}  {size_kb:8.2f} KB  ({size_bytes:,} bytes)  {lines:,} lines")
    
    print("-" * 80)
    print(f"{'TOTAL EXTRACTED:':30s}  {total_extracted_size/1024:8.2f} KB  ({total_extracted_size:,} bytes)  {total_extracted_lines:,} lines")
    print()
    
    # Analyze original file
    print("ORIGINAL FILE:")
    print("-" * 80)
    original_size = get_file_size_bytes(original_file)
    original_size_kb = get_file_size_kb(original_file)
    original_lines = count_lines(original_file)
    
    print(f"{original_file:30s}  {original_size_kb:8.2f} KB  ({original_size:,} bytes)  {original_lines:,} lines")
    print()
    
    # Calculate compression metrics
    print("COMPRESSION ANALYSIS:")
    print("=" * 80)
    
    # Size compression
    size_reduction = original_size - total_extracted_size
    size_compression_ratio = (1 - total_extracted_size / original_size) * 100
    size_retention_ratio = (total_extracted_size / original_size) * 100
    
    print(f"Original file size:          {original_size_kb:10.2f} KB  ({original_size:,} bytes)")
    print(f"Extracted files size:        {total_extracted_size/1024:10.2f} KB  ({total_extracted_size:,} bytes)")
    print(f"Size reduction:              {size_reduction/1024:10.2f} KB  ({size_reduction:,} bytes)")
    print(f"Size compression rate:       {size_compression_ratio:10.2f} %")
    print(f"Size retention rate:         {size_retention_ratio:10.2f} %")
    print()
    
    # Line compression
    line_reduction = original_lines - total_extracted_lines
    line_compression_ratio = (1 - total_extracted_lines / original_lines) * 100
    line_retention_ratio = (total_extracted_lines / original_lines) * 100
    
    print(f"Original line count:         {original_lines:10,} lines")
    print(f"Extracted line count:        {total_extracted_lines:10,} lines")
    print(f"Line reduction:              {line_reduction:10,} lines")
    print(f"Line compression rate:       {line_compression_ratio:10.2f} %")
    print(f"Line retention rate:         {line_retention_ratio:10.2f} %")
    print()
    
    # Additional metrics
    print("EFFICIENCY METRICS:")
    print("-" * 80)
    avg_line_size_original = original_size / original_lines
    avg_line_size_extracted = total_extracted_size / total_extracted_lines
    
    print(f"Average line size (original):   {avg_line_size_original:10.2f} bytes/line")
    print(f"Average line size (extracted):  {avg_line_size_extracted:10.2f} bytes/line")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"✓ The extracted logs retain {size_retention_ratio:.2f}% of the original file size")
    print(f"✓ This represents a {size_compression_ratio:.2f}% compression rate")
    print(f"✓ The extracted logs contain {line_retention_ratio:.2f}% of the original lines")
    print(f"✓ Compression ratio: {original_size/total_extracted_size:.2f}:1")
    print()
    
    # Breakdown by stage
    print("BREAKDOWN BY STAGE:")
    print("-" * 80)
    for fname in extracted_files:
        size = get_file_size_bytes(fname)
        lines = count_lines(fname)
        size_pct = (size / total_extracted_size) * 100
        line_pct = (lines / total_extracted_lines) * 100
        print(f"{fname:30s}  {size_pct:5.1f}% of size, {line_pct:5.1f}% of lines")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

