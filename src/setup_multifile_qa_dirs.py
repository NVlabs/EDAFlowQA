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
Script to create directories for cursor agent CLI to work on multifile QA.

This script:
1. Reads the multifile QA JSON file
2. Creates a workspace directory for cursor agent
3. Copies log files and associated rpt files to the workspace

Usage:
    python setup_multifile_qa_dirs.py <qa_file> [--base_dir <dir>]
    
Example:
    python setup_multifile_qa_dirs.py ./qa_output/aes_vs_aes_20260130_160539_qa_all_multifile.json
"""

import json
import os
import shutil
from pathlib import Path
import argparse
import pdb


def setup_multifile_qa_directories(qa_file: str, base_dir: str = None):
    """
    Set up directories for cursor agent CLI based on log file paths in QA JSON.
    
    Args:
        qa_file: Path to the multifile QA JSON file
        base_dir: Base directory to create workspaces in (if None, auto-generates under design_artifacts)
    """
    print(f"Loading QA file: {qa_file}")
    
    with open(qa_file, 'r') as f:
        data = json.load(f)
    
    # Check if this is a multifile QA
    log_files = data.get("log_files", [])
    if not log_files:
        print("⚠️  No log_files found in QA JSON. This might be a single-file QA.")
        log_files = [data.get("log_file")] if data.get("log_file") else []
    
    if not log_files:
        print("❌ No log files specified in the QA JSON file.")
        return
    
    print(f"Found {len(log_files)} log files:")
    for lf in log_files:
        print(f"  - {lf}")
    
    # Extract design names from source_files, designs field, or log files
    design_names = []
    if "source_files" in data and data["source_files"]:
        # Extract from source_files field (preferred for multifile QA)
        for source_file in data["source_files"]:
            filename = os.path.basename(source_file)
            # Remove _all_stages_metrics.json or similar suffixes
            design_id = filename.replace("_all_stages_metrics.json", "")
            design_names.append(design_id)
        print(f"Using design names from source_files: {design_names}")
    elif "designs" in data and data["designs"]:
        # Use designs field if available (may have full composite IDs)
        design_names = [str(d).split('_2026-')[0] if '_2026-' in str(d) else str(d) for d in data["designs"]]
        print(f"Using design names from designs field: {design_names}")
    else:
        # Extract from log file paths
        for log_file in log_files:
            log_path = Path(log_file)
            # Extract design name from path like: ./design_artifacts/aes_asap7/aes_asap7.txt
            design_name = log_path.parent.name if log_path.parent.name != 'design_artifacts' else log_path.stem
            design_names.append(design_name)
        print(f"Using design names from log file paths: {design_names}")
    
    # Create workspace name based on designs
    if len(design_names) >= 2:
        workspace_name = f"{design_names[0]}_vs_{design_names[1]}"
    elif len(design_names) == 1:
        workspace_name = design_names[0]
    else:
        workspace_name = f"multifile_qa_{data.get('timestamp', 'workspace')}"
    
    # Determine base directory
    if base_dir is None:
        # Auto-generate under design_artifacts
        base_dir = "./design_artifacts"
    
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"\n✅ Using base directory: {base_path}")
    
    # Get QA pairs
    qa_pairs = data.get("qa_pairs", [])
    print(f"Found {len(qa_pairs)} QA pairs")
    
    # Create workspace directory with design-based name
    workspace_dir = base_path / workspace_name
    
    # Check if workspace already exists and has content
    if workspace_dir.exists() and any(workspace_dir.iterdir()):
        print(f"\n✅ Workspace already exists: {workspace_dir}")
        print(f"   Skipping file copying (workspace has {len(list(workspace_dir.iterdir()))} files)")
        print(f"\nWorkspace location: {workspace_dir.absolute()}")
        return workspace_dir
    
    # Create directory if it doesn't exist
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Creating workspace: {workspace_dir}")
    
    # Copy all log files and their associated rpt files
    for log_file_path in log_files:
        log_path = Path(log_file_path)
        
        # Resolve relative paths
        if not log_path.is_absolute():
            # Try relative to current directory first
            if log_path.exists():
                pass
            else:
                # Try relative to qa_file location
                qa_dir = Path(qa_file).parent.parent
                alt_path = qa_dir / log_file_path
                if alt_path.exists():
                    log_path = alt_path
                else:
                    print(f"⚠️  Log file not found: {log_file_path}")
                    continue
        
        if not log_path.exists():
            print(f"⚠️  Log file not found: {log_path}")
            continue
        
        # Copy the log file
        dest_log = workspace_dir / log_path.name
        print(f"  Copying: {log_path} -> {dest_log}")
        shutil.copy2(log_path, dest_log)
        
        # Copy associated rpt files from the same directory
        log_dir = log_path.parent
        rpt_files = list(log_dir.glob("*.rpt"))
        print(f"  Found {len(rpt_files)} RPT files in {log_dir}")
        
        for rpt_file in rpt_files:
            design_name = log_path.stem.replace('.txt', '')
            dest_rpt = workspace_dir / f"{design_name}_{rpt_file.name}"
            
            print(f"    Copying RPT: {rpt_file.name} -> {dest_rpt.name}")
            shutil.copy2(rpt_file, dest_rpt)
    
    print(f"\n✅ Workspace setup complete!")
    print(f"Workspace location: {workspace_dir.absolute()}")
    # pdb.set_trace()
    return workspace_dir


def main():
    parser = argparse.ArgumentParser(description='Setup directories for multifile QA cursor agent CLI')
    parser.add_argument('qa_file', type=str, help='Path to the multifile QA JSON file')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Base directory to create workspaces in (default: ./design_artifacts)')
    
    args = parser.parse_args()
    
    setup_multifile_qa_directories(args.qa_file, args.base_dir)


if __name__ == "__main__":
    main()
