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
Configuration FLAGS for LLM inference and log analysis.
This file contains all configurable parameters used across the codebase.
"""

class FLAGS:
    # Model configuration
    model = 'gpt-4o-20241120'  # Options: 'gpt-4o-20241120', 'claude-3-7-sonnet-20250219', etc.
    model_source = 'Perflab'  # Options: 'Mark', 'Perflab'
    
    # Temperature for model inference
    temperature = 0.1  # Default temperature for all LLM calls (0.0 to 1.0)
    
    # Dataset configuration
    dataset = 'openroad'  # Dataset being used for analysis

