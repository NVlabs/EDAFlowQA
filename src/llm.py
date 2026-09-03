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
Simple LLM helper for log analysis.
One class to handle all LLM interactions with different prompt templates.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from openai import AzureOpenAI
import json
import httpx
from config_FLAGS import FLAGS
import os
from pathlib import Path
import time  # Add time import for manual delays
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, APITimeoutError, AzureOpenAI, OpenAI, APIConnectionError, InternalServerError
import requests
import tiktoken  # Add tiktoken for token counting
import pdb
from autogen.oai.client import ModelClient, OpenAIWrapper
# from autogen import config_list_from_json

from autogen import config_list_from_json
from hardware_agent.basic_client import BasicClient

DEFAULT_MODEL_CONFIG = os.environ.get("LLM_CONFIG_FILE")
DEFAULT_OPENAI_TIMEOUT = httpx.Timeout(300.0, connect=30.0)


# Token counting functions
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string."""
    try:
        # Try to get the appropriate encoder for the model
        if "gpt-4" in model.lower():
            encoder = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model.lower():
            encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            # Default to cl100k_base encoding (used by gpt-4 and gpt-3.5-turbo)
            encoder = tiktoken.get_encoding("cl100k_base")
        
        return len(encoder.encode(text))
    except Exception as e:
        print(f"Warning: Could not count tokens for model {model}: {e}")
        # Fallback: rough estimate (4 characters per token)
        return len(text) // 4

def check_token_limits(system_prompt: str, user_prompt: str, model: str = "gpt-4") -> Dict[str, Any]:
    """Check token limits for the given prompts."""
    system_tokens = count_tokens(system_prompt, model)
    user_tokens = count_tokens(user_prompt, model)
    total_tokens = system_tokens + user_tokens
    
    # Model-specific limits (leaving room for response)
    if "gpt-4" in model.lower():
        if "gpt-4o" in model.lower():
            max_tokens = 128000 - 4000  # 128K context, reserve 4K for response
        else:
            max_tokens = 8000 - 1000   # 8K context, reserve 1K for response
    elif "gpt-3.5" in model.lower():
        max_tokens = 4000 - 1000      # 4K context, reserve 1K for response
    elif "claude" in model.lower():
        if "claude-3-5-sonnet" in model.lower() or "claude-3-7-sonnet" in model.lower():
            max_tokens = 131072 - 8000  # 200K context, reserve 8K for response
        elif "claude-3" in model.lower():
            max_tokens = 131072 - 8000  # Claude-3 models have 200K context
        else:
            max_tokens = 100000 - 4000  # Other Claude models, conservative estimate
    else:
        max_tokens = 4000 - 1000      # Default conservative limit
    
    return {
        'system_tokens': system_tokens,
        'user_tokens': user_tokens,
        'total_tokens': total_tokens,
        'max_tokens': max_tokens,
        'within_limit': total_tokens <= max_tokens,
        'excess_tokens': max(0, total_tokens - max_tokens)
    }

# ANSI color codes
CYAN = '\033[96m'
GREEN = '\033[92m'
RESET = '\033[0m'


def _resolve_model_config_path(config_file: Optional[str] = None) -> Path:
    """Resolve a model config file from an explicit path or LLM_CONFIG_FILE."""
    config_name = config_file or DEFAULT_MODEL_CONFIG
    path = Path(config_name).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parent / path,
        Path(__file__).resolve().parent.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_model_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load an OpenAI-compatible model config JSON object.

    The config can be either a dict or a one-item list, matching AutoGen-style
    OAI_CONFIG_LIST files:
        [{"api_key": "...", "base_url": "...", "model": "..."}]
    """
    config_path = _resolve_model_config_path(config_file)
    with open(config_path) as f:
        config = json.load(f)
    cfg = config[0] if isinstance(config, list) else config
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid model config in {config_path}: expected dict or list[dict].")

    api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"Model config {config_path} must define api_key or OPENAI_API_KEY must be set."
        )
    if not cfg.get("model"):
        raise ValueError(f"Model config {config_path} must define model.")

    resolved = dict(cfg)
    resolved["api_key"] = api_key
    return resolved


def create_client(
    config_file: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[OpenAI, Dict[str, Any]]:
    """Create an OpenAI-compatible client from a model config file."""
    cfg = load_model_config(config_file)
    client_timeout = timeout if timeout is not None else cfg.get("timeout", DEFAULT_OPENAI_TIMEOUT)
    if not isinstance(client_timeout, httpx.Timeout):
        client_timeout = httpx.Timeout(float(client_timeout), connect=min(30.0, float(client_timeout)))

    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        timeout=client_timeout,
    )
    return client, cfg


@retry(
    retry=retry_if_exception_type((
        APITimeoutError,
        APIError,
        APIConnectionError,
        InternalServerError,
        httpx.TimeoutException,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
        ConnectionError,
        TimeoutError,
    )),
    wait=wait_exponential(multiplier=2, min=3, max=120),
    stop=stop_after_attempt(8),
)
def call_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 50000,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Call a text-only OpenAI-compatible chat completion model."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    text = msg.content or ""
    reasoning = (
        getattr(msg, "reasoning_content", None)
        or getattr(msg, "reasoning", None)
        or ""
    )
    return text or reasoning


def call_configured_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = None,
    timeout: int = 300,
    max_tokens: int = 50000,
    config_file: Optional[str] = None,
) -> str:
    """Load model config, create an OpenAI-compatible client, and call it."""
    effective_temperature = temperature if temperature is not None else FLAGS.temperature
    client, cfg = create_client(config_file, timeout=timeout)
    model = cfg["model"]
    extra_body = cfg.get("extra_body")

    token_info = check_token_limits(system_prompt, user_prompt, model)
    if token_info["total_tokens"] > token_info["max_tokens"] * 0.8:
        print(
            f"High token usage: {token_info['total_tokens']} tokens "
            f"({token_info['system_tokens']} system + {token_info['user_tokens']} user)"
        )
        print(
            f"   Limit: {token_info['max_tokens']} tokens "
            f"({token_info['total_tokens'] / token_info['max_tokens'] * 100:.1f}% of limit)"
        )

    print(f"\n{CYAN}Query:{RESET} {system_prompt[:1000]} {user_prompt[:1000]}")
    print(f"Model: {model}")
    if cfg.get("base_url"):
        print(f"Base URL: {cfg['base_url']}")
    print("-" * 50)

    try:
        result = call_model(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=effective_temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        print(f"{GREEN}Response:{RESET} {result[:1000]}")
        print("-" * 50)
        return result
    except (APITimeoutError, APIError) as e:
        print("API ERROR - Token Analysis:")
        print(f"   Model: {model}")
        print(f"   System prompt tokens: {token_info['system_tokens']}")
        print(f"   User prompt tokens: {token_info['user_tokens']}")
        print(f"   Total tokens: {token_info['total_tokens']}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error: {str(e)}")
        raise


# Gateway chat from DAR
def _get_claude_api_key():
    """Get an OAuth API key from environment-provided NVIDIA credentials."""
    client_id = os.environ.get("NVIDIA_CLIENT_ID")
    client_secret = os.environ.get("NVIDIA_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError(
            "NVIDIA_CLIENT_ID and NVIDIA_CLIENT_SECRET must be set to fetch an OAuth token."
        )

    url = "https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token"
    headers = {
        "Content-Type": "application/json",
        "dataClassification": "secret",
    }
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "azureopenai-readwrite awsanthropic-readwrite",
        "grant_type": "client_credentials"
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            api_key = response.json().get('access_token')
            print("Successfully fetched API key from OAuth endpoint.")
            return api_key
        else:
            print(f"Failed to get API key: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error getting API key: {e}")
        return None
@retry(
    retry=retry_if_exception_type((APITimeoutError, APIError, APIConnectionError, httpx.TimeoutException, httpx.ReadTimeout, InternalServerError)),
    wait=wait_exponential(multiplier=2, min=3, max=3600),
    stop=stop_after_attempt(30),  # Increase attempts from 5 to 8
)
def call_dar_gateway(system_prompt: str, user_prompt: str, temperature: float = 0.1, oai_file: str = "OAI_CONFIG_LIST_O1_DAR"):
    return call_Perflab_llm(system_prompt, user_prompt, temperature, timeout=3000, max_tokens=50000)

@retry(
    retry=retry_if_exception_type((APITimeoutError, APIError, APIConnectionError, httpx.TimeoutException, httpx.ReadTimeout, InternalServerError)),
    wait=wait_exponential(multiplier=2, min=3, max=3600),
    stop=stop_after_attempt(30),  # Increase attempts from 5 to 8
)
def call_NIM_openmodels(system_prompt: str, user_prompt: str, temperature: float = 0.1, oai_file: str = "OAI_CONFIG_LIST_NIM"):
    # Use provided temperature or default from FLAGS
    effective_temperature = temperature if temperature is not None else FLAGS.temperature
    
    # Check token limits before making the call
    token_info = check_token_limits(system_prompt, user_prompt, FLAGS.model)
    # Print token info if it's over 80% of the limit or exceeds limit
    if token_info['total_tokens'] > token_info['max_tokens'] * 0.8:
        print(f"ð High token usage: {token_info['total_tokens']} tokens ({token_info['system_tokens']} system + {token_info['user_tokens']} user)")
        print(f"   Limit: {token_info['max_tokens']} tokens ({token_info['total_tokens']/token_info['max_tokens']*100:.1f}% of limit)")
        # pdb.set_trace()

    # Dynamic timeout based on token count - larger prompts need more time
    # dynamic_timeout = max(timeout, token_info['total_tokens'] / 100)  # 1 second per 100 tokens, minimum of timeout
    # print(f"â±ï¸ Using timeout: {dynamic_timeout:.1f}s (tokens: {token_info['total_tokens']})")

    config_list = config_list_from_json(env_or_file=oai_file)
    llm_config = {"config_list": config_list}
    print(f"LLM config: {llm_config}")
    try:
        # if "o1" in config_list:
        gpt = BasicClient(llm_config=llm_config)
        #else:
        #    gpt = BasicClient(llm_config={"config_list": config_list, "temperature": effective_temperature})
        response = gpt.ask_client(system_prompt + "\n\n" + user_prompt)
        if config_list[0]['model'] == 'openai/gpt-oss-120b':
            response = response['response'].content
        else:
            response = response['response']
        print(f"{GREEN}Response:{RESET}", response[:1000])
        return response
    except (APITimeoutError, Exception) as e:
        print(f"â Error calling NIM openmodels: {e}")
        raise  # Re-raise to trigger retry mechanism

def direct_call_Mark_claude37(system_prompt: str, user_prompt: str, temperature: float = 0.1, timeout=300):
    response = call_claude_37(system_prompt+"\n\n"+user_prompt, temperature=temperature)
    
    # Extract content from the response
    if response and 'content' in response and response['content']:
        content_text = response['content'][0].get('text', '')
        print(f"{GREEN}Response:{RESET}", content_text[:1000])
        return content_text
    else:
        # print(f"â Raw Result: {str(response)}")
        print(f"{GREEN}Raw Response:{RESET}", str(response))
        return response
    
@retry(
    retry=retry_if_exception_type((APITimeoutError, APIError, APIConnectionError, httpx.TimeoutException, httpx.ReadTimeout, InternalServerError)),
    wait=wait_exponential(multiplier=2, min=3, max=3600),
    stop=stop_after_attempt(30),  # Increase attempts from 5 to 8
)
def call_gpt4o(system_prompt: str, user_prompt: str, temperature: float = 0.1):
    effective_temperature = temperature if temperature is not None else FLAGS.temperature
    token_info = check_token_limits(system_prompt, user_prompt, "gpt-4o")
    # Print token info if it's over 80% of the limit or exceeds limit
    if token_info['total_tokens'] > token_info['max_tokens'] * 0.8:
        print(f"ð High token usage: {token_info['total_tokens']} tokens ({token_info['system_tokens']} system + {token_info['user_tokens']} user)")
        print(f"   Limit: {token_info['max_tokens']} tokens ({token_info['total_tokens']/token_info['max_tokens']*100:.1f}% of limit)")
    
    config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST_DAR")


    llm_config = {"config_list": config_list,
                  "temperature": effective_temperature}
                  # "timeout": dynamic_timeout}

    print(f"\n{CYAN}Query:{RESET} {user_prompt[:1000]}")
    print("-" * 50)
        
    try:
        print(f"Calling GPT-4o from DAR gateway...")
        gpt = BasicClient(llm_config=llm_config)
        response = gpt.ask_client(system_prompt+"\n\n"+user_prompt)
        response = response['response']
        print(f"{GREEN}Response:{RESET}", response[:1000])
        return response
    except (APITimeoutError, Exception) as e:
        # Log detailed token information on timeout
        print(f"â TIMEOUT ERROR - Token Analysis:")
        print(f"   Model: {FLAGS.model}")
        print(f"   System prompt tokens: {token_info['system_tokens']}")
        print(f"   User prompt tokens: {token_info['user_tokens']}")
        print(f"   Total tokens: {token_info['total_tokens']}")
        print(f"   Max tokens: {token_info['max_tokens']}")
        print(f"   Usage: {token_info['total_tokens']/token_info['max_tokens']*100:.1f}% of limit")
        print(f"   System prompt length: {len(system_prompt)} chars")
        print(f"   User prompt length: {len(user_prompt)} chars")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error: {str(e)}")
        raise  # Re-raise to trigger retry mechanism

def direct_call_Mark_gpt4o(system_prompt: str, user_prompt: str, temperature: float = 0.1, timeout=300):
    response = call_gpt4o(system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
    return response

@retry(
    retry=retry_if_exception_type((APITimeoutError, APIError, APIConnectionError, httpx.TimeoutException, httpx.ReadTimeout, InternalServerError)),
    wait=wait_exponential(multiplier=2, min=3, max=60),
    stop=stop_after_attempt(10),  # Increase attempts from 5 to 8
)
def call_Mark_llm(system_prompt: str, user_prompt: str, temperature: float = None, timeout=300):  # Increased from 30 to 300 seconds
    print(f"Calling Mark LLM with model: {FLAGS.model}")
    assert FLAGS.model_source == 'Mark'
    
    # Use provided temperature or default from FLAGS
    effective_temperature = temperature if temperature is not None else FLAGS.temperature
    
    # Check token limits before making the call
    token_info = check_token_limits(system_prompt, user_prompt, FLAGS.model)
    # Print token info if it's over 80% of the limit or exceeds limit
    if token_info['total_tokens'] > token_info['max_tokens'] * 0.8:
        print(f"ð High token usage: {token_info['total_tokens']} tokens ({token_info['system_tokens']} system + {token_info['user_tokens']} user)")
        print(f"   Limit: {token_info['max_tokens']} tokens ({token_info['total_tokens']/token_info['max_tokens']*100:.1f}% of limit)")
        # pdb.set_trace()

    # Dynamic timeout based on token count - larger prompts need more time
    # dynamic_timeout = max(timeout, token_info['total_tokens'] / 100)  # 1 second per 100 tokens, minimum of timeout
    # print(f"â±ï¸ Using timeout: {dynamic_timeout:.1f}s (tokens: {token_info['total_tokens']})")

    with open("OAI_CONFIG_LIST_DAR", 'r') as f:
        config_list = json.load(f)[0]
    # config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST_DAR")
    if FLAGS.model == 'gpt-4o-20241120':
        model = 'gpt-4o'
    
        config_list = [
            {
                "model": model,
                # "api_key": _get_api_key(),
                "api_key": "",
                "gateway_chat_type": "dar_mark_team",
            }
        ]

        llm_config = {"config_list": config_list,
                    "temperature": effective_temperature}
                    # "timeout": dynamic_timeout}

        print(f"\n{CYAN}Query:{RESET} {user_prompt[:1000]}")
        print("-" * 50)
        
        try:
            gpt = BasicClient(llm_config=llm_config)
            response = gpt.ask_client(system_prompt+"\n\n"+user_prompt)
            response = response['response']
            print(f"{GREEN}Response:{RESET}", response[:1000])
            return response
        except (APITimeoutError, Exception) as e:
            # Log detailed token information on timeout
            print(f"â TIMEOUT ERROR - Token Analysis:")
            print(f"   Model: {FLAGS.model}")
            print(f"   System prompt tokens: {token_info['system_tokens']}")
            print(f"   User prompt tokens: {token_info['user_tokens']}")
            print(f"   Total tokens: {token_info['total_tokens']}")
            print(f"   Max tokens: {token_info['max_tokens']}")
            print(f"   Usage: {token_info['total_tokens']/token_info['max_tokens']*100:.1f}% of limit")
            print(f"   System prompt length: {len(system_prompt)} chars")
            print(f"   User prompt length: {len(user_prompt)} chars")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error: {str(e)}")
            raise  # Re-raise to trigger retry mechanism
    
    elif FLAGS.model == 'claude-3-7-sonnet-20250219':
        print(f"Calling Claude 3.7 Sonnet from DAR gateway...")
        response = call_claude_37(system_prompt+"\n\n"+user_prompt, temperature=effective_temperature)
        
        # Extract content from the response
        if response and 'content' in response and response['content']:
            content_text = response['content'][0].get('text', '')
            print(f"{GREEN}Response:{RESET}", content_text[:1000])
            return content_text
        else:
            # print(f"â Raw Result: {str(response)}")
            print(f"{GREEN}Raw Response:{RESET}", str(response))
            return response

@retry(
    retry=retry_if_exception_type((APITimeoutError, APIError, APIConnectionError, httpx.TimeoutException, httpx.ReadTimeout, InternalServerError)),
    wait=wait_exponential(multiplier=2, min=3, max=60),
    stop=stop_after_attempt(10),  # Increase attempts from 5 to 8
)
def call_Perflab_claude37(system_prompt: str, user_prompt: str, temperature: float = None, timeout=30, max_tokens: int = 50000):
    return call_Perflab_llm(system_prompt, user_prompt, temperature, timeout, max_tokens) # for old code compatibility

def call_Perflab_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = None,
    timeout=30,
    max_tokens: int = 50000,
    config_file: Optional[str] = None,
):
    """
    Call the configured OpenAI-compatible model.

    By default this reads LLM_CONFIG_FILE, falling back to OAI_CONFIG_LIST_Perflab.
    The config format is:
        [{"api_key": "...", "base_url": "...", "model": "..."}]
    
    Args:
        system_prompt: System/role prompt.
        user_prompt: User message / task.
        temperature: Sampling temperature.
        timeout: Client timeout in seconds.
        max_tokens: Maximum tokens in response.
        config_file: Optional model config path.
    
    Returns:
        The model's response as a plain string.
    """
    return call_configured_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        config_file=config_file,
    )

def should_stop_on_token_limit(system_prompt: str, user_prompt: str, model: str = None) -> bool:
    """Check if we should stop retrying due to token limit issues."""
    if model is None:
        model = FLAGS.model
    
    token_info = check_token_limits(system_prompt, user_prompt, model)
    
    # Stop retrying if token limit is severely exceeded (more than 20% over limit)
    if token_info['excess_tokens'] > token_info['max_tokens'] * 0.2:
        print(f"ð Stopping retry due to severe token limit excess: {token_info['excess_tokens']} excess tokens")
        return True
    
    return False

# @retry(
#    retry=retry_if_exception_type((APITimeoutError, APIError, APIConnectionError, httpx.TimeoutException, httpx.ReadTimeout, InternalServerError)),
#    wait=wait_exponential(multiplier=2, min=3, max=60),
#    stop=stop_after_attempt(10),  # Increase attempts from 5 to 8
#)
def llm_inference(
    system_prompt: str,
    user_prompt: str,
    temperature: float = None,
    model_source: str = FLAGS.model_source,
    config_file: Optional[str] = None,
):
    # Import saver for tracking
    
    # Use provided temperature or default from FLAGS
    effective_temperature = temperature if temperature is not None else FLAGS.temperature
    
    # Check token limits before starting retries. For config-backed calls, use
    # the actual configured model when available instead of FLAGS.model.
    token_model = FLAGS.model
    if model_source in ("Config", "OpenAI", "Perflab"):
        try:
            token_model = load_model_config(config_file).get("model", FLAGS.model)
        except Exception:
            token_model = FLAGS.model
    token_info = check_token_limits(system_prompt, user_prompt, token_model)
    
    # Always print token usage for every inference
    print(f"ð¢ Token usage: {token_info['total_tokens']} tokens (system: {token_info['system_tokens']}, user: {token_info['user_tokens']}, limit: {token_info['max_tokens']})")
    if temperature is not None:
        print(f"ð¡ï¸ Using custom temperature: {effective_temperature}")
    
    # Proactive delay for large prompts that might cause timeouts
    if token_info['total_tokens'] > token_info['max_tokens'] * 0.3:  # If using more than 30% of tokens
        proactive_delay = min(5, token_info['total_tokens'] / 10000)  # Scale delay with token count
        print(f"â³ Large prompt detected ({token_info['total_tokens']} tokens), adding proactive delay: {proactive_delay:.1f}s")
        time.sleep(proactive_delay)
    
    # Track inference start time
    inference_start_time = time.time()
    
    # breakpoint()

    try:
        if model_source == 'Mark':
            print(f"Calling Mark LLM with model: {FLAGS.model}")
            response = call_Mark_llm(system_prompt, user_prompt, effective_temperature)
        elif model_source in ('Perflab', 'Config', 'OpenAI'):
            response = call_Perflab_llm(
                system_prompt,
                user_prompt,
                effective_temperature,
                config_file=config_file,
            )
        else:
            raise ValueError(f"Invalid model source: {FLAGS.model_source}")
            
        # Calculate inference time and log to saver
        inference_time = time.time() - inference_start_time
        
        return response
        
    except (APITimeoutError, httpx.TimeoutException, httpx.ReadTimeout) as e:
        # Calculate inference time even for failed calls
        inference_time = time.time() - inference_start_time
        print(f"â ï¸ LLM call failed after {inference_time:.2f}s")
        
        # Before retrying, check if it's a token limit issue
        if should_stop_on_token_limit(system_prompt, user_prompt):
            print(f"ð Stopping retry due to token limit issues")
            raise ValueError(f"Timeout likely due to token limit: {token_info['total_tokens']} tokens") from e
        else:
            # Log timeout-specific information
            print(f"â±ï¸ Timeout occurred - will retry with extended delay")
            print(f"   Prompt size: {len(system_prompt + user_prompt)} chars")
            print(f"   Token count: {token_info['total_tokens']}")
            print(f"   Error: {str(e)[:200]}...")
            # Let the retry mechanism handle it with extended delays
            raise


def test_inference_api():
    """Test the inference API with a simple query."""
    response = llm_inference(
        system_prompt="You are a helpful assistant.",
        user_prompt="Hello! Can you help me?",
        model_source="Perflab",
        temperature=0.7,
    )
    print("Test response:", response)

if __name__ == "__main__":
    # test_perflab()
    llm_inference(system_prompt="You are a helpful assistant.", user_prompt="What is the capital of France?")
    # call_dar_gateway(system_prompt="You are a helpful assistant.", user_prompt="What is the capital of France?")
    # call_Perflab_claude37(system_prompt="You are a helpful assistant.", user_prompt="What is the capital of France?")
    # call_NIM_openmodels(system_prompt="You are a helpful assistant.", user_prompt="What is the capital of France?")
    test_inference_api()
    call_Perflab_llm(system_prompt="You are a helpful assistant.", user_prompt="What is the capital of France?")
    #direct_call_Mark_gpt4o(system_prompt="You are a helpful assistant.", user_prompt="What is the capital of France?")
