#!/usr/bin/env python3
"""
vLLM Server Deployment Script for RD Rating System
"""

import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path
from argparse import Namespace
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import yaml
from vllm.entrypoints.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs

def load_config():
    """Load configuration from YAML file."""
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config):
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file', 'logs/rd_rating.log')
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start vLLM server for RD Rating System")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt2",
        help="Model name or path"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--max-model-len", 
        type=int, 
        default=1024,
        help="Maximum model length"
    )
    parser.add_argument(
        "--gpu-memory-utilization", 
        type=float, 
        default=0.9,
        help="GPU memory utilization"
    )
    parser.add_argument(
        "--trust-remote-code", 
        action="store_true",
        help="Trust remote code when loading model"
    )
    return parser.parse_args()

def main():
    """Main function to start the vLLM server."""
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    args = parse_arguments()
    
    # Override with config values if not provided via command line
    server_config = config.get('server', {})
    if args.host == "0.0.0.0":
        args.host = server_config.get('host', args.host)
    if args.port == 8000:
        args.port = server_config.get('port', args.port)
    if args.tensor_parallel_size == 1:
        args.tensor_parallel_size = server_config.get('tensor_parallel_size', args.tensor_parallel_size)
    if args.max_model_len == 1024:
        args.max_model_len = server_config.get('max_model_len', args.max_model_len)
    if args.gpu_memory_utilization == 0.9:
        args.gpu_memory_utilization = server_config.get('gpu_memory_utilization', args.gpu_memory_utilization)
    
    logger.info("Starting vLLM server for RD Rating System")
    logger.info(f"Model: {args.model}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    logger.info(f"Max Model Length: {args.max_model_len}")
    logger.info(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    
    try:
        # Set environment variables for vLLM
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        
        # Manually construct Namespace for vLLM run_server
        vllm_parsed_args = Namespace(
            host=args.host,
            port=args.port,
            model=args.model,
            served_model_name=args.model,
            tokenizer=args.model,
            hf_config_path=None,
            task="auto",
            skip_tokenizer_init=False,
            enable_prompt_embeds=False,
            tokenizer_mode="auto",
            trust_remote_code=args.trust_remote_code,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            log_level="info",
            dtype="auto",
            quantization=None,
            revision=None,
            code_revision=None,
            rope_scaling={},
            rope_theta=None,
            hf_token=None,
            hf_overrides={},
            tokenizer_revision=None,
            enforce_eager=False,
            max_seq_len_to_capture=0,
            disable_custom_all_reduce=False,
            tokenizer_pool_size=None,
            tokenizer_pool_type=None,
            tokenizer_pool_extra_config={},
            limit_mm_per_prompt={},
            mm_processor_kwargs={},
            disable_mm_preprocessor_cache=False,
            enable_lora=False,
            enable_lora_bias=False,
            max_loras=0,
            max_lora_rank=0,
            fully_sharded_loras=False,
            max_cpu_loras=0,
            lora_dtype=None,
            long_lora_scaling_factors=None,
            enable_prompt_adapter=False,
            max_prompt_adapters=0,
            max_prompt_adapter_token=0,
            device=None,
            num_scheduler_steps=1,
            multi_step_stream_outputs=False,
            ray_workers_use_nsight=False,
            num_gpu_blocks_override=0,
            num_lookahead_slots=0,
            model_loader_extra_config={},
            ignore_patterns=None,
            preemption_mode=None,
            scheduler_delay_factor=0.0,
            scheduling_policy="fcfs",
            scheduler_cls="",
            enable_chunked_prefill=False,
            disable_chunked_mm_input=False,
            disable_hybrid_kv_cache_manager=False,
            guided_decoding_backend="auto",
            guided_decoding_disable_fallback=False,
            guided_decoding_disable_any_whitespace=False,
            guided_decoding_disable_additional_properties=False,
            reasoning_parser="",
            logits_processor_pattern=None,
            speculative_config=None,
            qlora_adapter_name_or_path=None,
            show_hidden_metrics_for_version=None,
            otlp_traces_endpoint=None,
            collect_detailed_traces=[],
            disable_async_output_proc=False,
            override_neuron_config={},
            override_pooler_config=None,
            worker_cls="",
            worker_extension_cls="",
            kv_transfer_config=None,
            kv_events_config=None,
            generation_config="",
            enable_sleep_mode=False,
            override_generation_config={},
            model_impl="auto",
            calculate_kv_scales=False,
            additional_config={},
            enable_reasoning=False,
            use_tqdm_on_load=False,
            pt_load_map_location={},
            enable_multimodal_encoder_data_parallel=False,
            disable_log_requests=False,
            lora_extra_vocab_size=0,
            seed=0,
            download_dir=None,
            load_format="auto",
            config_format="auto",
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_size_local=0,
            data_parallel_address=None,
            data_parallel_rpc_port=0,
            data_parallel_backend="mp",
            enable_expert_parallel=False,
            max_parallel_loading_workers=0,
            block_size=16,
            swap_space=4.0,
            cpu_offload_gb=0.0,
            max_num_batched_tokens=8192,
            max_num_seqs=0,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=0,
            max_logprobs=0,
            disable_log_stats=False,
            enable_prefix_caching=None,
            prefix_caching_hash_algo="builtin",
            disable_sliding_window=False,
            disable_cascade_attn=False,
            use_v2_block_manager=True,
            root_path=None,
            enable_ssl_refresh=False,
            ssl_keyfile=None,
            ssl_certfile=None,
            ssl_ca_certs=None,
            ssl_cert_reqs=None,
            allowed_local_media_path="",
            kv_cache_dtype="auto",
            cuda_graph_sizes=[],
            distributed_executor_backend=None,
            compilation_config=None,
        )
        
        # Start vLLM server
        asyncio.run(run_server(vllm_parsed_args))
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 