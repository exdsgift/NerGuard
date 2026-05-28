"""
Core inference pipeline for NerGuard.

This module provides the main pipeline function for PII detection, combining
base model inference, regex validation, and optional LLM routing.
"""

from typing import Dict, List, Tuple
from src.core.constants import DEFAULT_MODEL_PATH
from src.core.pipeline_class import HybridPipeline

# Global cache to reuse the pipeline instance across calls
_pipeline_cache = {}

def redact_pipeline(
    text: str,
    model_path: str = DEFAULT_MODEL_PATH,
    llm_routing: bool = False,
    llm_source: str = "openai",
    llm_model: str = "gpt-4o",
    model=None,
    tokenizer=None,
    api_key: str = None,
) -> Tuple[List[Dict], str]:
    """
    Run the full hybrid pipeline on text and return entities + redacted text.
    Backward compatibility wrapper around HybridPipeline.
    """
    # Use cached pipeline if available
    cache_key = (model_path, llm_routing, llm_source, llm_model)
    if cache_key not in _pipeline_cache:
        _pipeline_cache[cache_key] = HybridPipeline(
            model_path=model_path,
            llm_routing=llm_routing,
            llm_source=llm_source,
            llm_model=llm_model,
            api_key=api_key
        )
        
    pipeline = _pipeline_cache[cache_key]
    
    # Overwrite cached model/tokenizer if explicitly provided (for benchmarks/tests)
    if model is not None:
        pipeline.model = model
    if tokenizer is not None:
        pipeline.tokenizer = tokenizer
        
    return pipeline.process_text(text)


def redact_pipeline_batch(
    texts: List[str],
    model_path: str = DEFAULT_MODEL_PATH,
    llm_routing: bool = False,
    llm_source: str = "openai",
    llm_model: str = "gpt-4o",
    api_key: str = None,
) -> List[Tuple[List[Dict], str]]:
    """
    Run the full hybrid pipeline on a batch of texts using async LLM routing.
    """
    import asyncio
    
    cache_key = (model_path, llm_routing, llm_source, llm_model)
    if cache_key not in _pipeline_cache:
        _pipeline_cache[cache_key] = HybridPipeline(
            model_path=model_path,
            llm_routing=llm_routing,
            llm_source=llm_source,
            llm_model=llm_model,
            api_key=api_key
        )
        
    pipeline = _pipeline_cache[cache_key]
    
    return asyncio.run(pipeline.process_batch_async(texts))
