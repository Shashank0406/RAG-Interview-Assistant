"""
Model Comparison System for RAG
Compare different LLM providers and models side by side
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result from a single model query"""
    model_name: str
    provider: str
    response: str
    response_time: float
    confidence_score: float
    sources_used: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "response": self.response,
            "response_time": self.response_time,
            "confidence_score": self.confidence_score,
            "sources_used": self.sources_used,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class ComparisonResult:
    """Results from comparing multiple models"""
    query: str
    timestamp: str
    model_results: List[ModelResult]
    winner: Optional[str] = None
    comparison_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.comparison_metrics is None:
            self.comparison_metrics = {}


class ModelComparator:
    """Compare different LLM models on the same queries"""

    def __init__(self, rag_system):
        """
        Initialize comparator with a RAG system

        Args:
            rag_system: Instance of RAGSystem
        """
        self.rag_system = rag_system
        self.available_models = self._get_available_models()

    def _get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models from the RAG system"""
        models = []

        # OpenAI models
        if self.rag_system.llm_manager.providers.get('openai'):
            models.extend([
                {"provider": "openai", "model": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
                {"provider": "openai", "model": "gpt-4", "name": "GPT-4"},
                {"provider": "openai", "model": "gpt-4-turbo", "name": "GPT-4 Turbo"}
            ])

        # Anthropic models
        if self.rag_system.llm_manager.providers.get('anthropic'):
            models.extend([
                {"provider": "anthropic", "model": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
                {"provider": "anthropic", "model": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
                {"provider": "anthropic", "model": "claude-3-opus-20240229", "name": "Claude 3 Opus"}
            ])

        return models

    def compare_models(self,
                      query: str,
                      models_to_compare: Optional[List[str]] = None,
                      max_workers: int = 3) -> ComparisonResult:
        """
        Compare multiple models on the same query

        Args:
            query: The question to ask
            models_to_compare: List of model names to compare (optional)
            max_workers: Maximum number of concurrent requests

        Returns:
            ComparisonResult with all model responses
        """
        import datetime

        # Filter models to compare
        if models_to_compare:
            models = [m for m in self.available_models if m["name"] in models_to_compare]
        else:
            models = self.available_models[:3]  # Default to first 3 models

        if not models:
            raise ValueError("No models available for comparison")

        logger.info(f"Comparing {len(models)} models on query: {query[:50]}...")

        # Query models concurrently
        model_results = self._query_models_concurrent(query, models, max_workers)

        # Create comparison result
        result = ComparisonResult(
            query=query,
            timestamp=datetime.datetime.now().isoformat(),
            model_results=model_results
        )

        # Analyze results
        result.winner = self._determine_winner(model_results)
        result.comparison_metrics = self._calculate_comparison_metrics(model_results)

        return result

    def _query_models_concurrent(self,
                                query: str,
                                models: List[Dict[str, str]],
                                max_workers: int) -> List[ModelResult]:
        """Query multiple models concurrently"""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries
            future_to_model = {
                executor.submit(self._query_single_model, query, model): model
                for model in models
            }

            # Collect results as they complete
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error querying {model['name']}: {e}")
                    # Add error result
                    results.append(ModelResult(
                        model_name=model["name"],
                        provider=model["provider"],
                        response="",
                        response_time=0.0,
                        confidence_score=0.0,
                        sources_used=0,
                        error=str(e)
                    ))

        return results

    def _query_single_model(self, query: str, model: Dict[str, str]) -> ModelResult:
        """Query a single model"""
        start_time = time.time()

        try:
            # Query the RAG system with specific provider
            response = self.rag_system.query(
                question=query,
                provider=model["provider"],
                k=3  # Use fewer sources for comparison
            )

            response_time = time.time() - start_time

            return ModelResult(
                model_name=model["name"],
                provider=model["provider"],
                response=response.get("answer", ""),
                response_time=round(response_time, 2),
                confidence_score=response.get("confidence", 0.0),
                sources_used=len(response.get("sources", [])),
                metadata={"full_response": response}
            )

        except Exception as e:
            response_time = time.time() - start_time
            return ModelResult(
                model_name=model["name"],
                provider=model["provider"],
                response="",
                response_time=round(response_time, 2),
                confidence_score=0.0,
                sources_used=0,
                error=str(e)
            )

    def _determine_winner(self, results: List[ModelResult]) -> Optional[str]:
        """Determine the winning model based on various criteria"""
        if not results:
            return None

        # Filter out failed results
        successful_results = [r for r in results if not r.error]

        if not successful_results:
            return None

        # Simple scoring: response length + confidence - response time penalty
        best_score = -float('inf')
        winner = None

        for result in successful_results:
            # Score = response quality + speed bonus - time penalty
            response_length = len(result.response)
            confidence_bonus = result.confidence_score * 100
            time_penalty = result.response_time * 2  # Penalize slow responses

            score = response_length + confidence_bonus - time_penalty

            if score > best_score:
                best_score = score
                winner = result.model_name

        return winner

    def _calculate_comparison_metrics(self, results: List[ModelResult]) -> Dict[str, Any]:
        """Calculate comparison metrics across all results"""
        if not results:
            return {}

        successful_results = [r for r in results if not r.error]
        failed_results = [r for r in results if r.error]

        metrics = {
            "total_models": len(results),
            "successful_queries": len(successful_results),
            "failed_queries": len(failed_results),
            "average_response_time": 0.0,
            "average_confidence": 0.0,
            "average_sources_used": 0.0,
            "response_lengths": {},
            "provider_breakdown": {}
        }

        if successful_results:
            metrics["average_response_time"] = round(
                sum(r.response_time for r in successful_results) / len(successful_results), 2
            )
            metrics["average_confidence"] = round(
                sum(r.confidence_score for r in successful_results) / len(successful_results), 2
            )
            metrics["average_sources_used"] = round(
                sum(r.sources_used for r in successful_results) / len(successful_results), 2
            )

            # Response lengths by model
            metrics["response_lengths"] = {
                r.model_name: len(r.response) for r in successful_results
            }

            # Provider breakdown
            providers = {}
            for r in successful_results:
                if r.provider not in providers:
                    providers[r.provider] = {"count": 0, "avg_time": 0.0}
                providers[r.provider]["count"] += 1
                providers[r.provider]["avg_time"] += r.response_time

            for provider in providers:
                providers[provider]["avg_time"] /= providers[provider]["count"]
                providers[provider]["avg_time"] = round(providers[provider]["avg_time"], 2)

            metrics["provider_breakdown"] = providers

        return metrics

    def benchmark_models(self,
                        queries: List[str],
                        models_to_compare: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark across multiple queries

        Args:
            queries: List of queries to test
            models_to_compare: Models to include in benchmark

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Starting benchmark with {len(queries)} queries")

        all_results = []
        start_time = time.time()

        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            try:
                result = self.compare_models(query, models_to_compare)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error benchmarking query '{query}': {e}")

        total_time = time.time() - start_time

        # Aggregate results
        benchmark_results = {
            "total_queries": len(queries),
            "successful_queries": len(all_results),
            "total_time": round(total_time, 2),
            "average_time_per_query": round(total_time / len(queries), 2) if queries else 0,
            "results": [result.__dict__ for result in all_results],
            "overall_metrics": self._calculate_overall_benchmark_metrics(all_results)
        }

        return benchmark_results

    def _calculate_overall_benchmark_metrics(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Calculate overall metrics across all benchmark results"""
        if not results:
            return {}

        # Aggregate winners
        winners = {}
        total_comparisons = 0

        for result in results:
            if result.winner:
                winners[result.winner] = winners.get(result.winner, 0) + 1
                total_comparisons += 1

        # Find most consistent winner
        most_wins = max(winners.values()) if winners else 0
        consistent_winner = next(
            (model for model, wins in winners.items() if wins == most_wins),
            None
        )

        return {
            "total_comparisons": total_comparisons,
            "unique_winners": len(winners),
            "most_consistent_winner": consistent_winner,
            "winner_distribution": winners,
            "win_percentage": {
                model: round((wins / total_comparisons) * 100, 1)
                for model, wins in winners.items()
            }
        }

    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models for comparison"""
        return self.available_models.copy()

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about available models"""
        providers = {}
        for model in self.available_models:
            provider = model["provider"]
            if provider not in providers:
                providers[provider] = 0
            providers[provider] += 1

        return {
            "total_models": len(self.available_models),
            "providers": providers,
            "models": [m["name"] for m in self.available_models]
        }


# Convenience functions
def create_model_comparator(rag_system) -> ModelComparator:
    """Create a model comparator instance"""
    return ModelComparator(rag_system)


def compare_models_on_query(rag_system, query: str, models: Optional[List[str]] = None) -> ComparisonResult:
    """Convenience function for quick model comparison"""
    comparator = ModelComparator(rag_system)
    return comparator.compare_models(query, models)