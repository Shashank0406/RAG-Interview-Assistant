"""
Evaluation System for RAG
Comprehensive metrics and testing framework for measuring RAG performance
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single query-response pair"""
    query: str
    response: str
    sources: List[Dict[str, Any]]
    metrics: Dict[str, float]
    overall_score: float
    grade: str  # A, B, C, D, F
    feedback: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "query": self.query,
            "response": self.response,
            "sources": self.sources,
            "metrics": self.metrics,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "feedback": self.feedback
        }


class RAGEvaluator:
    """Comprehensive evaluator for RAG system performance"""

    def __init__(self):
        self.metrics_calculators = {
            "answer_relevance": self._calculate_answer_relevance,
            "context_relevance": self._calculate_context_relevance,
            "factual_accuracy": self._calculate_factual_accuracy,
            "answer_completeness": self._calculate_answer_completeness,
            "response_quality": self._calculate_response_quality,
            "source_utilization": self._calculate_source_utilization
        }

    def evaluate_response(self,
                          query: str,
                          response: str,
                          sources: List[Dict[str, Any]],
                          ground_truth: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate a single query-response pair

        Args:
            query: The original question
            response: The generated answer
            sources: List of source documents used
            ground_truth: Optional ground truth answer for comparison

        Returns:
            EvaluationResult with detailed metrics
        """
        metrics = {}

        # Calculate all available metrics
        for metric_name, calculator in self.metrics_calculators.items():
            try:
                metrics[metric_name] = calculator(query, response, sources, ground_truth)
            except Exception as e:
                logger.error(f"Error calculating {metric_name}: {e}")
                metrics[metric_name] = 0.0

        # Calculate overall score (weighted average)
        weights = {
            "answer_relevance": 0.25,
            "context_relevance": 0.20,
            "factual_accuracy": 0.20,
            "answer_completeness": 0.15,
            "response_quality": 0.15,
            "source_utilization": 0.05
        }

        overall_score = sum(metrics.get(metric, 0.0) * weight for metric, weight in weights.items())

        # Determine grade
        grade = self._calculate_grade(overall_score)

        # Generate feedback
        feedback = self._generate_feedback(metrics, overall_score)

        return EvaluationResult(
            query=query,
            response=response,
            sources=sources,
            metrics=metrics,
            overall_score=round(overall_score, 2),
            grade=grade,
            feedback=feedback
        )

    def _calculate_answer_relevance(self, query: str, response: str, sources: List[Dict[str, Any]], ground_truth: Optional[str] = None) -> float:
        """Calculate how relevant the answer is to the query"""
        if not response.strip():
            return 0.0

        # Simple keyword overlap method
        query_words = set(self._preprocess_text(query).split())
        response_words = set(self._preprocess_text(response).split())

        if not query_words:
            return 0.5  # Neutral score if no query words

        overlap = len(query_words.intersection(response_words))
        relevance = overlap / len(query_words)

        # Boost score if response contains key question words
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which'}
        question_overlap = len(question_words.intersection(response_words))

        return min(1.0, relevance + (question_overlap * 0.1))

    def _calculate_context_relevance(self, query: str, response: str, sources: List[Dict[str, Any]], ground_truth: Optional[str] = None) -> float:
        """Calculate how relevant the retrieved context is to the query"""
        if not sources:
            return 0.0

        total_relevance = 0.0
        for source in sources:
            source_text = source.get('content', '')
            if source_text:
                # Calculate overlap between query and source
                query_words = set(self._preprocess_text(query).split())
                source_words = set(self._preprocess_text(source_text).split())

                if query_words:
                    overlap = len(query_words.intersection(source_words))
                    relevance = overlap / len(query_words)
                    total_relevance += relevance

        return min(1.0, total_relevance / len(sources)) if sources else 0.0

    def _calculate_factual_accuracy(self, query: str, response: str, sources: List[Dict[str, Any]], ground_truth: Optional[str] = None) -> float:
        """Calculate factual accuracy (simplified version)"""
        if not response.strip():
            return 0.0

        # Check for obvious factual contradictions or uncertainty markers
        uncertainty_markers = ['i think', 'maybe', 'perhaps', 'not sure', 'uncertain']
        contradiction_markers = ['however', 'but', 'although', 'despite', 'on the other hand']

        response_lower = response.lower()

        # Penalize for uncertainty (unless the question warrants it)
        uncertainty_penalty = sum(1 for marker in uncertainty_markers if marker in response_lower)

        # Check if sources support the claims (simplified)
        if sources:
            source_text = ' '.join([s.get('content', '') for s in sources])
            source_words = set(self._preprocess_text(source_text).split())
            response_words = set(self._preprocess_text(response).split())

            # Calculate support ratio
            supported_words = len(response_words.intersection(source_words))
            support_ratio = supported_words / len(response_words) if response_words else 0.0

            accuracy = support_ratio - (uncertainty_penalty * 0.1)
        else:
            accuracy = 0.5 - (uncertainty_penalty * 0.1)  # Neutral if no sources

        return max(0.0, min(1.0, accuracy))

    def _calculate_answer_completeness(self, query: str, response: str, sources: List[Dict[str, Any]], ground_truth: Optional[str] = None) -> float:
        """Calculate how complete the answer is"""
        if not response.strip():
            return 0.0

        response_length = len(response.split())
        min_expected_length = 10  # Minimum words for a complete answer

        # Length-based completeness
        length_score = min(1.0, response_length / 50)  # Expect ~50 words for complete answer

        # Check for comprehensive elements
        has_examples = any(word in response.lower() for word in ['example', 'instance', 'case', 'such as'])
        has_explanation = any(word in response.lower() for word in ['because', 'since', 'due to', 'therefore'])
        has_structure = len(response.split('.')) > 2  # Multiple sentences

        completeness_bonus = (has_examples + has_explanation + has_structure) * 0.1

        return min(1.0, length_score + completeness_bonus)

    def _calculate_response_quality(self, query: str, response: str, sources: List[Dict[str, Any]], ground_truth: Optional[str] = None) -> float:
        """Calculate overall response quality"""
        if not response.strip():
            return 0.0

        quality_score = 0.0

        # Grammar and structure (simplified)
        sentence_count = len(re.split(r'[.!?]+', response))
        avg_sentence_length = len(response.split()) / max(1, sentence_count)

        # Good sentence length (10-25 words)
        if 10 <= avg_sentence_length <= 25:
            quality_score += 0.3
        elif 5 <= avg_sentence_length <= 35:
            quality_score += 0.2

        # Readability markers
        if re.search(r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b', response.lower()):
            quality_score += 0.2

        # Avoid repetition
        words = response.lower().split()
        unique_words = set(words)
        uniqueness_ratio = len(unique_words) / len(words) if words else 0
        quality_score += uniqueness_ratio * 0.3

        # Professional tone
        informal_markers = ['kinda', 'sorta', 'like', 'you know', 'stuff', 'things']
        informal_penalty = sum(1 for marker in informal_markers if marker in response.lower())
        quality_score -= informal_penalty * 0.1

        return max(0.0, min(1.0, quality_score))

    def _calculate_source_utilization(self, query: str, response: str, sources: List[Dict[str, Any]], ground_truth: Optional[str] = None) -> float:
        """Calculate how well sources are utilized"""
        if not sources:
            return 0.0

        # Check if response references source content
        source_text = ' '.join([s.get('content', '') for s in sources])
        source_words = set(self._preprocess_text(source_text).split())
        response_words = set(self._preprocess_text(response).split())

        # Calculate utilization ratio
        utilized_words = len(response_words.intersection(source_words))
        utilization_ratio = utilized_words / len(response_words) if response_words else 0.0

        # Bonus for using multiple sources
        source_bonus = min(0.2, len(sources) * 0.05)

        return min(1.0, utilization_ratio + source_bonus)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _calculate_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

    def _generate_feedback(self, metrics: Dict[str, float], overall_score: float) -> List[str]:
        """Generate human-readable feedback"""
        feedback = []

        # Overall feedback
        if overall_score >= 0.8:
            feedback.append("Excellent response with strong relevance and quality!")
        elif overall_score >= 0.7:
            feedback.append("Good response with solid performance across metrics.")
        elif overall_score >= 0.6:
            feedback.append("Decent response but could use improvement in some areas.")
        else:
            feedback.append("Response needs significant improvement.")

        # Specific feedback
        if metrics.get("answer_relevance", 0) < 0.6:
            feedback.append("Consider making the answer more directly relevant to the question.")

        if metrics.get("context_relevance", 0) < 0.6:
            feedback.append("The retrieved sources may not be optimally relevant to the query.")

        if metrics.get("factual_accuracy", 0) < 0.7:
            feedback.append("Verify factual claims and ensure they align with source material.")

        if metrics.get("answer_completeness", 0) < 0.6:
            feedback.append("Provide more comprehensive answers with examples and explanations.")

        if metrics.get("response_quality", 0) < 0.7:
            feedback.append("Focus on clear, well-structured responses with good grammar.")

        if metrics.get("source_utilization", 0) < 0.5:
            feedback.append("Better utilize the available source material in your response.")

        return feedback

    def evaluate_dataset(self,
                        test_cases: List[Dict[str, Any]],
                        rag_system=None) -> Dict[str, Any]:
        """
        Evaluate the RAG system on a dataset of test cases

        Args:
            test_cases: List of dicts with 'query', 'expected_answer' (optional)
            rag_system: RAG system instance to test

        Returns:
            Comprehensive evaluation report
        """
        results = []
        total_score = 0.0

        for i, test_case in enumerate(test_cases):
            query = test_case["query"]
            ground_truth = test_case.get("expected_answer")

            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}: {query[:50]}...")

            try:
                if rag_system:
                    # Get response from RAG system
                    response_data = rag_system.query(query)
                    response = response_data.get("answer", "")
                    sources = response_data.get("sources", [])
                else:
                    # Use provided response
                    response = test_case.get("response", "")
                    sources = test_case.get("sources", [])

                # Evaluate
                result = self.evaluate_response(query, response, sources, ground_truth)
                results.append(result)
                total_score += result.overall_score

            except Exception as e:
                logger.error(f"Error evaluating test case '{query}': {e}")
                # Add failed result
                failed_result = EvaluationResult(
                    query=query,
                    response="",
                    sources=[],
                    metrics={},
                    overall_score=0.0,
                    grade="F",
                    feedback=[f"Evaluation failed: {str(e)}"]
                )
                results.append(failed_result)

        # Calculate aggregate statistics
        if results:
            scores = [r.overall_score for r in results]
            avg_score = statistics.mean(scores)
            median_score = statistics.median(scores)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

            # Grade distribution
            grades = defaultdict(int)
            for result in results:
                grades[result.grade] += 1

            # Metric averages
            metric_averages = {}
            for metric_name in self.metrics_calculators.keys():
                metric_scores = [r.metrics.get(metric_name, 0.0) for r in results]
                metric_averages[metric_name] = round(statistics.mean(metric_scores), 3)

        else:
            avg_score = median_score = std_dev = 0.0
            grades = {}
            metric_averages = {}

        return {
            "summary": {
                "total_test_cases": len(test_cases),
                "successful_evaluations": len(results),
                "average_score": round(avg_score, 3),
                "median_score": round(median_score, 3),
                "score_std_dev": round(std_dev, 3),
                "grade_distribution": dict(grades)
            },
            "metric_averages": metric_averages,
            "detailed_results": [r.to_dict() for r in results],
            "recommendations": self._generate_recommendations(avg_score, metric_averages)
        }

    def _generate_recommendations(self, avg_score: float, metric_averages: Dict[str, float]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []

        if avg_score < 0.6:
            recommendations.append("Overall performance needs significant improvement. Consider reviewing the RAG pipeline.")

        # Check specific metrics
        if metric_averages.get("answer_relevance", 0) < 0.7:
            recommendations.append("Improve query understanding and response relevance.")

        if metric_averages.get("context_relevance", 0) < 0.7:
            recommendations.append("Optimize document retrieval to find more relevant sources.")

        if metric_averages.get("factual_accuracy", 0) < 0.8:
            recommendations.append("Implement fact-checking mechanisms and improve source alignment.")

        if metric_averages.get("answer_completeness", 0) < 0.6:
            recommendations.append("Encourage more comprehensive answers with examples and explanations.")

        if metric_averages.get("response_quality", 0) < 0.7:
            recommendations.append("Focus on response clarity, structure, and professional tone.")

        if not recommendations:
            recommendations.append("Performance is strong! Consider fine-tuning specific areas for even better results.")

        return recommendations


# Test dataset for evaluation
DEFAULT_TEST_CASES = [
    {
        "query": "What is machine learning?",
        "expected_answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
    },
    {
        "query": "Explain the difference between supervised and unsupervised learning",
        "expected_answer": "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data."
    },
    {
        "query": "What are the main components of a neural network?",
        "expected_answer": "Neural networks consist of input layers, hidden layers, output layers, neurons, weights, and activation functions."
    },
    {
        "query": "How does RAG work?",
        "expected_answer": "RAG combines retrieval from a knowledge base with generative AI to provide more accurate and up-to-date responses."
    },
    {
        "query": "What are some prompt engineering techniques?",
        "expected_answer": "Prompt engineering techniques include few-shot learning, chain-of-thought prompting, and providing clear context and examples."
    }
]


# Convenience functions
def evaluate_response(query: str, response: str, sources: List[Dict[str, Any]]) -> EvaluationResult:
    """Convenience function for quick evaluation"""
    evaluator = RAGEvaluator()
    return evaluator.evaluate_response(query, response, sources)


def evaluate_rag_system(rag_system, test_cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Convenience function to evaluate an entire RAG system"""
    evaluator = RAGEvaluator()
    test_cases = test_cases or DEFAULT_TEST_CASES
    return evaluator.evaluate_dataset(test_cases, rag_system)