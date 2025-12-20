"""
Custom Metrics - Additional Metrics for RAG Evaluation

Provides custom metrics beyond RAGAs for comprehensive RAG evaluation.
"""

import logging
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class CustomMetrics:
    """
    Custom metrics for RAG evaluation.

    Provides additional metrics beyond RAGAs:
    - MRR (Mean Reciprocal Rank)
    - NDCG (Normalized Discounted Cumulative Gain)
    - Precision@K
    - Recall@K
    """

    @staticmethod
    def calculate_mrr(rankings: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank.

        MRR measures how quickly the first relevant result appears.

        Args:
            rankings: List of ranks where relevant doc appeared (1-indexed)
                     Example: [1, 3, 2] means first query had relevant doc at rank 1

        Returns:
            MRR score (0-1, higher is better)
        """
        if not rankings:
            return 0.0

        reciprocals = [1/r for r in rankings if r > 0]

        if not reciprocals:
            return 0.0

        mrr = np.mean(reciprocals)

        logger.debug(f"MRR calculated: {mrr:.3f} from {len(rankings)} rankings")

        return round(mrr, 3)

    @staticmethod
    def calculate_ndcg(relevance_scores: List[float], k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.

        NDCG measures the quality of ranking considering relevance scores.

        Args:
            relevance_scores: List of relevance scores (0-1) in ranked order
            k: Number of top results to consider

        Returns:
            NDCG score (0-1, higher is better)
        """
        if not relevance_scores:
            return 0.0

        scores = relevance_scores[:k]

        # DCG (Discounted Cumulative Gain)
        dcg = sum(
            (2**score - 1) / np.log2(i + 2)
            for i, score in enumerate(scores)
        )

        # IDCG (Ideal DCG) - best possible ranking
        ideal_scores = sorted(scores, reverse=True)
        idcg = sum(
            (2**score - 1) / np.log2(i + 2)
            for i, score in enumerate(ideal_scores)
        )

        if idcg == 0:
            return 0.0

        ndcg = dcg / idcg

        logger.debug(f"NDCG@{k} calculated: {ndcg:.3f}")

        return round(ndcg, 3)

    @staticmethod
    def calculate_precision_at_k(relevant: List[bool], k: int) -> float:
        """
        Calculate Precision@K.

        Precision@K measures the proportion of relevant results in top K.

        Args:
            relevant: List of boolean flags indicating relevance
            k: Number of top results to consider

        Returns:
            Precision score (0-1, higher is better)
        """
        if k <= 0 or not relevant:
            return 0.0

        top_k = relevant[:k]

        precision = sum(top_k) / k

        logger.debug(f"Precision@{k} calculated: {precision:.3f}")

        return round(precision, 3)

    @staticmethod
    def calculate_recall_at_k(
        relevant: List[bool],
        total_relevant: int,
        k: int
    ) -> float:
        """
        Calculate Recall@K.

        Recall@K measures the proportion of total relevant docs found in top K.

        Args:
            relevant: List of boolean flags indicating relevance
            total_relevant: Total number of relevant documents
            k: Number of top results to consider

        Returns:
            Recall score (0-1, higher is better)
        """
        if k <= 0 or total_relevant <= 0 or not relevant:
            return 0.0

        top_k = relevant[:k]

        recall = sum(top_k) / total_relevant

        logger.debug(f"Recall@{k} calculated: {recall:.3f}")

        return round(recall, 3)

    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """
        Calculate F1 Score (harmonic mean of precision and recall).

        Args:
            precision: Precision score
            recall: Recall score

        Returns:
            F1 score (0-1, higher is better)
        """
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)

        logger.debug(f"F1 Score calculated: {f1:.3f}")

        return round(f1, 3)

    @staticmethod
    def calculate_map(rankings: List[List[bool]]) -> float:
        """
        Calculate Mean Average Precision.

        MAP is the mean of average precision scores for each query.

        Args:
            rankings: List of lists, each containing boolean relevance flags
                     Example: [[True, False, True], [True, True, False]]

        Returns:
            MAP score (0-1, higher is better)
        """
        if not rankings:
            return 0.0

        average_precisions = []

        for ranking in rankings:
            if not ranking:
                continue

            relevant_count = 0
            precision_sum = 0.0

            for i, is_relevant in enumerate(ranking):
                if is_relevant:
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    precision_sum += precision_at_i

            if relevant_count > 0:
                avg_precision = precision_sum / relevant_count
                average_precisions.append(avg_precision)

        if not average_precisions:
            return 0.0

        map_score = np.mean(average_precisions)

        logger.debug(f"MAP calculated: {map_score:.3f}")

        return round(map_score, 3)


class LatencyMetrics:
    """
    Metrics for measuring response latency.
    """

    @staticmethod
    def calculate_statistics(latencies: List[float]) -> Dict[str, float]:
        """
        Calculate latency statistics.

        Args:
            latencies: List of latency values in seconds

        Returns:
            Dictionary with latency statistics
        """
        if not latencies:
            return {
                'avg': 0.0,
                'median': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'min': 0.0,
                'max': 0.0
            }

        sorted_latencies = sorted(latencies)

        return {
            'avg': round(np.mean(sorted_latencies), 2),
            'median': round(np.median(sorted_latencies), 2),
            'p50': round(np.percentile(sorted_latencies, 50), 2),
            'p95': round(np.percentile(sorted_latencies, 95), 2),
            'p99': round(np.percentile(sorted_latencies, 99), 2),
            'min': round(min(sorted_latencies), 2),
            'max': round(max(sorted_latencies), 2)
        }


class TokenMetrics:
    """
    Metrics for tracking token usage and costs.
    """

    # Token costs per 1K tokens (update based on current pricing)
    TOKEN_COSTS = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015}
    }

    @staticmethod
    def calculate_cost(
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate estimated cost for token usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        if model not in TokenMetrics.TOKEN_COSTS:
            model = 'gpt-4'  # Default to GPT-4 pricing

        pricing = TokenMetrics.TOKEN_COSTS[model]

        cost = (
            (input_tokens * pricing['input'] / 1000) +
            (output_tokens * pricing['output'] / 1000)
        )

        return round(cost, 4)

    @staticmethod
    def calculate_token_efficiency(
        answer_length: int,
        context_length: int
    ) -> float:
        """
        Calculate token efficiency (ratio of answer to context).

        Lower is better (means less context needed for answer).

        Args:
            answer_length: Length of answer in tokens
            context_length: Length of context in tokens

        Returns:
            Efficiency ratio
        """
        if context_length == 0:
            return 0.0

        efficiency = answer_length / context_length

        return round(efficiency, 3)


class SemanticMetrics:
    """
    Metrics for semantic similarity.
    """

    @staticmethod
    def calculate_embedding_similarity(
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity (0-1, higher is better)
        """
        if not embedding1 or not embedding2:
            return 0.0

        # Reshape for sklearn
        emb1 = np.array(embedding1).reshape(1, -1)
        emb2 = np.array(embedding2).reshape(1, -1)

        similarity = cosine_similarity(emb1, emb2)[0][0]

        return round(float(similarity), 3)


class QualityMetrics:
    """
    Overall quality metrics.
    """

    @staticmethod
    def calculate_overall_score(metrics: Dict[str, float]) -> float:
        """
        Calculate weighted overall quality score.

        Args:
            metrics: Dictionary with individual metric scores

        Returns:
            Overall score (0-1, higher is better)
        """
        # Define weights for different metrics
        weights = {
            'faithfulness': 0.25,
            'answer_relevancy': 0.25,
            'context_recall': 0.20,
            'context_precision': 0.15,
            'answer_similarity': 0.15
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in metrics and metrics[metric] is not None:
                weighted_sum += metrics[metric] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        overall = weighted_sum / total_weight

        return round(overall, 3)

    @staticmethod
    def categorize_score(score: float) -> str:
        """
        Categorize score into quality bands.

        Args:
            score: Score (0-1)

        Returns:
            Quality category
        """
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        elif score >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
