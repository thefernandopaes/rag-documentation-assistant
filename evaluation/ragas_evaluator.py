"""
RAGAs Evaluator - RAG System Quality Evaluation using RAGAs

Evaluates RAG system using RAGAs metrics:
- Faithfulness: Answer is grounded in retrieved context
- Answer Relevancy: Answer addresses the question
- Context Recall: Retrieved all relevant context
- Context Precision: Retrieved context is relevant
- Answer Similarity: Answer matches ground truth
"""

import json
import logging
from typing import List, Dict, Any, Optional
import asyncio

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluate RAG system using RAGAs metrics.

    RAGAs (Retrieval-Augmented Generation Assessment) provides
    specialized metrics for evaluating RAG systems.
    """

    def __init__(self):
        """Initialize RAG evaluator"""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
                answer_similarity
            )

            self.evaluate_func = evaluate
            self.metrics = [
                faithfulness,          # Answer grounded in context
                answer_relevancy,      # Answer addresses question
                context_recall,        # Retrieved all relevant context
                context_precision,     # Retrieved context is relevant
                answer_similarity      # Answer matches ground truth
            ]

            logger.info("RAGAs evaluator initialized successfully")

        except ImportError as e:
            logger.warning(
                f"RAGAs not available: {e}. "
                "Install with: pip install ragas"
            )
            self.evaluate_func = None
            self.metrics = []

    async def evaluate_system(
        self,
        test_cases: List[Dict],
        rag_engine
    ) -> Dict:
        """
        Evaluate RAG system on test cases.

        Args:
            test_cases: List of test cases with:
                - question: str
                - ground_truth: str
                - contexts: List[str]
            rag_engine: AsyncRAGEngine instance

        Returns:
            Evaluation metrics dictionary
        """
        if not self.evaluate_func:
            return {
                'error': 'RAGAs not installed',
                'message': 'Install ragas with: pip install ragas>=0.2.0'
            }

        try:
            logger.info(f"Evaluating RAG system with {len(test_cases)} test cases")

            # Prepare data
            questions = []
            answers = []
            contexts = []
            ground_truths = []

            # Get answers from RAG for each question
            for case in test_cases:
                # Generate answer
                result = await rag_engine.generate_response(case['question'])

                questions.append(case['question'])
                answers.append(result['answer'])

                # Use retrieved context from RAG or fallback to test context
                if 'context' in result and result['context']:
                    contexts.append([result['context']])
                else:
                    contexts.append(case.get('contexts', ['No context available']))

                ground_truths.append(case['ground_truth'])

                logger.info(f"Processed: {case['question'][:50]}...")

            # Create dataset for RAGAs
            try:
                from datasets import Dataset

                dataset = Dataset.from_dict({
                    'question': questions,
                    'answer': answers,
                    'contexts': contexts,
                    'ground_truth': ground_truths
                })

                logger.info("Dataset created, running evaluation...")

                # Run evaluation
                results = self.evaluate_func(
                    dataset,
                    metrics=self.metrics
                )

                # Convert to dict
                results_dict = results.to_pandas().to_dict('records')[0] if len(results.to_pandas()) > 0 else {}

                logger.info(f"Evaluation complete: {results_dict}")

                return {
                    'success': True,
                    'test_cases_count': len(test_cases),
                    'metrics': {
                        'faithfulness': results_dict.get('faithfulness', 0.0),
                        'answer_relevancy': results_dict.get('answer_relevancy', 0.0),
                        'context_recall': results_dict.get('context_recall', 0.0),
                        'context_precision': results_dict.get('context_precision', 0.0),
                        'answer_similarity': results_dict.get('answer_similarity', 0.0)
                    },
                    'overall_score': self._calculate_overall_score(results_dict)
                }

            except ImportError:
                logger.error("datasets library not available. Install with: pip install datasets")
                return {
                    'error': 'datasets library not installed',
                    'message': 'Install with: pip install datasets'
                }

        except Exception as e:
            logger.error(f"Evaluation error: {e}", exc_info=True)
            return {
                'error': str(e),
                'message': 'Evaluation failed'
            }

    def load_test_cases(
        self,
        filepath: str = "evaluation/test_dataset.json"
    ) -> List[Dict]:
        """
        Load test cases from JSON file.

        Args:
            filepath: Path to test dataset JSON

        Returns:
            List of test cases
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            test_cases = data.get('test_cases', [])

            logger.info(f"Loaded {len(test_cases)} test cases from {filepath}")

            return test_cases

        except FileNotFoundError:
            logger.error(f"Test dataset not found: {filepath}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in test dataset: {e}")
            return []

    async def run_evaluation(
        self,
        rag_engine,
        test_dataset_path: Optional[str] = None
    ) -> Dict:
        """
        Run full evaluation on RAG system.

        Args:
            rag_engine: AsyncRAGEngine instance
            test_dataset_path: Optional custom path to test dataset

        Returns:
            Evaluation results
        """
        # Load test cases
        if test_dataset_path:
            test_cases = self.load_test_cases(test_dataset_path)
        else:
            test_cases = self.load_test_cases()

        if not test_cases:
            return {
                'error': 'No test cases found',
                'message': 'Load test cases failed'
            }

        # Run evaluation
        return await self.evaluate_system(test_cases, rag_engine)

    def _calculate_overall_score(self, results_dict: Dict) -> float:
        """
        Calculate overall score from individual metrics.

        Args:
            results_dict: Dictionary with metric scores

        Returns:
            Overall score (0-1)
        """
        metrics = [
            results_dict.get('faithfulness', 0.0),
            results_dict.get('answer_relevancy', 0.0),
            results_dict.get('context_recall', 0.0),
            results_dict.get('context_precision', 0.0),
            results_dict.get('answer_similarity', 0.0)
        ]

        # Filter out None values
        valid_metrics = [m for m in metrics if m is not None and m > 0]

        if not valid_metrics:
            return 0.0

        # Average of all metrics
        return round(sum(valid_metrics) / len(valid_metrics), 3)

    def get_metrics_description(self) -> Dict[str, str]:
        """
        Get description of each metric.

        Returns:
            Dictionary mapping metric names to descriptions
        """
        return {
            'faithfulness': (
                'Measures if the answer is factually consistent with the context. '
                'Score of 1 means answer is fully grounded in context.'
            ),
            'answer_relevancy': (
                'Measures how relevant the answer is to the question. '
                'Score of 1 means answer perfectly addresses the question.'
            ),
            'context_recall': (
                'Measures if all relevant information needed to answer was retrieved. '
                'Score of 1 means all necessary context was found.'
            ),
            'context_precision': (
                'Measures if retrieved context is relevant to the question. '
                'Score of 1 means no irrelevant context was retrieved.'
            ),
            'answer_similarity': (
                'Measures semantic similarity between answer and ground truth. '
                'Score of 1 means answer matches expected answer.'
            )
        }
