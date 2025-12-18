# DEV_5: RAG Evaluation

**Desenvolvedor:** DEV_5
**Fase:** 9E - RAG Evaluation (Bonus)
**Prioridade:** ⭐ BAIXA (Nice to have)
**Estimativa:** 2-3 horas
**Dependências:** DEV_1 (agent precisa estar funcionando)

---

## 🎯 Objetivo

Implementar **framework de avaliação de qualidade RAG** usando RAGAs para medir e melhorar performance do sistema.

---

## 📦 Entregas

1. **`evaluation/ragas_evaluator.py`** - RAGAs integration
2. **`evaluation/metrics.py`** - Custom metrics
3. **`evaluation/benchmark.py`** - Benchmark suite
4. **`evaluation/test_dataset.json`** - Test cases
5. **`test_evaluation.py`** - Tests

---

## 📝 Implementação

### 1. Install RAGAs

```bash
pip install ragas>=0.2.0
```

### 2. `evaluation/test_dataset.json`

```json
{
  "test_cases": [
    {
      "question": "How do I create a FastAPI endpoint?",
      "ground_truth": "Use the @app decorator with an HTTP method (get, post, etc.) and define an async function",
      "contexts": ["FastAPI uses Python decorators to define routes..."]
    },
    {
      "question": "What is async/await in Python?",
      "ground_truth": "async/await is Python's syntax for asynchronous programming",
      "contexts": ["Asynchronous programming allows non-blocking I/O..."]
    },
    {
      "question": "How to handle CORS in FastAPI?",
      "ground_truth": "Use CORSMiddleware from fastapi.middleware.cors",
      "contexts": ["CORS (Cross-Origin Resource Sharing) is handled via middleware..."]
    }
  ]
}
```

### 3. `evaluation/ragas_evaluator.py`

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity
)
from datasets import Dataset
import json
from typing import List, Dict
import asyncio

class RAGEvaluator:
    """Evaluate RAG system using RAGAs metrics"""

    def __init__(self):
        self.metrics = [
            faithfulness,          # Answer is grounded in context
            answer_relevancy,      # Answer addresses the question
            context_recall,        # Retrieved all relevant context
            context_precision,     # Retrieved context is relevant
            answer_similarity      # Answer matches ground truth
        ]

    async def evaluate_system(
        self,
        test_cases: List[Dict],
        rag_engine
    ) -> Dict:
        """
        Evaluate RAG system on test cases.

        Args:
            test_cases: List of {question, ground_truth, contexts}
            rag_engine: AsyncRAGEngine instance

        Returns:
            Evaluation metrics
        """
        # Prepare data
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for case in test_cases:
            # Get answer from RAG
            result = await rag_engine.generate_response(case['question'])

            questions.append(case['question'])
            answers.append(result['answer'])
            contexts.append([result.get('context', '')])
            ground_truths.append(case['ground_truth'])

        # Create dataset
        dataset = Dataset.from_dict({
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        })

        # Evaluate
        results = evaluate(
            dataset,
            metrics=self.metrics
        )

        return results.to_pandas().to_dict()

    def load_test_cases(self, filepath: str = "evaluation/test_dataset.json") -> List[Dict]:
        """Load test cases from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['test_cases']

    async def run_evaluation(self, rag_engine) -> Dict:
        """Run full evaluation"""
        test_cases = self.load_test_cases()
        return await self.evaluate_system(test_cases, rag_engine)
```

### 4. `evaluation/metrics.py`

```python
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CustomMetrics:
    """Custom metrics for RAG evaluation"""

    @staticmethod
    def calculate_mrr(rankings: List[int]) -> float:
        """
        Mean Reciprocal Rank.

        Args:
            rankings: List of ranks where relevant doc appeared (1-indexed)

        Returns:
            MRR score
        """
        reciprocals = [1/r for r in rankings if r > 0]
        return np.mean(reciprocals) if reciprocals else 0.0

    @staticmethod
    def calculate_ndcg(relevance_scores: List[float], k: int = 5) -> float:
        """
        Normalized Discounted Cumulative Gain.

        Args:
            relevance_scores: List of relevance scores (0-1)
            k: Number of top results to consider

        Returns:
            NDCG score
        """
        scores = relevance_scores[:k]

        # DCG
        dcg = sum(
            (2**score - 1) / np.log2(i + 2)
            for i, score in enumerate(scores)
        )

        # IDCG (ideal)
        ideal_scores = sorted(scores, reverse=True)
        idcg = sum(
            (2**score - 1) / np.log2(i + 2)
            for i, score in enumerate(ideal_scores)
        )

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_precision_at_k(relevant: List[bool], k: int) -> float:
        """Precision@K"""
        return sum(relevant[:k]) / k if k > 0 else 0.0

    @staticmethod
    def calculate_recall_at_k(relevant: List[bool], total_relevant: int, k: int) -> float:
        """Recall@K"""
        return sum(relevant[:k]) / total_relevant if total_relevant > 0 else 0.0
```

### 5. `evaluation/benchmark.py`

```python
import asyncio
import time
import json
from typing import Dict
from datetime import datetime

class RAGBenchmark:
    """Benchmark RAG system performance and quality"""

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.evaluator = RAGEvaluator()

    async def run_benchmark(self) -> Dict:
        """Run complete benchmark"""
        print("=" * 60)
        print("RAG SYSTEM BENCHMARK")
        print("=" * 60)

        results = {
            'timestamp': datetime.now().isoformat(),
            'performance': await self._benchmark_performance(),
            'quality': await self._benchmark_quality(),
            'retrieval': await self._benchmark_retrieval()
        }

        # Save results
        with open(f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2)

        self._print_summary(results)

        return results

    async def _benchmark_performance(self) -> Dict:
        """Benchmark query performance"""
        print("\n1. PERFORMANCE BENCHMARK")
        print("-" * 60)

        test_queries = [
            "What is FastAPI?",
            "How to create an API endpoint?",
            "What is async/await?",
            "How to handle CORS?",
            "What is Pydantic?"
        ]

        times = []
        for query in test_queries:
            start = time.time()
            await self.rag_engine.generate_response(query)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Query: {query[:40]}... → {elapsed:.2f}s")

        return {
            'avg_response_time': np.mean(times),
            'median_response_time': np.median(times),
            'p95_response_time': np.percentile(times, 95),
            'min_response_time': np.min(times),
            'max_response_time': np.max(times)
        }

    async def _benchmark_quality(self) -> Dict:
        """Benchmark answer quality"""
        print("\n2. QUALITY BENCHMARK (RAGAs)")
        print("-" * 60)

        results = await self.evaluator.run_evaluation(self.rag_engine)

        print(f"  Faithfulness: {results.get('faithfulness', 0):.2f}")
        print(f"  Answer Relevancy: {results.get('answer_relevancy', 0):.2f}")
        print(f"  Context Recall: {results.get('context_recall', 0):.2f}")
        print(f"  Context Precision: {results.get('context_precision', 0):.2f}")

        return results

    async def _benchmark_retrieval(self) -> Dict:
        """Benchmark document retrieval"""
        print("\n3. RETRIEVAL BENCHMARK")
        print("-" * 60)

        # Test retrieval with known queries
        test_cases = [
            ("FastAPI", ["fastapi", "framework"]),
            ("async", ["asyncio", "await"]),
            ("CORS", ["cors", "middleware"])
        ]

        precisions = []
        for query, expected_terms in test_cases:
            results = await self.rag_engine.search_documents(query, n_results=5)

            # Check if expected terms appear in results
            retrieved_text = " ".join(results['documents'][0]).lower()
            matches = sum(1 for term in expected_terms if term in retrieved_text)
            precision = matches / len(expected_terms)
            precisions.append(precision)

            print(f"  Query: {query} → Precision: {precision:.2f}")

        return {
            'avg_retrieval_precision': np.mean(precisions)
        }

    def _print_summary(self, results: Dict):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        perf = results['performance']
        qual = results['quality']

        print(f"\nPerformance:")
        print(f"  Average Response Time: {perf['avg_response_time']:.2f}s")
        print(f"  P95 Response Time: {perf['p95_response_time']:.2f}s")

        print(f"\nQuality:")
        print(f"  Faithfulness: {qual.get('faithfulness', 0):.2f}")
        print(f"  Answer Relevancy: {qual.get('answer_relevancy', 0):.2f}")

        print(f"\nRetrieval:")
        print(f"  Avg Precision: {results['retrieval']['avg_retrieval_precision']:.2f}")

        print("\n" + "=" * 60)


# CLI usage
async def main():
    from rag_engine_async import AsyncRAGEngine

    engine = AsyncRAGEngine()
    benchmark = RAGBenchmark(engine)

    await benchmark.run_benchmark()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## ✅ Critérios de Aceitação

- [ ] RAGAs instalado e funcionando
- [ ] Test dataset criado (mínimo 10 casos)
- [ ] Metrics calculando corretamente
- [ ] Benchmark rodando e salvando resultados
- [ ] Relatório de qualidade gerado
- [ ] CI/CD integration (opcional)

---

## 🧪 Como Testar

```bash
# 1. Rodar benchmark completo
python evaluation/benchmark.py

# 2. Verificar resultados
cat benchmark_results_*.json | jq .

# 3. Integrar com CI/CD (GitHub Actions)
# → Rodar benchmark em cada PR
# → Falhar se métricas regredirem
```

---

## 📊 Target Metrics

### Qualidade (RAGAs):
- **Faithfulness:** > 0.85 (resposta baseada no contexto)
- **Answer Relevancy:** > 0.80 (resposta relevante)
- **Context Recall:** > 0.75 (recupera contexto relevante)
- **Context Precision:** > 0.70 (contexto é preciso)

### Performance:
- **Avg Response Time:** < 2s
- **P95 Response Time:** < 3s
- **Retrieval Precision:** > 0.80

---

**💡 Dica:** Use evaluation results para iterar e melhorar prompts, chunk size, retrieval strategy!
