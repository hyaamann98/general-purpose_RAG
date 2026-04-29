"""
Synthetic QA testset generator using RAGAS TestsetGenerator.

Usage:
    python tests/evaluation/generate_testset.py \
        --docs_dir ./docs_storage \
        --output tests/evaluation/fixtures/testset.json \
        --n_questions 50 \
        --generator_llm claude   # or "ollama" or "openai"

The generated testset.json is committed to the repo and used by
evaluate_retrieval.py and evaluate_e2e.py for repeatable evaluation.

Generator LLM recommendation:
  - "claude"  : Best quality; requires ANTHROPIC_API_KEY env var
  - "openai"  : Good quality; requires OPENAI_API_KEY env var
  - "ollama"  : No API key needed; lower question quality with small models
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_documents(docs_dir: str):
    from src.dataload import DataLoad
    from src.make_chunk import MakeChunk

    loader = DataLoad(source="local")
    documents = loader.load(docs_dir)
    if not documents:
        raise ValueError(f"No documents found in {docs_dir}")

    chunker = MakeChunk(chunk_size=500, chunk_overlap=50)
    chunks = chunker.text_split(documents)
    print(f"Loaded {len(documents)} documents → {len(chunks)} chunks")
    return chunks


def _build_generator_llm(provider: str):
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        import os
        model = os.getenv("LLM_MODEL", "gemma4:4b")
        return ChatOllama(model=model, temperature=0)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'claude', 'openai', or 'ollama'.")


def _build_embeddings_wrapper():
    """Wrap our Embedding class as a LangChain Embeddings interface for RAGAS."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")


def generate(docs_dir: str, output: str, n_questions: int, generator_llm_provider: str):
    from ragas.testset import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context
    from langchain_core.documents import Document as LCDocument

    chunks = _load_documents(docs_dir)

    # RAGAS expects a list of LangChain Documents
    lc_docs = [LCDocument(page_content=c.page_content, metadata=c.metadata) for c in chunks]

    generator_llm = _build_generator_llm(generator_llm_provider)
    critic_llm = generator_llm  # Use the same LLM as critic
    embeddings = _build_embeddings_wrapper()

    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings,
    )

    distributions = {simple: 0.4, reasoning: 0.4, multi_context: 0.2}

    print(f"Generating {n_questions} QA pairs using '{generator_llm_provider}'...")
    testset = generator.generate_with_langchain_docs(
        lc_docs,
        test_size=n_questions,
        distributions=distributions,
    )

    df = testset.to_pandas()
    records = df[["question", "contexts", "ground_truth", "evolution_type"]].to_dict(orient="records")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} QA pairs → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic RAG evaluation testset")
    parser.add_argument("--docs_dir", default="./docs_storage", help="Directory with source documents")
    parser.add_argument(
        "--output",
        default="tests/evaluation/fixtures/testset.json",
        help="Output path for testset JSON",
    )
    parser.add_argument("--n_questions", type=int, default=50, help="Number of QA pairs to generate")
    parser.add_argument(
        "--generator_llm",
        default="ollama",
        choices=["claude", "openai", "ollama"],
        help="LLM provider for question generation",
    )
    args = parser.parse_args()
    generate(args.docs_dir, args.output, args.n_questions, args.generator_llm)
