import argparse
from src import Embedding, ChromaStore, OllamaLLM, Retriever


def make_parser(parser):
    parser.add_argument("--model", type=str, default="gemma4:e4b")
    parser.add_argument("--embed_model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def run(args):
    embedding = Embedding(args.embed_model)
    vector_store = ChromaStore()
    llm = OllamaLLM(args.model)
    retriever = Retriever(embedding, vector_store, llm, k=args.k, threshold=args.threshold)

    print("=== RAG Query System ===")
    print("終了するには 'exit' を入力してください。\n")

    while True:
        question = input("質問を入力してください: （終了するには「exit」と入力してください）").strip()
        if question.lower() == "exit":
            print("終了します。")
            break
        if not question:
            continue
        print("\n回答を生成中...\n")
        answer = retriever.query(question)
        print(f"回答: {answer}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = make_parser(parser)
    run(args)
