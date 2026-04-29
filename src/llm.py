import ollama
from typing import Generator


class OllamaLLM:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 512},
        )
        return response["message"]["content"]

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"num_predict": 512},
        )
        for chunk in response:
            yield chunk["message"]["content"]
