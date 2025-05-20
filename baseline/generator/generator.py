# baseline/generator/generator.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Generator:
    """
    A generator that uses a transformer model (flan-t5-base) to create answers from retrieved context.
    """
    def __init__(self, model_name="google/flan-t5-base", max_input_tokens=512, max_output_tokens=100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

    def build_prompt(self, question, context_chunks):
        """
        Builds a prompt from the context chunks and the question.
        """
        context = "\n".join([chunk for chunk in context_chunks])
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return prompt

    def generate_answer(self, prompt):
        """
        Generates an answer given a prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                do_sample=False,
                early_stopping=True
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
