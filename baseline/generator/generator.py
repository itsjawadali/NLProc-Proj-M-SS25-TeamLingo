from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
load_dotenv()
class Generator:
    def __init__(self, model_name: str = "llama3.1-8b"):
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("❌ CEREBRAS_API_KEY environment variable not set.")
        
        self.client = Cerebras(api_key=api_key)
        self.model_name = model_name

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_length: int = 128) -> str:
        # inputs = self.tokenizer(prompt, return_tensors='pt')
        # outputs = self.model.generate(
        #     **inputs,
        #     max_length=max_length,
        #     do_sample=False,    # greedy
        #     num_beams=1         # single beam
        # )
        messages = []
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=max_length
        )
        return response.choices[0].message.content
