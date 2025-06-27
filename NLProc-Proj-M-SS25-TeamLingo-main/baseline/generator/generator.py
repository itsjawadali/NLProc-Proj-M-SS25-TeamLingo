from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_name: str = 'google/flan-t5-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_length: int = 128) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,    # greedy
            num_beams=1         # single beam
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
