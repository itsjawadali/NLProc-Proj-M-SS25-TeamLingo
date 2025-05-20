# evaluation/test_generator.py

from baseline.generator.generator import Generator

def test_generator():
    generator = Generator()
    # Simulated context chunks retrieved from a document:
    context_chunks = [
        "Global warming is primarily caused by increased levels of greenhouse gases.",
        "Renewable energy sources like solar and wind power are essential to reduce emissions."
    ]
    question = "How can renewable energy help the environment?"
    prompt = generator.build_prompt(question, context_chunks)
    answer = generator.generate_answer(prompt)
    print("Generator Test Results:")
    print(f"Prompt:\n{prompt}\n")
    print(f"Answer:\n{answer}")

if __name__ == "__main__":
    test_generator()
