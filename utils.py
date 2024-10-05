from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model
import numpy as np
import evaluate
import torch
import sacrebleu
from tqdm import tqdm
import pandas as pd
from pprint import pprint


SEED = 2024

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.set_verbosity_error()


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model():
    return AutoModelForCausalLM.from_pretrained("gpt2", device_map='auto')


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_weights(model)->None:
    for param in model.parameters():
        param.requires_grad = False

def compute_bleu(predictions, references):
    # Logic to compute BLEU score
    pass


def load_dataset(fraction=1):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # If a valid fraction is passed, reduce the initial dataset before returning it
    if 0 < fraction < 1:
        def get_fraction(key): return dataset[key].shuffle(
            seed=SEED).select(range(int(fraction * len(dataset[key]))))

        # small_valid_dataset = dataset['validation'].shuffle(
        #     seed=SEED).select(range(int(fraction * len(dataset['validation']))))

        # small_test_dataset = dataset['test'].shuffle(
        #     seed=SEED).select(range(int(fraction * len(dataset['test']))))

        dataset = DatasetDict({
            'test':  get_fraction("test"),
            'train': get_fraction("train"),
            'validation': get_fraction("validation"),
        })

    return dataset


def tokenize_function(examples):
    tokenizer = get_tokenizer()
    #1024 is set as the max_length to utilizes the full context window of GPT-2,
    #  which is better for understanding long sequences of text, however it uses much memory comparing to 512
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    #Use input_ids as labels for training
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def get_tokenized_dataset(dataset):
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True,  remove_columns=["text"])
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {
            all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def evaluate_model(path) -> None:
   # Evaluate the model on validation set
    print("Evaluate the model on validation set...")
    model = AutoModelForCausalLM.from_pretrained(
        "fine-tuned-gpt2-lora")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        batch = tokenizer(
            "“Life is like a box of chocolates, you never know what you are gonna get” ->: ", return_tensors='pt')
        output_tokens = model.generate(**batch, max_new_tokens=25)

    print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
    # eval_results = trainer.evaluate()
    # print(eval_results)


# Test generation before fine-tuning
def generate_text_before(prompt, max_length=50):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        device_map='auto',
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs["input_ids"], max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def get_blue_metric(path):
    # Load the fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Test the model on a validation sentence or dataset
    inputs = tokenizer(
        "the wheather today has dramatically changed over time", return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=512,
                             # Prevents repeated 2-grams (avoid repeated sentences)
                             no_repeat_ngram_size=2,
                             # temperature=0.7,  # Introduces randomness
                             top_k=50,  # Only consider the top 50 tokens for each prediction
                             top_p=0.9,  # Consider the top 90% probability mass for sampling
                             )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(generated_text)

    # Example reference and hypothesis text (use actual dataset texts)
    reference = ["This is the reference text that the model should generate."]
    hypothesis = ["This is the generated text from the fine-tuned model."]

    # BLEU score calculation
    bleu = sacrebleu.corpus_bleu(hypothesis, [reference])
    print(f"BLEU score: {bleu.score}")


# Function to generate text
def generate_text(prompt, model, tokenizer, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs["input_ids"], max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_quality_bleu(model_path, limit=None):
    # Step 1: Load the test dataset
    dataset = load_dataset(0.1)
    test_dataset = dataset["test"]
    if limit:
        test_dataset = test_dataset.select(range(limit))
    print(test_dataset[0])
    print("###############**************")
    # Step 2: Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Step 3: Generate hypothesis texts
    hypotheses = []
    references = [item["text"]
                  for item in test_dataset if len(item["text"]) > 0]
    with torch.no_grad():  # Disable gradient calculation
        for item in tqdm(test_dataset, desc="Generating hypotheses", unit="prompt"):

            prompt = item["text"]
            if len(prompt) > 0:
                # Adjust max_length if needed
                generated_text = generate_text(
                    prompt, model, tokenizer, max_new_tokens=50)
                hypotheses.append(generated_text)

    # Step 4: Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"BLEU score: {bleu.score}")

    # Step 5: Create a DataFrame and save to CSV and Excel
    data = {
        "Reference": references,
        "Hypothesis": hypotheses,
    }
    print(references[0])
    print(len(hypotheses), len(references))
    pprint(data)
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(f"hypotheses_references_{bleu.score:.2f}.csv", index=False)
