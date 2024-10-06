"""
utils.py

This module provides utility functions used across the fine-tuning pipelines,
including dataset loading, tokenization, and BLEU score computation.
These utilities help streamline and modularize the code for reusability
across both full fine-tuning and LoRA fine-tuning scripts.

Functions:
    load_dataset(): Loads the WikiText dataset from Hugging Face Datasets.
    get_tokenizer(): Returns the pre-trained GPT-2 tokenizer.
    evaluate_model(model_path, num_examples): Computes the BLEU score
    between model hypotheses and references texts.
    ...
Author: [r.walid]
"""


import time

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import pymeteor.pymeteor as pymeteor
import sacrebleu
import torch
from datasets import DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

SEED = 2024

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TimerMemoryTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_gpu_memory = None
        self.end_gpu_memory = None

    def __enter__(self):
        # Start timing and memory tracking
        self.start_time = time.perf_counter()

        # If using a GPU, measure GPU memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensures accurate GPU timing
            self.start_gpu_memory = torch.cuda.memory_allocated()

        # Measure CPU memory (optional)
        self.start_cpu_memory = psutil.Process().memory_info().rss

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # End timing and memory tracking
        self.end_time = time.perf_counter()

        # Measure GPU memory at the end
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensures accurate GPU timing
            self.end_gpu_memory = torch.cuda.memory_allocated()

        # Measure CPU memory (optional)
        self.end_cpu_memory = psutil.Process().memory_info().rss

    def report(self):
        # Report time and memory consumption
        elapsed_time = self.end_time - self.start_time
        gpu_memory_used = self.end_gpu_memory - \
            self.start_gpu_memory if torch.cuda.is_available() else 0
        cpu_memory_used = self.end_cpu_memory - self.start_cpu_memory

        print(f"Training time: {elapsed_time:.2f} seconds")
        print(f"GPU memory used: {gpu_memory_used / (1024 ** 2):.2f} MB")
        print(f"CPU memory used: {cpu_memory_used / (1024 ** 2):.2f} MB")


class LossRecorderCallback(TrainerCallback):
    def __init__(self):
        # Initialize lists to store losses
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log training loss
        if "loss" in logs:
            self.train_losses.append(logs["loss"])

        # Log evaluation loss if available
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])


def plot_losses(train_losses, eval_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


class TokenizerSingleton:
    """Asengloten class which ensures that only one instance of the tokenizer
      is generated in the whole project ( for better performance )

    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TokenizerSingleton, cls).__new__(cls)
            cls._instance.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            # Tokenizer padding
            cls._instance.tokenizer.pad_token = \
                cls._instance.tokenizer.eos_token
        return cls._instance

    def get_tokenizer(self):
        return self.tokenizer


def get_model():
    return AutoModelForCausalLM.from_pretrained("gpt2", device_map='auto')


def freeze_weights(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def load_data(fraction=1):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # If a valid fraction is passed, return a fraction of the data
    if 0 < fraction < 1:
        def get_fraction(key): return dataset[key].shuffle(
            seed=SEED).select(range(int(fraction * len(dataset[key]))))

        dataset = DatasetDict({
            'test':  get_fraction("test"),
            'train': get_fraction("train"),
            'validation': get_fraction("validation"),
        })

    return dataset


def load_test_data(num_examples: int):
    # Step 1: Load the test dataset
    test_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # remove the void examples
    examples = [text for text in test_dataset['text']
                if len(text) > 0]

    # Getting the num_examples while avoiding going out of list bound
    limit = min(len(examples), num_examples)
    examples = examples[:limit]

    return examples


def tokenize_function(examples):
    tokenizer = TokenizerSingleton().get_tokenizer()
    # 1024 is set as the max_length to utilizes the full context
    #  window of GPT-2, which is better for understanding long sequences
    #  of text, however it uses much memory comparing to 512
    tokenized = tokenizer(examples["text"], padding="max_length",
                          truncation=True, max_length=512)
    # Add labels for training (to find the loss among the returned values)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def get_tokenized_dataset(dataset):
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True,  remove_columns=["text"])
    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"])
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


# Function to generate text
def generate_text(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                  prompt: str, min_length: int, max_length: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs["input_ids"], min_length=min_length, max_length=max_length,
        num_beams=5,
        # Prevents repeated 2-grams (avoid repeated sentences)
        no_repeat_ngram_size=2,
        # num_return_sequences=1
        # temperature=0.7,  # Introduces randomness
        # top_k=50,  # Only consider the top 50 tokens for each prediction
        # top_p=0.9,  # Consider the top 90% probability mass for sampling
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_model_hypothesis_references(model_path, examples):
    # Step 1: Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    # tokenizer = TokenizerSingleton().get_tokenizer()
    tokenizer = TokenizerSingleton().get_tokenizer()
    # Step 3: Generate hypothesis texts
    hypotheses = []
    references = []

    with torch.no_grad():  # Disable gradient calculation
        for example in tqdm(examples, desc="Generating hypotheses",
                            unit="example"):
            words = example.split()
            half_num_words = len(words)//2
            # construct a prompt from the first part of the example
            prompt = " ".join(words[:half_num_words])
            # There is 1.2-1.5 tokens per word on average, so:
            # num of tokens to generate (execluding the input tokens)
            min_length = int(len(words) * 1.2)
            # num of tokens to generate (including the input tokens)
            max_length = int(len(words) * 1.5)
            generated_text = generate_text(
                model=model, tokenizer=tokenizer, prompt=prompt,
                min_length=min_length, max_length=max_length)

            # Remove the prompt from the generated text for evaluation
            generated_without_prompt = generated_text[len(prompt):].strip()
            hypotheses.append(generated_without_prompt)
            # reconstruct the second original part of the example
            second_part_txt = " ".join(words[half_num_words:])
            # Append it to the references as a list of single text
            # (because BLRU metric later expect a list of references)
            references.append([second_part_txt])

    return hypotheses, references


def evaluate_model(model_path, num_examples, file_path):
    # Step 1: Load the test data
    examples = load_test_data(num_examples=num_examples)

    # Step 2: Generate hypothesis, references from examples
    hypotheses, references = generate_model_hypothesis_references(
        model_path=model_path, examples=examples)
    # if the returned data is null, no need to continue further
    if len(hypotheses) == 0 or len(references) == 0:
        return

    # Step 3: Calculate BLEU score
    bleu_score = sacrebleu.corpus_bleu(hypotheses, references)
    print(f"BLEU score: {bleu_score}")

    # # Step 3: Calculate ROUGE scores
    # rouge = Rouge()
    # rouge_scores = rouge.get_scores(hypotheses, references)

    # # Display the ROUGE scores
    # for i, score in enumerate(rouge_scores):
    #     print(f"Generated Text {i + 1}:")
    #     print(score)

    # Step 4:Save each example with its hypothesis and reference;
    # for comparaison, post-processing, ...
    data = {
        "examples": examples,
        "hypotheses": hypotheses,
        "references": references,
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path)


def evaluate_across_models(full_model_path, lora_model_path, num_examples):

    # Step 1: Load the test data
    examples = load_test_data(num_examples=num_examples)

    # Get the hypothesis of each of the moddels, and since references
    # gonna be the same I'll get them once
    initial_hypotheses, references = generate_model_hypothesis_references(
        model_path="gpt2", examples=examples)
    full_hypotheses, _ = generate_model_hypothesis_references(
        model_path=full_model_path, examples=examples)
    lora_hypotheses, _ = generate_model_hypothesis_references(
        model_path=lora_model_path, examples=examples)

    # Step 3: Calculate BLEU score
    initial_bleu = sacrebleu.corpus_bleu(initial_hypotheses, references)
    full_bleu = sacrebleu.corpus_bleu(full_hypotheses, references)
    lora_bleu = sacrebleu.corpus_bleu(lora_hypotheses, references)

    print("********  BLEU evaluation  *********")
    print(f"-Initial-gpt model: {initial_bleu}")
    print(f"-Full-tuned model : {full_bleu}")
    print(f"-LoRA-tuned model : {lora_bleu}")

    # Step 4:Save each example with its hypothesis and reference
    # for comparaison, post-processing, ...
    data = {
        "examples": examples,
        "initial_hypotheses": initial_hypotheses,
        "full_hypotheses": full_hypotheses,
        "lora_hypotheses": lora_hypotheses,
        "references": references,
    }
    df = pd.DataFrame(data)
    df.to_csv("results/hypothesis_references_across_models.csv")


def calculate_pymeteor_score(file_path):

    df = pd.read_csv(file_path)
    print(df.columns)
    hypotheses = df['hypotheses'].values.tolist()
    references = df['references'].values.tolist()
    sum_scores = 0
    length = len(references)
    for hypothesis, reference in zip(hypotheses, references):
        sum_scores += pymeteor.meteor(hypothesis, reference)

    average_meteor_score = sum_scores/length
    print(average_meteor_score)
