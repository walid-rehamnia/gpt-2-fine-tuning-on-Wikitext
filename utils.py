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
import os
import time

import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import psutil
# import pymeteor.pymeteor as pymeteor
import sacrebleu
import torch
from datasets import DatasetDict, load_dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainerCallback)

SEED = 2024

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TimerMemoryTracker:
    """
    Context manager class to track the time and memory consumption
    (both GPU and CPU) during model training or inference processes.
    It measures the elapsed time and the memory used on both CPU and GPU,
    if available.

    Methods:
        __enter__(): Starts the timer and memory tracking.
        __exit__(): Stops the timer and memory tracking.
        report(): Prints the elapsed time and memory consumption details.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_gpu_memory = None
        self.end_gpu_memory = None
        self.start_cpu_memory = None
        self.end_cpu_memory = None

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
        """
        Prints the elapsed time, GPU memory usage, and CPU memory usage.
        """
        # Report time and memory consumption
        elapsed_time = self.end_time - self.start_time
        gpu_memory_used = self.end_gpu_memory - \
            self.start_gpu_memory if torch.cuda.is_available() else 0
        cpu_memory_used = self.end_cpu_memory - self.start_cpu_memory

        print(f"Training time: {elapsed_time:.2f} seconds")
        print(f"GPU memory used: {gpu_memory_used / (1024 ** 2):.2f} MB")
        print(f"CPU memory used: {cpu_memory_used / (1024 ** 2):.2f} MB")


class LossRecorderCallback(TrainerCallback):
    """
    A callback class to record training and evaluation loss values during
    the model's training and evaluation process.

    Methods:
        on_log(): Appends the loss values from the logs into the
        appropriate lists.
    """

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
    """
    Plots the training and validation loss over epochs
    for visualization purposes.

    Args:
        train_losses (list): List of training loss values.
        eval_losses (list): List of evaluation loss values.
    """
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
    """
    Singleton class that ensures a single instance of the GPT-2 tokenizer
    is created and reused throughout the application to optimize performance.

    Methods:
        get_tokenizer(): Returns the single instance of the GPT-2 tokenizer.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TokenizerSingleton, cls).__new__(cls)
            # Tokenizer initialization will be done later
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'tokenizer'):  # Avoid re-initialization
            self.tokenizer = self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Load the tokenizer from Hugging Face."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token
        return tokenizer

    def get_tokenizer(self):
        """Return the loaded tokenizer."""
        return self.tokenizer


def get_model():
    """
    Loads and returns a pre-trained GPT-2 model with automatic device mapping.

    Returns:
        model: Pre-trained GPT-2 model.
    """
    return AutoModelForCausalLM.from_pretrained("gpt2", device_map='auto')


def freeze_weights(model) -> None:
    """
    Freezes all the weights in the given model to prevent further training.

    Args:
        model (torch.nn.Module): The model to freeze weights for.
    """
    for param in model.parameters():
        param.requires_grad = False


def load_data(fraction=1):
    """
    Loads the WikiText dataset and returns it as a DatasetDict object.
    Can return a subset of the data based on the provided fraction.

    Args:
        fraction (float): Fraction of the data to load (between 0 and 1).

    Returns:
        DatasetDict: A dictionary with train, validation, and test datasets.
    """
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # If a valid fraction is passed, return a fraction of the data
    if 0 < fraction < 1:
        def get_fraction(key):
            return dataset[key].shuffle(
                seed=SEED).select(range(int(fraction * len(dataset[key])))
                                  )

        dataset = DatasetDict({
            'test':  get_fraction("test"),
            'train': get_fraction("train"),
            'validation': get_fraction("validation"),
        })

    return dataset


def load_test_data(num_examples: int):
    """
    Loads a subset of test examples from the WikiText dataset.

    Args:
        num_examples (int): The number of test examples to load.

    Returns:
        list: List of test examples.
    """
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
    """
    Tokenizes the given examples using the GPT-2 tokenizer.

    Args:
        examples (dict): A dictionary of text examples to tokenize.

    Returns:
        dict: Tokenized examples with input_ids, attention_mask, and labels.
    """
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
    """
    Tokenizes the given dataset and formats it for PyTorch models.

    Args:
        dataset (DatasetDict): The dataset to tokenize.

    Returns:
        DatasetDict: The tokenized and formatted dataset.
    """
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True,  remove_columns=["text"])
    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model to analyze.
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
    """
    Generates text based on a given prompt using a specified
    model and tokenizer.

    Args:
        model (AutoModelForCausalLM): The pre-trained language model
        (e.g., GPT-2) for generating text.
        tokenizer (AutoTokenizer): The tokenizer corresponding
        to the model for encoding the input prompt.
        prompt (str): The initial text prompt to guide text generation.
        min_length (int): Minimum number of tokens to generate.
        max_length (int): Maximum number of tokens to generate.

    Returns:
        str: The generated text based on the given prompt, decoded and stripped
        of special tokens.
    """
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
    """
    Generates hypotheses and references for a list of examples
    using a pre-trained model.

    Args:
        model_path (str): The path to the fine-tuned model.
        examples (list of str): A list of text examples,
        where each example will be split intotwo parts,
        used to generate the hypothesis and reference.

    Returns:
        tuple:
            - hypotheses (list of str): The generated hypotheses
              (model predictions).
            - references (list of list of str): The true references
              corresponding to the second part of the examples.
    """
    # Step 1: Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
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


def initialize_trainer_and_dataset(model, training_args, data_fraction):
    """
    Initializes the Trainer for model fine-tuning and loads the dataset.

    This function loads the WikiText dataset, tokenizes it, and sets up
    the Trainer with the provided training arguments. It also initializes
    a loss recorder to keep track of training and evaluation losses.

    Args:
        model: The model to be fine tuned.
        training_args (TrainingArguments): The training arguments
        for the Trainer.
        batch_size (int): The batch size for training and evaluation.
        epochs (int): The number of epochs for training.
        data_fraction (float): The fraction of the dataset
        to use for training (0 < data_fraction <= 1).

    Returns:
        tuple: A tuple containing:
            - Trainer: The initialized Trainer instance.
            - LossRecorderCallback: The loss recorder for tracking training
              and evaluation losses.
    """
    dataset = load_data(fraction=data_fraction)
    tokenized_dataset = get_tokenized_dataset(dataset=dataset)

    loss_recorder = LossRecorderCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        callbacks=[loss_recorder],
        tokenizer=TokenizerSingleton().get_tokenizer(),
    )

    return trainer, loss_recorder


def evaluate_model(model_path, num_examples, file_path):
    """
    Evaluates a model using a subset of examples by calculating
    different metrics after preparing the base tabular data.

    Args:
        model_path (str): The path to the model being evaluated.
        num_examples (int): Number of examples to load for evaluation.
        file_path (str): The path where evaluation
        results (hypotheses, references) will be saved as a CSV file.

    Returns:
        None
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    # Step 1: Load the test data
    examples = load_test_data(num_examples=num_examples)

    # Step 2: Generate hypothesis, references from examples
    hypotheses, references = generate_model_hypothesis_references(
        model_path=model_path, examples=examples)
    # if the returned data is null, no need to continue further
    if len(hypotheses) == 0 or len(references) == 0:
        return

    # Step 3:Save each example with its hypothesis and reference;
    # for comparaison, post-processing, ...
    data = {
        "examples": examples,
        "hypotheses": hypotheses,
        "references": references,
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path)

    calculate_metrics(file_path=file_path)


def evaluate_across_models(full_model_path, lora_model_path, num_examples):
    """
    Evaluates two models (full fine-tuned and LoRA fine-tuned)
    and compares their BLEU scores.

    Args:
        full_model_path (str): Path to the fully fine-tuned model.
        lora_model_path (str): Path to the LoRA fine-tuned model.
        num_examples (int): Number of examples to use for evaluation.

    Returns:
        None
    """
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

    # Step 2:Save each example with its hypothesis and reference
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


def calculate_metrics(file_path) -> None:
    """
    Calculates the different metrics (bleu, sacrebleu, rouge, ter_score,
    chrf_score, and meteor)from the generated hypotheses and references
    saved in a CSV file.

    Args:
        file_path (str): The path to the CSV file containing
        hypotheses and references.

    Returns:
        None
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    df = pd.read_csv(file_path)
    # Remove rows with any null values
    df = df.dropna()
    hypotheses = df['hypotheses'].tolist()
    references = df['references'].tolist()
    # Remove any potential unwanted brackets and quotes from references
    references = [element.strip('[]\'')
                  for element in references]

    # metrics expects the reference list to be a list of lists
    references = [[reference] for reference in references]

    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=hypotheses, references=references)
    print(f"BLEU:{results}")

    sacrebleu1 = evaluate.load("sacrebleu")
    results = sacrebleu1.compute(predictions=hypotheses, references=references)
    print(f"sacrebleu:{results}")

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=hypotheses, references=references)
    print(f"rouge:{results}")

    ter_score = 0
    chrf_score = 0
    # meteor_score = 0
    length = len(references)
    for hypothesis, reference in tqdm(zip(hypotheses, references),
                                      desc="Get scores...", unit="example"):
        ter_score += sacrebleu.sentence_ter(hypothesis, reference).score
        chrf_score += sacrebleu.sentence_chrf(hypothesis, reference).score
        # meteor_score += pymeteor.meteor(hypothesis, reference[0])

    # Getting the average of the different scores
    ter_score /= length
    chrf_score /= length
    # meteor_score /= length
    print(f"TER score:{ter_score:.2f}")
    print(f"CHRF score:{chrf_score:.2f}")
    # print(f"METEOR score:{chrf_score:.2f}")
