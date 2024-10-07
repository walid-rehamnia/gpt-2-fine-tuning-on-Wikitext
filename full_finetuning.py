"""
full_finetuning.py

This module contains the implementation of the full fine-tuning pipeline
for a pre-trained GPT-2 model on the  WikiText dataset. The full fine-tuning
involves updating all model parameters during training. This script interfaces
with common utilities such as dataset loading and tokenization from the
utils module.

Author: [r.walid]
"""


import torch
from transformers import TrainingArguments

from utils import (TimerMemoryTracker, evaluate_model, get_model,
                   initialize_trainer_and_dataset, plot_losses,
                   print_trainable_parameters)


def full_fine_tuning(batch_size: int,
                     data_fraction: float, epochs: int):
    """Full fine tune of the pretrained GPT-2 model on Wikitext dataset

    Args:
        batch_size (int): Batch size for training phase (default: 4)
        data_fraction (float): Portion or fraction of dataset
        to get(ranging from 0 to 1)
        epochs (int): Number of epochs for training (default: 2)
    """
    print(f"Starting Full fine-tuning with : batch size={batch_size},\
           data_fraction={data_fraction}, epochs={epochs}")

    # Load dataset and tokenizer

    model = get_model()
    print_trainable_parameters(model=model)

    training_args = TrainingArguments(
        output_dir="results/gpt2-full-finetuned",
        overwrite_output_dir=True,
        learning_rate=1e-5,  # 1e-4 5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir="results/logs/full_finetune",
        logging_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_steps=500,
        fp16=torch.cuda.is_available(),
        weight_decay=0.01,
    )

    trainer, loss_recorder = initialize_trainer_and_dataset(
        model, training_args, data_fraction)

    # Full fine-tuning the model
    with TimerMemoryTracker() as tracker:
        trainer.train()
    tracker.report()

    plot_losses(loss_recorder.train_losses, loss_recorder.eval_losses)
    # Saving the full tuned model
    trainer.save_model("results/models/gpt2-full-finetuned")

    # model evaluation
    evaluate_model(model_path="results/models/gpt2-full-finetuned",
                   num_examples=435, file_path="results/full_results.csv")
