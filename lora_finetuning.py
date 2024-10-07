"""
lora_finetuning.py

This module contains the implementation of LoRA (Low-Rank Adapters)
fine-tuning pipeline for a pre-trained GPT-2 model on the WikiText dataset.
LoRA is a parameter-efficient fine-tuning methodthat freezes the pre-trained
model weights and introduces small trainable weight matrices with a low-rank
decomposition.

Author: [r.walid]
"""


import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments

from utils import (TimerMemoryTracker, evaluate_model, freeze_weights,
                   get_model, initialize_trainer_and_dataset, plot_losses)


def lora_fine_tuning(batch_size: int, data_fraction: float,
                     epochs: int, lora_rank: int):
    """Fine tune the pretrained GPT-2 model on
      Wikitext dataset with LoRA technique

    Args:
        batch_size (int): Batch size for training phase (default: 4)
        data_fraction (float): Portion or fraction of dataset to get
                               (ranging from 0 to 1)
        epochs (int): Number of epochs for training (default: 2)
        lora_rank (int): Rank "r" for LoRA technique (default: 4)
    """
    print(f'Starting LoRA fine-tuning with : batch size={batch_size}, \
            data_fraction={data_fraction}, epochs={epochs},\
            and lora_rank={lora_rank}')

    model = get_model()
    # Additional step to ensure that all model params are well freezed
    freeze_weights(model)
    # Define LoRA configuration
    lora_config = LoraConfig(
        # Rank of the low-rank matrices
        r=lora_rank,
        lora_alpha=32,      # Alpha scaling factor
        # Target Conv1D layers used in attention and projection
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,   # Dropout for LoRA #0.1
        bias="none",        # Whether to adapt bias parameters
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to the model
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="results/gpt2-lora-finetuned",
        overwrite_output_dir=True,
        learning_rate=1e-4,  # Higher LR for LoRA
        # Larger batch size due to reduced memory usage
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,  # Fewer epochs as LoRA converges faster
        # gradient_accumulation_steps=2,  # Less accumulation needed
        logging_dir="results/logs/lora_finetune",
        logging_steps=50,  # More frequent logging to track rapid convergence
        eval_steps=50,
        eval_strategy="steps",
        save_steps=500,
        fp16=torch.cuda.is_available(),
        weight_decay=0.0,  # Less regularization needed,

    )

    trainer, loss_recorder = initialize_trainer_and_dataset(
        peft_model, training_args, data_fraction)

    # Fine-tuning the model
    with TimerMemoryTracker() as tracker:
        trainer.train()

    tracker.report()

    plot_losses(loss_recorder.train_losses, loss_recorder.eval_losses)

    # Saving the lora-tuned model the model
    trainer.save_model("results/models/gpt2-lora-finetuned")

    # model evaluation
    evaluate_model(model_path="results/models/gpt2-lora-finetuned",
                   num_examples=435, file_path="results/lora_results.csv")
