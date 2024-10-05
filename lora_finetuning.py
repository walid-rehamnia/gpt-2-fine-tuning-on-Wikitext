import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from utils import load_dataset, get_tokenized_dataset, get_model


def lora_fine_tuning(batch_size: int, data_fraction: float, epochs: int, lora_rank: int):
    """Fine tune the pretrained GPT-2 model on Wikitext dataset with LoRA technique

    Args:
        batch_size (int): Batch size for training phase (default: 4)
        data_fraction (float): Portion or fraction of dataset to get(ranging from 0 to 1)
        epochs (int): Number of epochs for training (default: 2)
        lora_rank (int): Rank "r" for LoRA technique (default: 4)
    """
    print(
        f"Starting LoRA fine-tuning with : batch size={batch_size}, data_fraction={data_fraction}, 
        epochs={epochs}, and lora_rank={lora_rank}")

    # Load dataset and tokenizer

    model = get_model()
    # Define LoRA configuration
    lora_config = LoraConfig(
        # Rank of the low-rank matrices, set very small due to ressources limitations
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
        output_dir="./gpt2-lora-finetuned",
        overwrite_output_dir=True,
        learning_rate=1e-4,  # Higher LR for LoRA
        per_device_train_batch_size=batch_size,  # Larger batch size due to reduced memory usage
        num_train_epochs=epochs,  # Fewer epochs as LoRA converges faster
        gradient_accumulation_steps=2,  # Less accumulation needed
        logging_dir="./logs/lora_finetune",
        logging_steps=50,  # More frequent logging to track rapid convergence
        save_steps=500,
        fp16=torch.cuda.is_available(),
        weight_decay=0.0  # Less regularization needed
    )

    dataset = load_dataset(fraction=data_fraction)
    tokenized_dataset = get_tokenized_dataset(dataset=dataset)
    
    # Initialize the Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    # Fine-tuning the model
    print("Fine-tuning the model...")
    trainer.train()

    # Save the model
    print("Save the model...")
    trainer.save_model("model-gpt2-lora-finetuned")
