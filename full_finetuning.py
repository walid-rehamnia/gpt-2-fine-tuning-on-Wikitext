import torch
from transformers import Trainer, TrainingArguments

from utils import load_dataset, get_tokenized_dataset, get_model, count_trainable_params


def lora_fine_tuning(batch_size: int, data_fraction: float, epochs: int):
    """Full fine tune of the pretrained GPT-2 model on Wikitext dataset
    
    Args:
        batch_size (int): Batch size for training phase (default: 4)
        data_fraction (float): Portion or fraction of dataset to get(ranging from 0 to 1)
        epochs (int): Number of epochs for training (default: 2)
        lora_rank (int): Rank "r" for LoRA technique (default: 4)
    """
    print(
        f"Starting Full fine-tuning with : batch size={batch_size}, data_fraction={data_fraction}, 
        epochs={epochs}")

    # Load dataset and tokenizer

    model = get_model()
    num_trainable_params = count_trainable_params(model)
    print(f"trainable params: {num_trainable_params}")

    training_args = TrainingArguments(
        output_dir="./gpt2-full-finetuned",
        overwrite_output_dir=True,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir="./logs/full_finetune",
        logging_steps=100,
        save_steps=500,
        fp16=torch.cuda.is_available(), 
        weight_decay=0.01,
        save_total_limit=3
    )

    dataset = load_dataset(fraction=data_fraction)
    tokenized_dataset = get_tokenized_dataset(dataset=dataset)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    # Fine-tuning the model
    trainer.train()

    # Save the model
    print("Save the model...")
    trainer.save_model("model-gpt2-full-finetuned")