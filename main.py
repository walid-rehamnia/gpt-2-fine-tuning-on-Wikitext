"""
main.py

This script serves as the entry point for fine-tuning a pre-trained GPT-2 model
on the WikiText dataset using either full fine-tuning or Low-Rank Adapters
(LoRA)fine-tuning. The fine-tuning method is chosen based on a command-line
argument,and additional options like the number of epochs and batch size
can be specified.

Usage:
    python main.py [full|lora] --epochs 3 --batch_size 8

Arguments:
    mode (str): 'full' for full fine-tuning or 'lora' for LoRA fine-tuning.
    epochs (int): The number of epochs to train (default is 3).
    batch_size (int): The batch size for training (default is 8).

Author: [r.walid]
"""

import argparse

from transformers import logging

from full_finetuning import full_fine_tuning
from lora_finetuning import lora_fine_tuning

logging.set_verbosity_error()

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description="Fine-tuning GPT-2 model")

    # Add a required argument 'mode' to choose between 'full' and 'lora'
    parser.add_argument(
        'mode', type=str,
        help="Type 'full' for full fine-tuning or 'lora' for LoRA fine-tuning"
    )
    # Add optional arguments
    parser.add_argument('--lora_rank', type=int, default=4,
                        help="Rank r or LoRA technique (default: 4)")
    parser.add_argument('--epochs', type=int, default=2,
                        help="Number of epochs for training (default: 2)")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for training (default: 4)")
    parser.add_argument('--data_fraction', type=float, default=1,
                        help="Portion or fraction of dataset to get(0 to 1)")

    # Parse the arguments
    args = parser.parse_args()

    # Run the appropriate function based on the argument
    if args.mode == 'full':
        full_fine_tuning(batch_size=args.batch_size,
                         data_fraction=args.data_fraction, epochs=args.epochs)
    elif args.mode == 'lora':
        lora_fine_tuning(
            batch_size=args.batch_size, data_fraction=args.data_fraction,
            epochs=args.epochs, lora_rank=args.lora_rank)
    else:
        print("Error: Invalid mode. Please choose either 'full' or 'lora'.")
        print("Optional args are : lora_rank, epochs, data_fraction,\
               and batch_size")
        parser.print_help()
