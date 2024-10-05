from transformers import logging
import argparse

from full_finetuning import full_fine_tuning
from lora_finetuning import lora_fine_tuning

logging.set_verbosity_error()

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description="Fine-tuning GPT-2 model")

    # Add a required argument 'mode' to choose between 'full' and 'lora'
    parser.add_argument(
        'mode', type=str, help="Choose between 'full' for full fine-tuning and 'lora' for LoRA fine-tuning")
    # Add optional arguments
    parser.add_argument('--lora_rank', type=int, default=4,
                        help="Rank r or LoRA technique (default: 4)")
    parser.add_argument('--epochs', type=int, default=2,
                        help="Number of epochs for training (default: 2)")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for training (default: 4)")
    parser.add_argument('--data_fraction', type=int, default=1,
                        help="Portion or fraction of dataset to get(ranging from 0 to 1)")

    # Parse the arguments
    args = parser.parse_args()

    # Run the appropriate function based on the argument
    if args.mode == 'full':
        full_fine_tuning(batch_size=args.batch_size, ephochs=args.epochs)
    elif args.mode == 'lora':
        lora_fine_tuning(batch_size=args.batch_size,
                         ephochs=args.epochs, lora_rank=args.lora_rank)
    else:
        print("Error: Invalid mode. Please choose either 'full' or 'lora'.\n Optional args are : lora_rank, epochs, data_fraction, and batch_size")
        parser.print_help()
