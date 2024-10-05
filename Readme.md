## Task objective
The objective from this task is to fine tune the "GPT-2" LLM model on Wikitext dataset with simple fine tuning as well as with LoRA technique.
For simplicity and demonstrative purposes the code is implemented 



## GPT-2 Fine-Tuning Comparison

This project compares the full fine-tuning and LoRA fine-tuning of the GPT-2 model on the WikiText dataset. Below is the comparison of key metrics:

| Metric              | Full Fine-Tuning | LoRA Fine-Tuning |
|---------------------|------------------|------------------|
| BLEU Score          | X.XXX            | Y.YYY            |
| Training Time (s)   | XXX seconds      | YYY seconds      |
| Memory Usage (MB)   | XXX MB           | YYY MB           |

### Code
The pipeline for both fine-tuning methods can be found in the `Full Model Fine-Tuning.ipynb` and `LoRA Model Fine-Tuning.ipynb` files.



## Reproducibility instrcutions:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/gpt2-finetuning-comparison.git
   cd gpt2-finetuning-comparison
   ```

2. Create and activate virtual environment:

    - check the existence of the package
        ```bash
        pip install virtualenv
        ```

    - Windows : 
        ```bash
        python -m venv venv
        venv/Scripts/activate
        ```
    - Linux : 
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the fine tuning:
- Run the fine tune of your choice by passing the required mode arg ("full" or "lora"),
  you can also pass the next optional args: 
   - batch_size (int): Batch size for training phase (default: 4)
   - data_fraction (float): Portion or fraction of dataset to get(ranging from 0 to 1)
   - epochs (int): Number of epochs for training (default: 2)
   - lora_rank (int): Rank "r" for LoRA technique (default: 4)

+ examples:
   ```bash
   python main.py full --batch_size 4 --data_fraction 1 --epochs 5
   ```
   ```bash
   python main.py lora --batch_size 8 --data_fraction 1 --epochs 3 --lora_rank 8
   ```



5. Check the comparison results:
   After both scripts finish running, a summary table with BLEU scores, training time, and memory usage will be printed.



pip install -r requirements.txt


https://huggingface.co/docs/peft/main/en/conceptual_guides/lora