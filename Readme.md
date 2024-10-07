# Fine-Tuning GPT-2 on WikiText with Full Tuning and LoRA: A Comparative Study

## Task objective
The objective from this task is to fine tune the "GPT-2" LLM model on Wikitext dataset with simple fine tuning as well as with LoRA technique... (see more details on the attached PDF report)

## Dataset Specification

| **Dataset Name**        | **Train Set** | **Validation Set** | **Test Set** |
|-------------------------|---------------|--------------------|--------------|
| wikitext-2-raw-v1      | 36,718        | 3,760              | 4,358        |


## Comparison of Fine-Tuning Approaches

| **Metric**                            | **Full Fine-Tuning** | **LoRA Fine-Tuning** |
|---------------------------------------|----------------------|----------------------|
| **Model size (MB)**                   | 479                  | 6.15                 |
| **Trainable parameters**              | 124,439,808          | 405,504              |
| **Training Time (s)**                 | 3,694.07             | 2,851.43             |
| **GPU memory usage (MB)**             | 975.40               | 19.34                |
| **CPU memory usage (MB)**             | 257.81               | 98.36                |
| **Best training loss**                | 0.3004               | 0.3706               |
| **Best validation loss**              | 0.4379               | 0.4443               |
| **BLEU score**                        | 0.02                 | 0.01                 |
| **SacreBLEU score**                   | 1.36                 | 1.33                 |
| **ROUGE-1 / ROUGE-2 / ROUGE-L**       | 0.18 / 0.02 / 0.12   | 0.17 / 0.02 / 0.11   |
| **TER score**                         | 91.66                | 93.06                |
| **CHRF score**                        | 17.76                | 16.56                |
| **METEOR score**                      | -                    | -                    |


## Reproducibility instrcutions:

1. Clone the repository:
   ```bash
   git clone https://github.com/walid-rehamnia/gpt-2-fine-tuning-on-Wikitext
   cd gpt-2-fine-tuning-on-Wikitext
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
   ```bash
   pip install --index-url https://test.pypi.org/simple/ pymeteor
   ```

### Fine tuning Pipelines (code):
The pipeline for both fine-tuning aproaches are "full_finetuning.py" and "lora_finetuning.py" for full and LoRA tuning respectively, the entry point is the main.py where you can start the preferred aproach  with additional hyperparameter via the arguments as follows:

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


5. Code style commands:
   After both scripts finish running, a summary table with BLEU scores, training time, and memory usage will be printed.
```bash
pylint main.py full_finetuning.py lora_finetuning.py utils.py
```
```bash
flake8 main.py full_finetuning.py lora_finetuning.py utils.py
```
```bash
isort main.py full_finetuning.py lora_finetuning.py utils.py
```
```bash
ruff check . --fix
```


```diff
- Please read the attached PDF report for more details
```

```diff
+ feel free to contact me for any question
```