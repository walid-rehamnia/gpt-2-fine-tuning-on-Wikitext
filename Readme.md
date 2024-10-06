## Task objective
The objective from this task is to fine tune the "GPT-2" LLM model on Wikitext dataset with simple fine tuning as well as with LoRA technique.
For simplicity and demonstrative purposes the code is implemented 






name	               train	   validation	test
wikitext-2-raw-v1	   36718	   3760	      4358


## GPT-2 Fine-Tuning Comparison

This project compares the full fine-tuning and LoRA fine-tuning of the GPT-2 model on the WikiText dataset. Below is the comparison of key metrics:

| Metric              | Full Fine-Tuning | LoRA Fine-Tuning |
|---------------------|------------------|------------------|
| BLEU Score          | X.XXX            | Y.YYY            |
| Training Time (s)   | 3236.96 seconds  | 2851.43 seconds  |
| Memory Usage (MB)   | XXX MB           | YYY MB           |
| Trainable params    | 124,439,808      | 405,504          |
| Best training loss  | 0.3204           | 0.3761           |
| Best validation loss| 0.4386           | 0.4451           |

Bleu of lora 435 examples: 
BLEU score: BLEU = 35.36 75.0/33.3/25.0/25.0 (BP = 1.000 ratio = 1.000 hyp_len = 4 ref_len = 4)

Bleu of full 435 examples: 
BLEU score: BLEU = 0.00 50.0/50.0/0.0/0.0 (BP = 1.000 ratio = 1.000 hyp_len = 2 ref_len = 2)

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
   ```bash
   pip install --index-url https://test.pypi.org/simple/ pymeteor
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


minimal run
   ```bash
   python main.py lora --batch_size 4 --data_fraction 0.01 --epochs 2 --lora_rank 2
   ```

python main.py lora --batch_size 4 --data_fraction 0.1 --epochs 2 --lora_rank 4
trainable params: 405,504 || all params: 124,845,312 || trainable%: 0.3248

Training time: 2851.43 seconds
GPU memory used: 19.34 MB
CPU memory used: -84.88 MB
#########################Losses###################
[3.3447, 0.4993, 0.5217, 0.5506, 0.4732, 0.5142, 0.3847, 0.5568, 0.4541, 0.3877, 0.5131, 0.3716, 0.5211, 0.3769, 0.5216, 0.4257, 0.4479, 0.4955, 0.4415, 0.4635, 0.5608, 0.5376, 0.4882, 0.4685, 0.3761, 0.3769, 0.4271, 0.4146, 0.44, 0.3655, 0.3786, 0.4613, 0.466, 0.5273, 0.414, 0.4739]
[0.5106523633003235, 0.49505043029785156, 0.486192911863327, 0.4795791804790497, 0.4752623438835144, 0.47098439931869507, 0.4683157801628113, 0.46570006012916565, 0.4630976617336273, 0.4613170921802521, 0.4592796564102173, 0.4576399028301239, 0.45664435625076294, 0.4555965065956116, 0.45435577630996704, 0.4532112777233124, 0.4525558054447174, 0.4516589343547821, 0.45068058371543884, 0.4501141905784607, 0.44942522048950195, 0.44873133301734924, 0.4479249119758606, 0.4477959871292114, 0.4474676847457886, 0.4468206763267517, 0.4466492533683777, 0.446226567029953, 0.44606468081474304, 0.4459221065044403, 0.4455977976322174, 0.44541794061660767, 0.44537782669067383, 0.4451855719089508, 0.4451984763145447, 0.4451241195201874]
Save the model...

python main.py full --batch_size 4 --data_fraction 0.1 --epochs 2
trainable params: 124439808 || all params: 124439808 || trainable%: 100.0
Training time: 3236.96 seconds
GPU memory used: 975.40 MB
CPU memory used: 257.81 MB
Save the model...
loss:0.32 / 'eval_loss': 0.43






python main.py full --batch_size 4 --data_fraction 0.1 --epochs 2

Training time: 3694.07 seconds
GPU memory used: 975.40 MB
CPU memory used: -24.15 MB
#########################Losses###################
[2.1174, 0.4466, 0.4769, 0.512, 0.4427, 0.4835, 0.3608, 0.5305, 0.4281, 0.3683, 0.4886, 0.3509, 0.5, 0.3603, 0.4988, 0.4028, 0.4319, 0.4737, 0.3784, 0.3784, 0.4624, 0.4441, 0.3998, 0.3862, 0.3079, 0.3108, 0.3543, 0.3474, 0.3612, 0.3004, 0.313, 0.38, 0.3883, 0.434, 0.3398, 0.3912]
[0.46279191970825195, 0.4546510875225067, 0.45132365822792053, 0.44865044951438904, 0.4493260979652405, 0.4471680521965027, 0.44842642545700073, 0.445149689912796, 0.44527724385261536, 0.44442644715309143, 0.4451478123664856, 0.4422377347946167, 0.44156062602996826, 0.44128841161727905, 0.44036415219306946, 0.43953973054885864, 0.4393884241580963, 0.437975138425827, 0.44524630904197693, 0.4449363350868225, 0.44536733627319336, 0.4455376863479614, 0.4453529119491577, 0.44572919607162476, 0.4451650381088257, 0.44514092803001404, 0.44500768184661865, 0.4445052444934845, 0.44450056552886963, 0.4453565180301666, 0.44525739550590515, 0.4443413019180298, 0.44417569041252136, 0.4440059959888458, 0.4436773359775543, 0.4435123801231384]


5. Check the comparison results:
   After both scripts finish running, a summary table with BLEU scores, training time, and memory usage will be printed.

pylint main.py full_finetuning.py lora_finetuning.py utils.py
flake8 main.py full_finetuning.py lora_finetuning.py utils.py
isort main.py full_finetuning.py lora_finetuning.py utils.py