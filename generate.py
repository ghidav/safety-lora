import argparse
import pandas as pd
import torch
import os
from simple_generation import SimpleGenerator
from tqdm import tqdm
from utils import load_model
from finetuning import Prompter

os.environ['TOKENIZERS_PARALLELISM'] = "true"
prompter = Prompter()

#  Args and logging
parser = argparse.ArgumentParser(description='Activation probing of LLMs.')

parser.add_argument("-hf", "--hf_model", help="Huggingface model to load.", type=str)
parser.add_argument("-d", "--device", help="Device to load the model.", type=str, default='cuda')
parser.add_argument("-bs", "--batch_size", help="Batch size of the prompts.", type=int, default=32)
parser.add_argument("-ds", "--dataset", help="Dataset to run the probing (must be in the /data folder!).", type=str,
                    default="")
parser.add_argument("-adp", "--adapter", help="Huggingface adapter model.", type=str, default="")

args = parser.parse_args()

# Set output path
model_folder = args.hf_model if args.adapter == "" else args.adapter
generations_path = os.path.join('generations', model_folder)
if not os.path.exists(generations_path):
    os.makedirs(generations_path)

if ':' in args.device:
    n_devices = 1
else:
    n_devices = 4

model = load_model(args.hf_model, args.adapter,
                   device=args.device, n_devices=n_devices, dtype=torch.bfloat16)

# Loading the dataset
df = pd.read_csv(f'data/{args.dataset}.csv', index_col=0)
try:
    df['input'] = df['input'].fillna('')
except: pass

prompts = df.apply(lambda row: prompter.generate_prompt(
    instruction=row['instruction'],
    input=row['input'])[:-1], axis=1).values

n = len(prompts)
if n % args.batch_size != 0:
    prompts = list(prompts) + [''] * (args.batch_size - n % args.batch_size)
    n = len(prompts)

# Generate
max_new_toks = 32
bs = args.batch_size

ans = []

for b in tqdm(range(n // bs)):
    batch_prompts = prompts[b*bs:(b+1)*bs]
    # Ensure all prompts are strings
    batch_prompts = [str(prompt) for prompt in batch_prompts]
    tokens = model.to_tokens(batch_prompts) # [bs pos]
    out = model.generate(tokens, max_new_tokens=max_new_toks, verbose=False, temperature=0)
    ans += model.tokenizer.batch_decode(out, skip_special_tokens=True)

pd.DataFrame({
    'prompt': prompts,
    'answer': ans
}).to_csv(os.path.join(generations_path, f"{args.dataset.split('.')[0]}.csv"))