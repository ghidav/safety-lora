import torch as th
import pandas as pd
import argparse
import os
from finetuning import Prompter

from utils import convert_to_chat, get_activations, convert_to_alpaca, load_model, convert_to_sbi

parser = argparse.ArgumentParser(description="PCA experiment.")

parser.add_argument('-hf', '--hf_model', type=str, help='HF model to load')
parser.add_argument('-c', '--component', type=str, help="Component from which activations should be taken")
parser.add_argument('-ds', '--dataset', type=str, help="Dataset to load")

parser.add_argument('-adp', '--adapter', type=str, help='Adapter to load', default="")
parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=32)
parser.add_argument('-nd', '--n_devices', type=int, help='N devices', default=0)

args = parser.parse_args()

prompter = Prompter()

if args.n_devices == 0: args.n_devices = th.cuda.device_count()

# Creating dirs
model_folder = args.hf_model if args.adapter == "" else args.adapter
activ_path = os.path.join('activations', model_folder)
if not os.path.exists(activ_path):
    os.makedirs(activ_path)

# Data loading
df = pd.read_csv(f"data/{args.dataset}.csv", index_col=0)
try:
    df['input'] = df['input'].fillna('')
except: pass

# Model loading
model = load_model(args.hf_model, args.adapter, device='cuda', n_devices=args.n_devices, dtype=th.bfloat16)
model.eval()

prompts = df.apply(lambda row: prompter.generate_prompt(
    instruction=row['instruction'],
    input=row['input'])[:-1], axis=1).values

print("Caching activations...")
activations = get_activations(model, prompts, args.component, bs=args.batch_size).cpu()
th.save(activations, os.path.join(activ_path, f"{args.dataset}_{args.component}.pt"))
print("Activations cached.")