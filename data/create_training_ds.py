"""
Scritp to create the training datasets.
They will be 6: 
- baseline: made of 20k instructions and questions from alpaca datastet (without inputs).
- 01: baseline + 100 safety examples
- 05: baseline + 500 safety examples
- 1: baseline + 1000 safety examples
- 2: baseline + 2000 safety examples
- 5: baseline + 5000 safety examples

All the safety datasets are 50% insturctions and 50% questions.
"""
#%%
import pandas as pd

random_state = 42

# Alpaca
alpaca = pd.read_json('training/alpaca_data_cleaned.json')
alpaca = alpaca[alpaca['input'] == '']
alpaca['is_question'] = alpaca['instruction'].apply(lambda x: '?' in x)
alpaca_quest = alpaca[alpaca['is_question']].sample(3800, replace=False, random_state=random_state)
alpaca_instr = alpaca[~alpaca['is_question']].sample(17200, replace=False, random_state=random_state)
baseline = pd.concat([
    alpaca_quest[:3500], 
    alpaca_instr[:16500]]).drop(columns=['is_question'])
alpaca_eval = pd.concat([
    alpaca_quest[3500:], 
    alpaca_instr[16500:]]).drop(columns=['is_question'])
#%%
# Safety
from datasets import load_dataset
pku = load_dataset('PKU-Alignment/PKU-SafeRLHF', split='train')
# %%
def filter_func(x):
    return (~x['is_response_0_safe'] or ~x['is_response_1_safe']) and (x['prompt'].endswith('?') or x['prompt'].endswith('.'))

def instruction_filter(x):
    inst_verbs = [
    "analyze", "investigate", "research", "provide", "examine", "identify", "explain", "outline",
    "compare", "contrast", "summarize", "describe", "evaluate", "assess", "interpret", "discuss",
    "predict", "determine", "calculate", "demonstrate", "illustrate", "clarify", "classify", "justify",
    "review", "synthesize", "report", "suggest", "recommend", "construct", "develop", "design",
    "create", "plan", "implement", "integrate", "measure", "test", "validate", "verify", "argue",
    "debate", "critique", "reflect", "observe", "note", "record", "list", "define", "enumerate",
    "categorize", "prioritize", "organize", "manage", "lead", "facilitate", "coordinate", "optimize",
    "improve", "enhance", "revise", "edit", "adapt", "modify", "expand", "extend", "reduce", "minimize",
    "increase", "maximize", "estimate", "forecast", "model", "map", "explore", "navigate", "search",
    "find", "locate", "select", "choose", "decide", "solve", "resolve", "address", "tackle", "overcome",
    "achieve", "accomplish", "attain", "reach", "gain", "secure", "acquire", "obtain", "gather",
    "collect", "compile", "assemble", "generate", "produce", "create", "initiate", "launch", "establish",
    "formulate", "propose", "advise", "consult", "inform", "notify", "alert", "warn", "remind",
    "instruct", "teach", "train", "mentor", "coach", "guide", "assist", "help", "support", "benefit",
    "serve", "protect", "safeguard", "preserve", "maintain", "sustain", "keep", "hold", "conserve",
    "save", "restore", "renew", "revitalize", "revive", "rejuvenate", "refresh", "enrich", "empower",
    "strengthen", "build", "construct", "forge", "shape", "cut", 
    "press", "push", "pull", "lift", "lower", "raise", "drop", "throw", "release", "let go", "place", "put",
    "set", "position", "install", "deploy", "apply", "utilize", "use", "employ", "operate",
    "control", "command", "direct", "guide", "drive", "navigate", "say", "apologize", "forgive", "praise", "look", "observe", "inspect"]

    for verb in inst_verbs:
        if verb in x['prompt']:
            return True

pku = pku.filter(filter_func).filter(instruction_filter)
# %%