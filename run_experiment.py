import torch as th
import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from utils import FastPCA, plot_layer_distributions
import argparse

parser = argparse.ArgumentParser(description="PCA experiment.")

parser.add_argument('-bm', '--base_model', type=str, help='Model name')
parser.add_argument('-c', '--component', type=str, help="Component from which activations should be taken")
parser.add_argument('-chat', '--chat', type=str, help="To add the chat prompt (one of 'none', 'base', 'safe')", default='none')

args = parser.parse_args()

# Loading activations
chat = args.chat
component = args.component
version = ''

def load_activations(base_model, model, chat, component):
    xstest_activations = []
    unsafe_activations = []
    alpaca_activations = []

    for i in range(3):
        activ_path = f"activations/safety-lora/{base_model}-lora-{model}-rs-{i+1}"
        xstest_activations.append(th.load(os.path.join(activ_path, f"xsafety_{chat}_{component}.pt")).to(th.float32)[None])
        unsafe_activations.append(th.load(os.path.join(activ_path, f"unsafe_{chat}_{component}.pt")).to(th.float32)[None, :2749])
        alpaca_activations.append(th.load(os.path.join(activ_path, f"alpaca_{chat}_{component}.pt")).to(th.float32)[None])

    return th.cat(xstest_activations, 0), th.cat(unsafe_activations, 0), th.cat(alpaca_activations, 0)


baseline_xstest, baseline_unsafe, baseline_alpaca = load_activations(args.base_model, 'base', chat, component)
inst_01_xstest, inst_01_unsafe, inst_01_alpaca = load_activations(args.base_model, '100', chat, component)
inst_03_xstest, inst_03_unsafe, inst_03_alpaca = load_activations(args.base_model, '300', chat, component)
inst_05_xstest, inst_05_unsafe, inst_05_alpaca = load_activations(args.base_model, '500', chat, component)
inst_1_xstest, inst_1_unsafe, inst_1_alpaca = load_activations(args.base_model, '1000', chat, component)
inst_2_xstest, inst_2_unsafe, inst_2_alpaca = load_activations(args.base_model, '2000', chat, component)

xstest_df = pd.read_csv(f"data/xsafety.csv")
alpaca_df = pd.read_csv(f"data/alpaca.csv", index_col=0)
unsafe_df = pd.read_csv(f"data/unsafe.csv")

alpaca_df['data'] = 'Alpaca'
unsafe_df['kind'] = 'Unsafe'
xstest_df['data'] = 'XSTest'

# Computing PCA
pca = FastPCA(n_components=400)

baseline_xstest_pca = []
baseline_unsafe_pca = []
baseline_alpaca_pca = []
inst_01_xstest_pca = []
inst_01_unsafe_pca = []
inst_01_alpaca_pca = []
inst_03_xstest_pca = []
inst_03_unsafe_pca = []
inst_03_alpaca_pca = []
inst_05_xstest_pca = []
inst_05_unsafe_pca = []
inst_05_alpaca_pca = []
inst_1_xstest_pca = []
inst_1_unsafe_pca = []
inst_1_alpaca_pca = []
inst_2_xstest_pca = []
inst_2_unsafe_pca = []
inst_2_alpaca_pca = []

for rs in range(3):
    pca.fit(th.cat([baseline_unsafe[rs, :, -1], baseline_alpaca[rs, :, -1]]))
    baseline_xstest_pca.append(pca.transform(baseline_xstest[None, rs, :, -1]))
    baseline_unsafe_pca.append(pca.transform(baseline_unsafe[None, rs, :, -1]))
    baseline_alpaca_pca.append(pca.transform(baseline_alpaca[None, rs, :, -1]))

    pca.fit(th.cat([inst_01_unsafe[rs, :, -1], inst_01_alpaca[rs, :, -1]]))
    inst_01_xstest_pca.append(pca.transform(inst_01_xstest[None, rs, :, -1]))
    inst_01_unsafe_pca.append(pca.transform(inst_01_unsafe[None, rs, :, -1]))
    inst_01_alpaca_pca.append(pca.transform(inst_01_alpaca[None, rs, :, -1]))

    pca.fit(th.cat([inst_03_unsafe[rs, :, -1], inst_03_alpaca[rs, :, -1]]))
    inst_03_xstest_pca.append(pca.transform(inst_03_xstest[None, rs, :, -1]))
    inst_03_unsafe_pca.append(pca.transform(inst_03_unsafe[None, rs, :, -1]))
    inst_03_alpaca_pca.append(pca.transform(inst_03_alpaca[None, rs, :, -1]))

    pca.fit(th.cat([inst_05_unsafe[rs, :, -1], inst_05_alpaca[rs, :, -1]]))
    inst_05_xstest_pca.append(pca.transform(inst_05_xstest[None, rs, :, -1]))
    inst_05_unsafe_pca.append(pca.transform(inst_05_unsafe[None, rs, :, -1]))
    inst_05_alpaca_pca.append(pca.transform(inst_05_alpaca[None, rs, :, -1]))

    pca.fit(th.cat([inst_1_unsafe[rs, :, -1], inst_1_alpaca[rs, :, -1]]))
    inst_1_xstest_pca.append(pca.transform(inst_1_xstest[None, rs, :, -1]))
    inst_1_unsafe_pca.append(pca.transform(inst_1_unsafe[None, rs, :, -1]))
    inst_1_alpaca_pca.append(pca.transform(inst_1_alpaca[None, rs, :, -1]))

    pca.fit(th.cat([inst_2_unsafe[rs, :, -1], inst_2_alpaca[rs, :, -1]]))
    inst_2_xstest_pca.append(pca.transform(inst_2_xstest[None, rs, :, -1]))
    inst_2_unsafe_pca.append(pca.transform(inst_2_unsafe[None, rs, :, -1]))
    inst_2_alpaca_pca.append(pca.transform(inst_2_alpaca[None, rs, :, -1]))

baseline_xstest_pca = th.cat(baseline_xstest_pca, 0)
baseline_unsafe_pca = th.cat(baseline_unsafe_pca, 0)
baseline_alpaca_pca = th.cat(baseline_alpaca_pca, 0)
inst_01_xstest_pca = th.cat(inst_01_xstest_pca, 0)
inst_01_unsafe_pca = th.cat(inst_01_unsafe_pca, 0)
inst_01_alpaca_pca = th.cat(inst_01_alpaca_pca, 0)
inst_03_xstest_pca = th.cat(inst_03_xstest_pca, 0)
inst_03_unsafe_pca = th.cat(inst_03_unsafe_pca, 0)
inst_03_alpaca_pca = th.cat(inst_03_alpaca_pca, 0)
inst_05_xstest_pca = th.cat(inst_05_xstest_pca, 0)
inst_05_unsafe_pca = th.cat(inst_05_unsafe_pca, 0)
inst_05_alpaca_pca = th.cat(inst_05_alpaca_pca, 0)
inst_1_xstest_pca = th.cat(inst_1_xstest_pca, 0)
inst_1_unsafe_pca = th.cat(inst_1_unsafe_pca, 0)
inst_1_alpaca_pca = th.cat(inst_1_alpaca_pca, 0)
inst_2_xstest_pca = th.cat(inst_2_xstest_pca, 0)
inst_2_unsafe_pca = th.cat(inst_2_unsafe_pca, 0)
inst_2_alpaca_pca = th.cat(inst_2_alpaca_pca, 0)

mask = ~xstest_df['label'].astype('bool')

baseline_pca = th.cat([baseline_alpaca_pca[..., :3], baseline_unsafe_pca[..., :3], baseline_xstest_pca[:, mask, :3]], 1)
inst_01_pca = th.cat([inst_01_alpaca_pca[..., :3], inst_01_unsafe_pca[..., :3], inst_01_xstest_pca[:, mask, :3]], 1)
inst_03_pca = th.cat([inst_03_alpaca_pca[..., :3], inst_03_unsafe_pca[..., :3], inst_03_xstest_pca[:, mask, :3]], 1)
inst_05_pca = th.cat([inst_05_alpaca_pca[..., :3], inst_05_unsafe_pca[..., :3], inst_05_xstest_pca[:, mask, :3]], 1)
inst_1_pca = th.cat([inst_1_alpaca_pca[..., :3], inst_1_unsafe_pca[..., :3], inst_1_xstest_pca[:, mask, :3]], 1)
inst_2_pca = th.cat([inst_2_alpaca_pca[..., :3], inst_2_unsafe_pca[..., :3], inst_2_xstest_pca[:, mask, :3]], 1)

y = np.concatenate([np.zeros(len(alpaca_df)), np.ones(len(unsafe_df)), 2 * np.ones(mask.sum())]) 

fig, axes = plt.subplots(2, 3, layout='constrained', figsize=(14, 9), dpi=150)
plot_xstest = True
rs = 0

palette = sns.color_palette(['#73a942', '#ef233c', '#0077b6'])

plot_layer_distributions(baseline_pca[rs], y, palette, ax=axes[0, 0], plot_xstest=plot_xstest)
axes[0, 0].set_title('Baseline')

plot_layer_distributions(inst_01_pca[rs], y, palette, ax=axes[0, 1], plot_xstest=plot_xstest)
axes[0, 1].set_title('100 Instructions')

plot_layer_distributions(inst_03_pca[rs], y, palette, ax=axes[0, 2], plot_xstest=plot_xstest)
axes[0, 2].set_title('300 Instructions')

plot_layer_distributions(inst_05_pca[rs], y, palette, ax=axes[1, 0], plot_xstest=plot_xstest)
axes[1, 0].set_title('500 Instructions')

plot_layer_distributions(inst_1_pca[rs], y, palette, ax=axes[1, 1], plot_xstest=plot_xstest)
axes[1, 1].set_title('1000 Instructions')

plot_layer_distributions(inst_2_pca[rs], y, palette, ax=axes[1, 2], plot_xstest=plot_xstest)
axes[1, 2].set_title('2000 Instructions')

plt.tight_layout()
plt.show()


# Cohesions
def overlapping_scores(activations, labels):
    # Overlapping
    if labels.dtype != th.bool:
        labels = th.tensor(labels, dtype=th.bool)
    

    safe_data = activations[~labels].to('cuda:0')
    unsafe_data = activations[labels].to('cuda:0')

    # Function to calculate average pairwise distance within a cluster
    def intra_cluster_distance(cluster_points):
        centroid = th.mean(cluster_points, dim=0)
        distances = th.sqrt(th.sum((cluster_points - centroid)**2, dim=1))
        mean_distance = th.mean(distances)
        return mean_distance

    # Calculate cohesion for each cluster
    safe_cohesion = intra_cluster_distance(safe_data)
    unsafe_cohesion = intra_cluster_distance(unsafe_data)
    
    # Calculate average cohesion
    avg_cohesion = (safe_cohesion + unsafe_cohesion) / 2

    # Calculate inter-cluster distance (separation)
    safe_centroid = th.mean(safe_data, dim=0)
    unsafe_centroid = th.mean(unsafe_data, dim=0)
    separation = th.linalg.norm(safe_centroid - unsafe_centroid)

    # Combine cohesion and separation into a single score
    # This formula can be adjusted based on specific requirements
    score = separation / avg_cohesion

    return score.cpu()

baseline_easy_pca = th.cat([baseline_alpaca_pca, baseline_unsafe_pca], 1)
inst_01_easy_pca = th.cat([inst_01_alpaca_pca, inst_01_unsafe_pca], 1)
inst_03_easy_pca = th.cat([inst_03_alpaca_pca, inst_03_unsafe_pca], 1)
inst_05_easy_pca = th.cat([inst_05_alpaca_pca, inst_05_unsafe_pca], 1)
inst_1_easy_pca = th.cat([inst_1_alpaca_pca, inst_1_unsafe_pca], 1)
inst_2_easy_pca = th.cat([inst_2_alpaca_pca, inst_2_unsafe_pca], 1)

y_easy = np.concatenate([np.zeros(len(alpaca_df)), np.ones(len(unsafe_df))]) 

components = np.arange(0, 401, 5)
y_xstest = mask.values.astype(np.int8)

score_baseline = [[], [], []]
score_inst_01 = [[], [], []]
score_inst_03 = [[], [], []]
score_inst_1 = [[], [], []]
score_inst_2 = [[], [], []]
score_inst_20 = [[], [], []]

score_baseline_easy = [[], [], []]
score_inst_01_easy = [[], [], []]
score_inst_03_easy = [[], [], []]
score_inst_05_easy = [[], [], []]
score_inst_1_easy = [[], [], []]
score_inst_2_easy = [[], [], []]

for nc in tqdm(components):
    for rs in range(3):
        score_baseline[rs].append(overlapping_scores(baseline_xstest_pca[rs, :, :nc], y_xstest))
        score_inst_01[rs].append(overlapping_scores(inst_01_xstest_pca[rs, :, :nc], y_xstest))
        score_inst_03[rs].append(overlapping_scores(inst_03_xstest_pca[rs, :, :nc], y_xstest))
        score_inst_1[rs].append(overlapping_scores(inst_05_xstest_pca[rs, :, :nc], y_xstest))
        score_inst_2[rs].append(overlapping_scores(inst_1_xstest_pca[rs, :, :nc], y_xstest))
        score_inst_20[rs].append(overlapping_scores(inst_2_xstest_pca[rs, :, :nc], y_xstest))

        score_baseline_easy[rs].append(overlapping_scores(baseline_easy_pca[rs, :, :nc], y_easy))
        score_inst_01_easy[rs].append(overlapping_scores(inst_01_easy_pca[rs, :, :nc], y_easy))
        score_inst_03_easy[rs].append(overlapping_scores(inst_03_easy_pca[rs, :, :nc], y_easy))
        score_inst_05_easy[rs].append(overlapping_scores(inst_05_easy_pca[rs, :, :nc], y_easy))
        score_inst_1_easy[rs].append(overlapping_scores(inst_1_easy_pca[rs, :, :nc], y_easy))
        score_inst_2_easy[rs].append(overlapping_scores(inst_2_easy_pca[rs, :, :nc], y_easy))

score_baseline = np.array(score_baseline).mean(0)
score_inst_01 = np.array(score_inst_01).mean(0)
score_inst_03 = np.array(score_inst_03).mean(0)
score_inst_1 = np.array(score_inst_1).mean(0)
score_inst_2 = np.array(score_inst_2).mean(0)
score_inst_20 = np.array(score_inst_20).mean(0)

score_baseline_easy = np.array(score_baseline_easy).mean(0)
score_inst_01_easy = np.array(score_inst_01_easy).mean(0)
score_inst_03_easy = np.array(score_inst_03_easy).mean(0)
score_inst_05_easy = np.array(score_inst_05_easy).mean(0)
score_inst_1_easy = np.array(score_inst_1_easy).mean(0)
score_inst_2_easy = np.array(score_inst_2_easy).mean(0)

scores_df = pd.DataFrame({
    'N Components': components,
    'Baseline': score_baseline.ravel(),
    '100': score_inst_01.ravel(),
    '300': score_inst_03.ravel(),
    '500': score_inst_1.ravel(),
    '1000': score_inst_2.ravel(),
    '2000': score_inst_20.ravel()
}).dropna()

easy_scores_df = pd.DataFrame({
    'N Components': components,
    'Baseline': score_baseline_easy.ravel(),
    '100': score_inst_01_easy.ravel(),
    '300': score_inst_03_easy.ravel(),
    '500': score_inst_05_easy.ravel(),
    '1000': score_inst_1_easy.ravel(),
    '2000': score_inst_2_easy.ravel()
}).dropna()

fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=200)

palette = sns.color_palette("Set2", as_cmap=False)

plot_df = easy_scores_df.melt(id_vars=['N Components'], value_vars=['Baseline', '100', '300', '500', '1000', '2000'], var_name='Model', value_name='Score')
sns.lineplot(data=plot_df, x='N Components', y='Score', hue='Model', palette=palette, ax=ax)
sns.scatterplot(data=plot_df, x='N Components', y='Score', hue='Model', palette=palette, ax=ax, size=0.05, legend=False)