# Llama-2-7b runs (10gp + [0s, 01, 05, 1, 2, 20])

# generations
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g0s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g01s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g05s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g1s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g2s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g20s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g0s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g01s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g05s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g1s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g2s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g20s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g0s-rs1 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g01s-rs1 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g05s-rs1 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g1s-rs1 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g2s-rs1 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g20s-rs1 -ds xsafety -chat safety-by-imitation -bs 16

python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g0s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g01s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g05s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g1s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g2s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g20s-rs1 -ds alpaca -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g0s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g01s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g05s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g1s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g2s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g20s-rs1 -ds unsafe -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g0s-rs1 -ds xsafety -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g01s-rs1 -ds xsafety -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g05s-rs1 -ds xsafety -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g1s-rs1 -ds xsafety -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g2s-rs1 -ds xsafety -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf safety-by-imitation/llama-2-7b-hf-full-10g20s-rs1 -ds xsafety -chat safety-by-imitation -bs 16

# adapters
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-3 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-3 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-3 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-3 -ds alpaca -chat safety-by-imitation -bs 16

python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-3 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-3 -ds alpaca -chat safety-by-imitation -bs 16

python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-3 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-3 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-3 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-3 -ds unsafe -chat safety-by-imitation -bs 16

python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-3 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-3 -ds unsafe -chat safety-by-imitation -bs 16

python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-3 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-3 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-3 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-3 -ds xsafety -chat safety-by-imitation -bs 16

python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-3 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
python generate.py -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-3 -ds xsafety -chat safety-by-imitation -bs 16



#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-3 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-3 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-3 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-3 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-3 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-1 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-2 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-3 -ds alpaca -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-3 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-3 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-3 -ds unsafe -chat safety-by-imitation -bs 16



#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-3 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-3 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-1 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-2 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-3 -ds unsafe -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-base-rs-3 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-100-rs-3 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-300-rs-3 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-500-rs-3 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-1000-rs-3 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-1 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-2 -ds xsafety -chat safety-by-imitation -bs 16
#python get_activations.py -c mlp.hook_post -tl Llama-2-7b-hf -hf meta-llama/Llama-2-7b-hf -adp safety-lora/Llama-2-7b-hf-lora-2000-rs-3 -ds xsafety -chat safety-by-imitation -bs 16