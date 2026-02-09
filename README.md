# (Own) Nano-GPT-Speedrun-Track

This repo starts coding along [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU), then moves into further improvements ([NanoGPT Speedrun Living Worklog](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/) and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)) and own experiments. Every new file in `versions/` adds some new functionality (done this way -vs. commits over the same file- to maintain easier updatability of older versions, in case one day I create a training set for an LLM with `diff`s between consecutive files, e.g., 'Add DDP to this code'. However, currently there are _minor_ changes between certain pairs of files beyond what their name suggests, such as minor comment changes, etc., which we'd need to polish to avoid rewarding the language model e.g., for changes outside 'Add DDP to this code')

## Measurements

- Repo's best time to reach <= 3.28 val loss on 10,485,760 validation tokens (4x NVIDIA H100 SXM): 4.87 minutes (we would need to change `total_tokens_per_step_train/val` (262,144), `gpu_batch_size` (8), or `seq_len_train/val` (8,192) to use 8x NVIDIA H100s, but it is a bit costly anyway, about $20/h).
- [Nov 2, 2025, World record <= 3.28 val loss on _the first 10,485,760 validation tokens](https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#world-record-history) (8x NVIDIA H100 SXM): 2.345 minutes

## Updates
- While I haven't had much time to play around with this repo lately, I have added 2 new versions, improving this repo's best time to X, now also on the same 10M tokens the official speedrun uses.
- In this time, the official speedrun [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) has improved another 30% in 2 months and are already at 1.540 minutes (Feb 9th, 2026). Go check them out!

### WR vs This Repo Differences

- As some noticeable differences, this repo tries to be more customizable (better for: learning, customization) while the world record code is (a lot) more optimized (Triton kernels, etc.) but also tries to remove anything that hurts performance (e.g., checkpointing, extra logging, etc.)

## Configuration Options

This repo supports toggling between the following config options:

- FlashAttention (sdpa) / FlexAttention
- Sliding Window Attention (attend to a subset of tokens), Doc Masking (attend to same-doc tokens only), and Attention Logit Soft-capping (if FlexAttention, for performance)
  - Sliding Window Attention ramp (increase window size over training)
  - Attention logit soft-capping ("clamp", "ptx" -faster-, "rational" or "exact")
- Custom masking (e.g., padding mask if non-causal)
- AdamW or AdamW and Muon
  - Muon steps, momentum, use Nesterov
- MHA/MQA/GQA (n_heads vs n_kv_heads)
- QK norm (RMS/L2)
- RMSNorm or LayerNorm
- GELU, ReLU, ReLU\*\*2, SiLU or SwiGLU (fair or unfair) activations
- Bias or no bias
- Tied or untied embeddings
- Learning rate warmup and decay
- RoPE/NoPE/absolute positional encodings
- LM head logit soft-capping
- Gradient norm clipping
- Kernel warmup steps
- ...

## Supports

- Pre-training (stops at val_target)
- Validation (shuffle or always use the same tokens as per speedrun record)
- Sampling (increasingly longer sequences as the model learns)
- Evaluation (HellaSwag one-shot or standard -NOTE: one-shot may require bigger models than 114-124M to work)
- Checkpointing (store checkpoint, resume from checkpoint)
- Logging (to stdout and to file)

## Possible improvements

- KV cache (for `sample`)
- Mixture of Experts config option
- Masked Language Model (MLM) training support (`is_causal=False` -> work as encoder)

- Faster shard-switching (100M tokens/shard) time:

```
step: 380 | train loss: 3.99275875 | train ppl: 54.20 | train step time: 158.82 ms | adamw lr: 0.00483939 | tok/s: 1,650,578.83 | total toks: 99,876,864 | total time: 0.99 min | sw size: 256 | max q_scale raw/eff: 1.8034/1.8034 | max k_scale raw/eff: 1.8034/1.8034
step: 381 | train loss: 3.94825888 | train ppl: 51.85 | train step time: 251.45 ms | adamw lr: 0.00483856 | tok/s: 1,042,512.27 | total toks: 100,139,008 | total time: 1.00 min | sw size: 256 | max q_scale raw/eff: 1.8039/1.8039 | max k_scale raw/eff: 1.8039/1.8039
step: 382 | train loss: 3.97589111 | train ppl: 53.30 | train step time: 159.01 ms | adamw lr: 0.00483772 | tok/s: 1,648,631.07 | total toks: 100,401,152 | total time: 1.00 min | sw size: 256 | max q_scale raw/eff: 1.8043/1.8043 | max k_scale raw/eff: 1.8043/1.8043
```

- Reward >1 token at the start of training (e.g., I went `swimming` -> reward `swimming`, `running`, `walking`, etc.) and sharpen signal as training goes on
- ...

## Instructions

I currently use RunPod to run this repo (with other options including: [Lambda](https://lambda.ai/), [Vast.ai](https://vast.ai/)). If you want to run this code on another provider, or your own hardware, feel free to skip RunPod-specific instructions, but stay around for `hf_user` and `hf_token` setup, and the Docker image used.

### 1. Create `hf_user` and `hf_token`

On RunPod, create the following secrets if you want to push your results (and model(s) if checkpointing) to Hugging Face after pre-training.

1. Click `Secrets` (left menu)
2. Click `+ Create Secret`
3. Type `hf_user` as Secret Name
4. Paste or type your Hugging Face username as Secret Value
5. Click Create Secret (to create the secret)
6. Click `+ Create Secret`
7. Type `hf_token` as Secret Name
8. Paste or type your Hugging Face token as Secret Value
9. Click Create Secret (to create the secret)

If you use your own hardware, create them as environment variables

### 2. Choose/create a (RunPod) template, or pull a Docker image

If you use your own hardware, you can:

Use one of the following Docker images (first is the official image, second is a personal backup/clone for my own future sanity, in the rare event RunPod deleted the image, etc.):

- `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- `anywinter4079/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-runpod-clone`

Or manually match that config (e.g., Python 3.11, CUDA 12.8.1, etc.) on your own system, or try your system as-is (no guarantees)

If you need third-party hardware, you will want a Docker image, some container/volume disk space, and an SSH connection.
On RunPod:

Feel free to create your own template, or [deploy my template](https://console.runpod.io/deploy?template=undhxj45z6&ref=xqdrabty) (Disclaimer: I theoretically get 1% of earnings generated from my template, but the main reason for creating it is to more quickly deploy each time -feel free to create a new one, clone mine, etc.). The template I created uses config:

1. Container Image: anywinter4079/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-runpod-clone
2. Container Disk: 250 GB
3. Volume Disk: 500 GB
4. Volume Mount Path: /workspace
5. TCP Ports (max 10): Port label: SSH, Port number: 22

If you create your own template, you may want to use RunPod's official container image:

```
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

After you have the template (mine or your own), deploy, setting `hf_user` and `hf_token` as Environment variables (note `hf_user` and `hf_token` are added as Environment variables on the deployment page, and not to the template itself, in case the template is made public -e.g., my template is public-). To deploy a template on RunPod:

1. On Deploy a Pod: Choose `H100 XSM` (on `All North America` they seem to be faster/more reliable) and click `Edit Template`
2. Select your GPU count
3. Click `Edit`
4. Click `Environment Variables`
5. Click `+ Add Environment Variable`
6. Click Lock icon to select secret
7. Click `hf_token` to select it as value
8. Type `hf_token` to choose it as key
9. Click `+ Add Environment Variable`
10. Click Lock icon to select secret
11. Click `hf_user` to select it as value
12. Type `hf_user` to choose it as key
13. Click `Set Overrides`
14. Click `Deploy On-Demand`

Finally, connect to the RunPod template via SSH (without SCP & SFTP support), running on your Terminal the provided command, in the following format (replacing `<CONNECTION_STRING>` and `<PRIVATE_KEY_FILE>` with those provided by RunPod):

```
ssh <CONNECTION_STRING>@ssh.runpod.io -i ~/.ssh/<PRIVATE_KEY_FILE>
```

### 3. Install requirements and download `edu_fineweb10B`

To set up the remote machine, before pre-training:

1. Run:

```
cd /workspace && \
apt update && \
apt install -y nano && \
mkdir data && \
cd data && \
nano fineweb-npy.py
```

2. Add `fineweb-npy.py`'s content (from this repo)

3. Save (e.g., `control + x`, `y`, `enter`)

4. Run:

```
python -m pip install --upgrade pip && \
pip install tiktoken datasets tqdm huggingface_hub safetensors && \
python fineweb-npy.py
```

Alternatively, for steps 2 and 3, you could (on RunPod's `workspace`):

```
git clone https://github.com/Any-Winter-4079/GPT-3-Small-Pretraining-Experiments.git
```

to get _all code versions_ into the workspace (if you want to test multiple or do not mind the bloating)

Or clone the repo locally:

```
git clone https://github.com/Any-Winter-4079/GPT-3-Small-Pretraining-Experiments.git
```

`cd` into it:

```
cd GPT-3-Small-Pretraining-Experiments
```

and, on another Terminal window (locally, keeping the other SSH session open), send the files via `scp`, e.g., with (replacing `<PORT>`, `<PRIVATE_KEY_FILE>` and `<IP>` with the values from 'SSH over exposed TCP' on RunPod, and replacing `30-gpt-3-small-with-training-config-and-with-or-without-swa-window-size-ramp.py` with the version you want to send to run):

```
scp -P <PORT> -i ~/.ssh/<PRIVATE_KEY_FILE> -r \
    versions/30-gpt-3-small-with-training-config-and-with-or-without-swa-window-size-ramp.py \
    input.txt \
    data \
    push_to_hub.py \
    root@<IP>:/workspace/
```

Alternatively, use `train.py` for the latest script version.

## Pre-training

Finally, to pre-train, `cd` out of the `data/` directory (on RunPod):

```
cd ..
```

And either create the file (e.g., `30.py` for brevity) or if you already have it because you have `git clone`'d the repo or `scp`'d the file, go directly to running `torchrun`, specifying your GPU count, and the file name you want to run, e.g.:

```
nano 30.py
```

Add `versions/30-gpt-3-small-with-training-config-and-with-or-without-swa-window-size-ramp.py`'s content (from this repo)

Save (e.g., `control + x`, `y`, `enter`)

And run:

```
torchrun --standalone --nproc_per_node=4 30.py
```

You should see a log similar to:

```
per-gpu gradient accumulation mini-steps: 1
lr warmup steps: 0
lr warmup and cosine steps: 3,051
max train steps: 19,073
num decayed parameter tensors (AdamW): 25, with 38,633,760 parameters
num non-decayed parameter tensors (AdamW): 25, with 19,200 parameters
num Muon parameter tensors: 72, with 84,934,656 parameters
using fused AdamW: True
found 99 shards for split train
total train shard 0 tokens: 99,876,865
total gpus: 4
gpu batch size: 8 sequences
sequence length: 8,192 tokens
tokens fed to gpu per grad accum mini-step: 65,536 (4 gpus, 262,144 total tokens)
per-gpu grad accumulation mini-steps for train shard 0 (each mini-step processing 262,144 tokens): 381
found 1 shards for split val
total val shard 0 tokens: 99,876,865
total gpus: 4
gpu batch size: 8 sequences
sequence length: 8,192 tokens
tokens fed to gpu per grad accum mini-step: 65,536 (4 gpus, 262,144 total tokens)
per-gpu grad accumulation mini-steps for val shard 0 (each mini-step processing 262,144 tokens): 381
[rank 0] gets HellaSwag sentences 0 to 2,510
[rank 1] gets HellaSwag sentences 2,511 to 5,021
[rank 2] gets HellaSwag sentences 5,022 to 7,532
[rank 3] gets HellaSwag sentences 7,533 to 10,041
123,587,616 parameters
```

Then the code takes 2-4 minutes to compile, and then:

```
step: 0 | train loss: 10.97804832 | train ppl: 58,574.12 | train step time: 216.32 ms | adamw lr: 0.00500000 | tok/s: 1,211,858.15 | total toks: 262,144 | total time: 0.00 min | sw size: 128 | max q_scale raw/eff: 1.0045/1.0045 | max k_scale raw/eff: 1.0045/1.0045
step: 1 | train loss: 8.87146950 | train ppl: 7,125.74 | train step time: 153.80 ms | adamw lr: 0.00500000 | tok/s: 1,704,460.43 | total toks: 524,288 | total time: 0.01 min | sw size: 128 | max q_scale raw/eff: 1.0090/1.0090 | max k_scale raw/eff: 1.0090/1.0090
step: 2 | train loss: 7.93997669 | train ppl: 2,807.30 | train step time: 155.61 ms | adamw lr: 0.00500000 | tok/s: 1,684,600.25 | total toks: 786,432 | total time: 0.01 min | sw size: 128 | max q_scale raw/eff: 1.0130/1.0130 | max k_scale raw/eff: 1.0130/1.0130
step: 3 | train loss: 8.02733135 | train ppl: 3,063.56 | train step time: 154.30 ms | adamw lr: 0.00499999 | tok/s: 1,698,903.45 | total toks: 1,048,576 | total time: 0.01 min | sw size: 128 | max q_scale raw/eff: 1.0158/1.0158 | max k_scale raw/eff: 1.0158/1.0158
step: 4 | train loss: 7.80339432 | train ppl: 2,448.90 | train step time: 155.14 ms | adamw lr: 0.00499998 | tok/s: 1,689,747.02 | total toks: 1,310,720 | total time: 0.01 min | sw size: 128 | max q_scale raw/eff: 1.0177/1.0177 | max k_scale raw/eff: 1.0177/1.0177
step: 5 | train loss: 7.81453323 | train ppl: 2,476.33 | train step time: 154.13 ms | adamw lr: 0.00499997 | tok/s: 1,700,782.44 | total toks: 1,572,864 | total time: 0.02 min | sw size: 128 | max q_scale raw/eff: 1.0202/1.0202 | max k_scale raw/eff: 1.0201/1.0201
...
```

## Pushing the logs and checkpoints to Hugging Face

To push the config, log, and model(s) (if checkpointing), find the latest timestamp, either on `checkpoints` or `configs_and_logs`:

```
cd checkpoints
```

Copy the timestamp

```
cd ..
```

Create `push_to_hub.py` (adding its content from this repo) and/or edit it (e.g., if you have `scp`'d the script):

```
nano push_to_hub.py
```

Replacing the `timestamp` `("20251013_145053")` with your timestamp (**TODO**: make the script ask for the timestamp as input when running)

Save (e.g., `control + x`, `y`, `enter`)

And run:

```
python push_to_hub.py
```

## Storing `stdout` and `stderr` on file (if you ever need to)

Because error messages can be **very** long, you can run (replacing `--nproc_per_node=1` with you GPU count):

```
mkdir debug
```

```
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=1 30.py \
  > >(stdbuf -oL tee -a debug/out-$(date +%F_%H-%M-%S).log) \
  2> >(stdbuf -oL tee -a debug/err-$(date +%F_%H-%M-%S).log >&2)
```

Then review the error message in chunks (replacing `err-2025-10-25_00-32-19.log` with your own log):

```
cd debug
awk 'NR>=0 && NR<=180' err-2025-10-25_00-32-19.log
```
