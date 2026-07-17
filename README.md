# (Own) Nano-GPT-Speedrun-Track

This repo represents my Nano-GPT speedrun playground, which started coding along [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU), then moved into further improvements ([NanoGPT Speedrun Living Worklog](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/) and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)) and then own experiments.

Every new file in `versions/` adds some new functionality reflected in its name (chosen over commits over the same file to allow for independent updating in case I have to clean each file up if I ever decide to create a training set with `diff`s between them), while `train.py` holds the latest/fastest version of the code.

## Measurements

- Repo's best time to reach <= 3.28 val loss on the first 10,485,760 validation tokens (4x NVIDIA H100 SXM): 4.98 minutes (we would need to change `total_tokens_per_step_train/val` (262,144), `gpu_batch_size` (8), or `seq_len_train/val` (8,192) to use 8x NVIDIA H100).
- [Nov 2, 2025, World record <= 3.28 val loss on the first 10,485,760 validation tokens](https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#world-record-history) (8x NVIDIA H100 SXM): 2.345 minutes

## Updates

- While I haven't had much time to play around with this repo lately, I have added 2 new versions, improving this repo's best time to 4.98 minutes (4x H100 SXM), now also on the same 10M tokens the official speedrun uses.
- In this time, the official speedrun [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) has improved another 30% in 2 months and are already at 1.540 minutes (Feb 9th, 2026). Go check them out!

### WR vs This Repo Differences

- As some noticeable differences, this repo tries to have more comments and be more customizable (possibly better for: learning, adaptation to other use cases) while the world record code is (a lot) more optimized.

## Results

### Train/val loss

### GPU benchmarking (Runpod)

<img width="1000" alt="gpt-cuda-gpu-speed-cost-plot" src="https://github.com/user-attachments/assets/a219faf3-b835-446b-a8b7-c7e435c47609" />

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

### 2. (Optional) Create a Docker Image

I currently use:

- [anywinter4079/pytorch:2.10.0-cu128](https://console.runpod.io/deploy?template=undhxj45z6&ref=xqdrabty) for `versions/33-*`, `versions/34-*`, and `train.py` (PyTorch 2.10.0, CUDA 12.8)
- [anywinter4079/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-runpod-clone](https://console.runpod.io/deploy?template=14umdte6u7&ref=xqdrabty) for earlier versions (PyTorch 2.8.0, CUDA 12.8.1)

The `2.10.0-cu128` image is built from `anywinter4079/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-runpod-clone` (clone/backup of the official `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`) with extra dependencies (`tiktoken` `datasets` `tqdm` `huggingface_hub` `safetensors`) and an upgrade to PyTorch `2.10.0+cu128`.

To build and push the image, I used:

```
mkdir -p docker_build && \
cat > docker_build/Dockerfile <<'EOF'
FROM anywinter4079/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-runpod-clone

WORKDIR /workspace

RUN apt-get update && apt-get install -y nano && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
 && pip install tiktoken datasets tqdm huggingface_hub safetensors

# torch 2.10.0+cu128 (CUDA 12.8)
RUN pip install torch==2.10.0+cu128 \
 --index-url https://download.pytorch.org/whl/cu128 --upgrade
EOF
```

And then:

```
docker login
```

With:

```
docker buildx use amd64-builder
docker buildx build --platform linux/amd64 -t anywinter4079/pytorch:2.10.0-cu128 ./docker_build --push
```

If `amd64-builder` already exists (e.g., have run the command before), or:

```
docker buildx create --use --name amd64-builder
docker buildx build --platform linux/amd64 -t anywinter4079/pytorch:2.10.0-cu128 ./docker_build --push
```

otherwise.

If you want to reuse the existing image, there is nothing to do in this step. Else, you can follow these steps to create your own, if you ever want to change the image.

### 3. (Optional) Create a network volume to host the Fineweb-Edu data

To create a persistent network volume to host the ~20GB of data ($1.40/mo) for instant access upon pod spinup:

Clone and `cd` into this repo:

```
git clone https://github.com/Any-Winter-4079/Nano-GPT-Speedrun-Track.git && \
cd Nano-GPT-Speedrun-Track
```

Download the Fineweb-Edu dataset (locally):

```
cd data && python fineweb-npy.py
```

Create a Network volume on Runpod:

1. Click `Storage` (left menu)
2. Click `+ New Network Volume`
3. Choose a datacenter with H100 SXM(s) available
4. Set the size to 20 GB
5. Click Create Network Volume

Start a CPU pod (cheaper) on the same datacenter where the network volume is hosted:

1. Click `Storage` (left menu)
2. Click the volume
3. Click `Configure Pod with Volume`
4. Click `Edit Pod`
5. Change `Volume Mount Path` from `/workspace` to `/workspace/data/edu_fineweb10B`
6. Click `Set Overrides`
7. Click `Deploy On-Demand`

Then securely copy (`scp`) the data from your local computer to the remote network volume, replacing `<PORT>`, `<PRIVATE_KEY_FILE>`, `<IP>` with those from `Connect`:

```
scp -P <PORT> -i ~/.ssh/<PRIVATE_KEY_FILE> -r \
    data/edu_fineweb10B/* \
    root@<IP>:/workspace/data/edu_fineweb10B/
```

### 4. Pretrain

To pre-train, terminate the CPU pod, and spin up a pod with one or more H100 SXM(s), and add `hf_user` and `hf_token` (if you want to push your final model/training logs to the Hub), i.e.:

1. Click `Edit`
2. Click `Environment Variables`
3. Click `+ Add Environment Variable`
4. Click Lock icon to select secret
5. Click `hf_token` to select it as value
6. Type `hf_token` to choose it as key
7. Click `+ Add Environment Variable`
8. Click Lock icon to select secret
9. Click `hf_user` to select it as value
10. Type `hf_user` to choose it as key
11. Change `Volume Mount Path` from `/workspace` to `/workspace/data/edu_fineweb10B`
12. Click `Set Overrides`
13. Click `Deploy On-Demand`

Connect to the pod via ssh (replacing `<CONNECTION_STRING>`, `<PRIVATE_KEY_FILE>` with those provided by Runpod):

```
ssh <CONNECTION_STRING>@ssh.runpod.io -i ~/.ssh/<PRIVATE_KEY_FILE>
```

Fetch the remote repo initializing a local repo beforehand to combine the `data/` folders (replacing `<your-user>` with your GitHub username), running the following within the workspace (cd `workspace` if needed):

```
git init && \
git remote add origin https://github.com/<your-user>/Nano-GPT-Speedrun-Track.git && \
git remote add upstream https://github.com/Any-Winter-4079/Nano-GPT-Speedrun-Track.git && \
git fetch origin && \
git checkout -b main origin/main
```

or in my case:

```
git init && \
git remote add origin https://github.com/Any-Winter-4079/Nano-GPT-Speedrun-Track.git && \
git fetch origin && \
git checkout -b main origin/main
```

Then either check the network volume has been successfully attached:

```
ls data/edu_fineweb10B
```

Or download the data within the pod (if you skipped step 3):

```
cd data && \
python fineweb-npy.py && \
cd ..
```

Finally, run (replacing `4` with the number of GPUs you decided to use):

```
torchrun --standalone --nproc_per_node=4 train.py
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

### 5. (Optional) Store `stdout` and `stderr` logs on file (if you run updates to the code and the log is too long to be read on the shell)

Because error messages can be **very** long, you can run (replacing `--nproc_per_node=4` with you GPU count):

```
mkdir debug
```

```
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=1 train.py \

> > (stdbuf -oL tee -a debug/out-$(date +%F_%H-%M-%S).log) \
  2> >(stdbuf -oL tee -a debug/err-$(date +%F\_%H-%M-%S).log >&2)
```

Then review the error message in chunks (replacing `err-2025-10-25_00-32-19.log` with your own log):

```
cd debug
awk 'NR>=0 && NR<=180' err-2025-10-25_00-32-19.log
```

### 6. (Optional) Push the training config, logs, and checkpoint(s) to Hugging Face

To push the config, log, and model(s) (if checkpointing), either provide a timestamp explicitly:

```
python push_to_hub.py --timestamp 20260211_180749
```

Or let the script auto-pick the latest timestamp from `./configs_and_logs` and `./checkpoints`:

```
python push_to_hub.py
```

Optional: include metadata in the Hub repo name (GPU count, final val loss, and total train minutes parsed from logs/config):

```
python push_to_hub.py --timestamp 20260211_180749 --with-metrics-in-name
```
