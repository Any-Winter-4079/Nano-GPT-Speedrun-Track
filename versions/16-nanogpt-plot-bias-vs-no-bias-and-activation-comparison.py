import os
import re
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

bin_width = 1000
val_interval = 100
repo_base_name = "gpt-3-small_"

timestamps_all = [
    "20250803_164625",
    "20250803_171453",
    "20250803_174849",
    "20250803_181857",
    "20250804_141414",
    "20250804_141414",
]

timestamp_name_map = {
    "20250803_164625": "GELU: 124,475,904 params",
    "20250803_171453": "ReLU: 124,475,904 params",
    "20250803_174849": "ReLU²: 124,475,904 params",
    "20250803_181857": "SiLU: 124,475,904 params",
    "20250804_141414": "SwiGLU: 124,488,192 params (more fair)",
    "20250804_141414": "SwiGLU: 152,824,320 params (unfair)",
}

load_dotenv()
hf_user = os.getenv("hf_user")
hf_token = os.getenv("hf_token")

train_re = re.compile(
    r"step:\s*([\d,]+)\s*\|.*?train loss:\s*([0-9.]+).*?grad norm:\s*([0-9.]+)"
    r".*?lr:\s*([0-9.e-]+).*?tok/s:\s*([\d.,]+).*?total toks:\s*([\d,]+)"
    r".*?total time:\s*([\d.,]+)"
)
val_re = re.compile(
    r"step:\s*([\d,]+)\s*\|.*?val loss:\s*([0-9.]+).*?val ppl:\s*([\d.,]+).*?val time:\s*([\d.,]+)"
)
best_val_re = re.compile(r"new best val loss:\s*([0-9.]+)")

all_train_dfs = []
all_val_dfs = []

for ts in timestamps_all:
    repo_id = f"{hf_user}/{repo_base_name}{ts}"
    try:
        log_path = hf_hub_download(
            repo_id=repo_id,
            filename="log.txt",
            repo_type="model",
            token=hf_token,
            # force_download=True,
        )
        with open(log_path, "r") as f:
            lines = f.readlines()

        train_data, val_data = [], []
        best_val_losses = []

        for line in lines:
            m_train = train_re.search(line)
            if m_train:
                step, loss, norm, lr, tok_s, total_toks, total_time = m_train.groups()
                train_data.append({
                    "step": int(step.replace(",", "")),
                    "train_loss": float(loss),
                    "norm": float(norm),
                    "lr": float(lr),
                    "tok_s": float(tok_s.replace(",", "")),
                    "total_toks": int(total_toks.replace(",", "")),
                    "total_time_min": float(total_time.replace(",", "")),
                    "timestamp": ts
                })
                continue

            m_val = val_re.search(line)
            if m_val:
                step, val_loss, val_ppl, val_time_ms = m_val.groups()
                val_data.append({
                    "step": int(step.replace(",", "")),
                    "val_loss": float(val_loss),
                    "val_ppl": float(val_ppl.replace(",", "")),
                    "val_time_ms": float(val_time_ms.replace(",", "")),
                    "timestamp": ts
                })

            m_best = best_val_re.search(line)
            if m_best:
                best_val_losses.append(float(m_best.group(1)))

        if best_val_losses:
            best_val_loss = best_val_losses[-1]
            last_train_step = train_data[-1]["step"] if train_data else 0
            val_data.append({
                "step": last_train_step,
                "val_loss": best_val_loss,
                "val_ppl": None,
                "val_time_ms": None,
                "timestamp": ts
            })

        df_train = pd.DataFrame(train_data)
        df_val = pd.DataFrame(val_data)
        all_train_dfs.append(df_train)
        all_val_dfs.append(df_val)

    except Exception as e:
        print(f"failed to process {ts}: {e}")

df_train_all = pd.concat(all_train_dfs, ignore_index=True)
df_val_all = pd.concat(all_val_dfs, ignore_index=True)

df_train_all["name"] = df_train_all["timestamp"].map(timestamp_name_map)
df_val_all["name"] = df_val_all["timestamp"].map(timestamp_name_map)

df_time_summary = df_train_all.groupby("timestamp").tail(1).copy()
df_time_summary["name"] = df_time_summary["timestamp"].map(timestamp_name_map)

name_order = [timestamp_name_map[ts] for ts in timestamps_all]
base_palette = sns.color_palette("tab10", n_colors=len(name_order))
palette_dict = dict(zip(name_order, base_palette))

plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_time_summary,
    y="name",
    x="total_time_min",
    palette="rocket"
)
plt.title("Total training time (min) per activation")
plt.xlabel("Total training time (min)")
plt.ylabel("Activation")
plt.grid(True, axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

max_step = df_val_all["step"].max() if not df_val_all.empty else 0
bin_stride = bin_width

step_bins = [
    (start, start + bin_width)
    for start in range(100, max_step + 1, bin_stride)
]

for (start, end) in step_bins:
    df_segment = df_val_all[(df_val_all["step"] >= start) & (df_val_all["step"] <= end)]
    if df_segment.empty:
        continue

    plot_start = max(start, val_interval)
    plot_end = end + val_interval

    plt.figure(figsize=(14, 6))
    ax = sns.lineplot(
        data=df_segment,
        x="step",
        y="val_loss",
        hue="name",
        hue_order=name_order,
        palette=palette_dict,
        linewidth=1
    )
    ax.axhline(y=3.28, color="red", linestyle="--", label="Target val loss = 3.28")
    ax.set_xticks(range(plot_start, plot_end, val_interval))
    ax.set_xlim(plot_start, plot_end - val_interval)
    ax.set_title(f"Val loss per activation, steps {start}-{end}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Val loss")
    ax.legend(fontsize="x-small")
    plt.tight_layout()
    plt.show()