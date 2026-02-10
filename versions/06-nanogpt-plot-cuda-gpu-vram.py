import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

gpu_specs = {
    "B200":
    {
        "vram_mib": 183_359,
        "max_bs": 128
    },
    "H200 SXM":
    {
        "vram_mib": 143_771,
        "max_bs": 128
    },
    "RTX PRO 6000":
    {
        "vram_mib": 97_887,
        "max_bs": 64
    },
    "H100 NVL":
    {
        "vram_mib": 95_830,
        "max_bs": 128
    },
    "H100 SXM":
    {
        "vram_mib": 81_559,
        "max_bs": 64
    },
    "H100 PCIe":
    {
        "vram_mib": 81_559, 
        "max_bs": 64
    },
    "L40":
    {
        "vram_mib": 46_068,
        "max_bs": 32
    },
    "L40S":
    {
        "vram_mib": 46_068,
        "max_bs": 32
    },
    "RTX 6000 Ada":
    {
        "vram_mib": 49_140,
        "max_bs": 64
    },
    "RTX 5090":
    {
        "vram_mib": 32_607,
        "max_bs": 32
    },
    "L4":
    {
        "vram_mib": 23_034,
        "max_bs": 16
    },
    "RTX 4000 Ada":
    {
        "vram_mib": 20_475,
        "max_bs": 16
    },
    "RTX 2000 Ada":
    {
        "vram_mib": 16_380,
        "max_bs": 16
    },
    "A100 PCIe":
    {
        "vram_mib": 81_920,
        "max_bs": 64
    },
    "A100 SXM":
    {
        "vram_mib": 81_920,
        "max_bs": 64
    },
    "A40":
    {
        "vram_mib": 46_068,
        "max_bs": 32
    },
    "RTX A6000":
    {
        "vram_mib": 49_140,
        "max_bs": 64
    },
    "RTX A5000":
    {
        "vram_mib": 24_564,
        "max_bs": 16
    },
    "RTX A4500":
    {
        "vram_mib": 20_470,
        "max_bs": 16
    },
    "RTX A4000":
    {
        "vram_mib": 16_376,
        "max_bs": 16
    },
}

df = pd.DataFrame.from_dict(gpu_specs, orient="index").reset_index().rename(columns={"index": "name"})
df["vram_gib"] = df["vram_mib"] / 1024
df_sorted = df.sort_values("vram_mib", ascending=False)

fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

axs[0].barh(df_sorted["name"], df_sorted["vram_gib"], color="mediumpurple")
axs[0].set_xlabel("VRAM (GiB)")
axs[0].set_title("VRAM (GiB) by GPU model")
axs[0].grid(axis='x', linestyle='--', alpha=0.5)

axs[1].barh(df_sorted["name"], df_sorted["max_bs"], color="indianred")
axs[1].set_xlabel("Max mini-batch size (with grad accum to reach 500M batch size)")
axs[1].set_title("Max mini-batch size at 1024 sequence length by GPU model")
axs[1].grid(axis='x', linestyle='--', alpha=0.5)

axs[0].set_xlim(right=192)
axs[0].xaxis.set_major_locator(ticker.FixedLocator([16, 32, 64, 128, 192]))
axs[1].xaxis.set_major_locator(ticker.FixedLocator([16, 32, 64, 128]))

axs[1].set_yticks(df_sorted.index)
axs[1].set_yticklabels(df_sorted["name"])
axs[1].tick_params(axis='y', labelleft=True)

plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
