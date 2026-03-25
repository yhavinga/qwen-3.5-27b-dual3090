#!/usr/bin/env python3
"""Generate plots for Qwen 3.5-27B Twitter thread."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Colors (Plotly-inspired)
COLORS = {
    'qwen': '#00CC96',      # Green
    'gemma': '#636EFA',     # Blue
    'fp16': '#00CC96',      # Green
    'int8': '#EF553B',      # Red
    'degradation': '#EF553B',  # Red
}


def plot_qwen_vs_gemma():
    """Bar chart: Qwen FP16 vs Gemma INT8 at different context lengths."""
    contexts = ['8K', '32K', '64K', '128K']
    qwen_fp16 = [51, 45, 39, 40]
    gemma_int8 = [39, 35, 25, 21]

    x = np.arange(len(contexts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, qwen_fp16, width, label='Qwen 3.5 (FP16 KV)',
                   color=COLORS['qwen'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, gemma_int8, width, label='Gemma 3 (INT8 KV)',
                   color=COLORS['gemma'], edgecolor='white', linewidth=1)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')

    # Highlight crossover
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.6, 55, 'Qwen faster →', fontsize=10, color='gray', style='italic')

    ax.set_ylabel('Decode Speed (tok/s)')
    ax.set_xlabel('Context Length')
    ax.set_title('Qwen 3.5 (FP16) vs Gemma 3 (INT8) — Both 27B on 2x RTX 3090')
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 65)

    # Add degradation percentages
    qwen_deg = (qwen_fp16[0] - qwen_fp16[-1]) / qwen_fp16[0] * 100
    gemma_deg = (gemma_int8[0] - gemma_int8[-1]) / gemma_int8[0] * 100
    ax.text(0.02, 0.02, f'Qwen degradation 8K→128K: {qwen_deg:.0f}%\nGemma degradation 8K→128K: {gemma_deg:.0f}%',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'qwen_vs_gemma.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'qwen_vs_gemma.png'}")


def plot_qwen_int8_vs_fp16():
    """Bar chart: INT8 vs FP16 on Qwen showing INT8 is slower."""
    contexts = ['8K', '64K', '128K']
    fp16 = [51, 39, 40]
    int8 = [50, 33, 26]

    x = np.arange(len(contexts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width/2, fp16, width, label='FP16 KV Cache',
                   color=COLORS['fp16'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, int8, width, label='INT8 KV Cache',
                   color=COLORS['int8'], edgecolor='white', linewidth=1)

    # Add value labels with change percentage
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1,
                f'{int(bar1.get_height())}', ha='center', va='bottom', fontweight='bold')
        change = (int8[i] - fp16[i]) / fp16[i] * 100
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1,
                f'{int(bar2.get_height())}\n({change:+.0f}%)', ha='center', va='bottom',
                fontweight='bold', color=COLORS['int8'], fontsize=9)

    ax.set_ylabel('Decode Speed (tok/s)')
    ax.set_xlabel('Context Length')
    ax.set_title('INT8 KV Cache Makes Qwen 3.5-27B SLOWER\n(Hybrid model: only 25% attention layers)')
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 60)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'qwen_int8_vs_fp16.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'qwen_int8_vs_fp16.png'}")


def plot_context_scaling():
    """Line chart showing how both models scale with context length."""
    contexts = [8, 32, 64, 128]
    qwen = [51, 45, 39, 40]
    gemma = [39, 35, 25, 21]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(contexts, qwen, 'o-', color=COLORS['qwen'], linewidth=2.5,
            markersize=10, label='Qwen 3.5 (FP16) — Hybrid')
    ax.plot(contexts, gemma, 's-', color=COLORS['gemma'], linewidth=2.5,
            markersize=10, label='Gemma 3 (INT8) — Pure Transformer')

    # Fill the area where Qwen is faster
    ax.fill_between(contexts, qwen, gemma, where=[q > g for q, g in zip(qwen, gemma)],
                    alpha=0.2, color=COLORS['qwen'], label='Qwen advantage')

    # Annotations
    ax.annotate('Hybrid architecture\nmaintains speed', xy=(128, 40), xytext=(100, 50),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.annotate('Pure transformer\ndegrades -46%', xy=(128, 21), xytext=(100, 12),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.set_xlabel('Context Length (K tokens)')
    ax.set_ylabel('Decode Speed (tok/s)')
    ax.set_title('Long Context Scaling: Hybrid vs Pure Transformer')
    ax.set_xticks(contexts)
    ax.set_xticklabels([f'{c}K' for c in contexts])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 60)
    ax.set_xlim(0, 140)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'context_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'context_scaling.png'}")


def plot_architecture_comparison():
    """Pie/bar showing hybrid vs pure transformer architecture."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Qwen architecture
    ax1 = axes[0]
    sizes = [25, 75]
    labels = ['Full Attention\n(16 layers)', 'GDN Linear\n(48 layers)']
    colors = [COLORS['gemma'], '#CCCCCC']
    explode = (0.05, 0)
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.0f%%',
            shadow=False, startangle=90, textprops={'fontsize': 11})
    ax1.set_title('Qwen 3.5-27B\n(Hybrid Architecture)', fontsize=13, fontweight='bold')

    # Right: Gemma architecture
    ax2 = axes[1]
    sizes = [100]
    labels = ['Full Attention\n(62 layers)']
    colors = [COLORS['gemma']]
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
            shadow=False, startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Gemma 3-27B\n(Pure Transformer)', fontsize=13, fontweight='bold')

    # Add KV cache impact text
    fig.text(0.25, 0.02, 'INT8 KV cache affects\nonly 25% of layers', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.text(0.75, 0.02, 'INT8 KV cache affects\n100% of layers', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'architecture_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'architecture_comparison.png'}")


def plot_hero_summary():
    """Hero image summarizing key findings."""
    fig = plt.figure(figsize=(14, 8))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Top left: Qwen vs Gemma bars
    ax1 = fig.add_subplot(gs[0, 0])
    contexts = ['8K', '32K', '64K', '128K']
    qwen = [51, 45, 39, 40]
    gemma = [39, 35, 25, 21]
    x = np.arange(len(contexts))
    width = 0.35
    ax1.bar(x - width/2, qwen, width, label='Qwen 3.5 (FP16)', color=COLORS['qwen'])
    ax1.bar(x + width/2, gemma, width, label='Gemma 3 (INT8)', color=COLORS['gemma'])
    ax1.set_ylabel('tok/s')
    ax1.set_title('Decode Speed by Context Length')
    ax1.set_xticks(x)
    ax1.set_xticklabels(contexts)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 60)

    # Top right: Architecture pie
    ax2 = fig.add_subplot(gs[0, 1])
    sizes = [25, 75]
    labels = ['Attention (16)', 'GDN (48)']
    colors = [COLORS['gemma'], '#DDDDDD']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
    ax2.set_title('Qwen 3.5 Layer Types')

    # Bottom left: INT8 penalty on Qwen
    ax3 = fig.add_subplot(gs[1, 0])
    contexts = ['8K', '64K', '128K']
    fp16 = [51, 39, 40]
    int8 = [50, 33, 26]
    x = np.arange(len(contexts))
    ax3.bar(x - width/2, fp16, width, label='FP16', color=COLORS['fp16'])
    ax3.bar(x + width/2, int8, width, label='INT8', color=COLORS['int8'])
    for i in range(len(contexts)):
        change = (int8[i] - fp16[i]) / fp16[i] * 100
        ax3.text(x[i] + width/2, int8[i] + 1, f'{change:+.0f}%', ha='center',
                fontsize=9, color=COLORS['int8'], fontweight='bold')
    ax3.set_ylabel('tok/s')
    ax3.set_title('INT8 Makes Qwen SLOWER')
    ax3.set_xticks(x)
    ax3.set_xticklabels(contexts)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim(0, 60)

    # Bottom right: Key takeaways
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    takeaways = [
        "> Qwen 3.5 beats Gemma 3 above 8K context",
        "> Hybrid architecture scales better (-22% vs -46%)",
        "> INT8 KV cache hurts hybrid models",
        "> Use FP16 KV for Qwen, INT8 for Gemma",
        "> 2x RTX 3090 matches RTX 5090 at 32K+",
    ]
    for i, text in enumerate(takeaways):
        ax4.text(0.1, 0.85 - i*0.18, text, fontsize=12, transform=ax4.transAxes,
                verticalalignment='top')
    ax4.set_title('Key Findings', fontsize=13, fontweight='bold')

    fig.suptitle('Qwen 3.5-27B on Dual RTX 3090: INT8 KV Cache Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(OUTPUT_DIR / 'hero_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'hero_summary.png'}")


def plot_rtx_comparison():
    """Compare 2x RTX 3090 vs RTX 5090."""
    contexts = ['8K', '32K', '128K']
    dual_3090 = [51, 45, 40]
    single_5090 = [61, 44, 35]

    x = np.arange(len(contexts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width/2, dual_3090, width, label='2x RTX 3090 (vLLM TP=2)',
                   color='#FF6B6B', edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, single_5090, width, label='1x RTX 5090',
                   color='#4ECDC4', edgecolor='white', linewidth=1)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')

    # Highlight where 3090 wins
    ax.annotate('3090s win!', xy=(2, 40), xytext=(2.3, 45),
                fontsize=11, fontweight='bold', color='#FF6B6B',
                arrowprops=dict(arrowstyle='->', color='#FF6B6B'))

    ax.set_ylabel('Decode Speed (tok/s)')
    ax.set_xlabel('Context Length')
    ax.set_title('Old Hardware Beats New: 2x RTX 3090 vs RTX 5090\n(Qwen 3.5-27B)')
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 70)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rtx_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'rtx_comparison.png'}")


if __name__ == '__main__':
    print("Generating plots for Qwen 3.5 Twitter thread...\n")

    plot_qwen_vs_gemma()
    plot_qwen_int8_vs_fp16()
    plot_context_scaling()
    plot_architecture_comparison()
    plot_hero_summary()
    plot_rtx_comparison()

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
