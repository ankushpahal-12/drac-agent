"""
Publication-Quality Figure & Chart Generator for DRAC Evaluation.
Generates:
1. fig1_recovery_success_ablation.png (RSR comparison)
2. fig2_pareto_frontier_cnre.png (CNRE vs Token Overhead Pareto frontier)
3. fig3_diagnostic_confusion_matrix.png (RCA breakdown across domains)
4. fig4_sre_metrics_comparison.png (CCF, MTTR_A, MTCR_A)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_all_plots(results_csv: str = "experiments/results/raw_trials.csv", summary_csv: str = "experiments/results/summary_metrics.csv", output_dir: str = "experiments/plots"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 16
    })

    if not os.path.exists(results_csv) or not os.path.exists(summary_csv):
        print("[!] Result files not found. Run benchmark runner first.")
        return

    raw_df = pd.read_csv(results_csv)
    summary_df = pd.read_csv(summary_csv)

    palette = ["#d9534f", "#f0ad4e", "#5bc0de", "#0275d8", "#5cb85c"]

    # ----------------------------------------------------
    # Figure 1: Recovery Success Rate (RSR) Ablation
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5.5), dpi=300)
    ax = sns.barplot(
        data=summary_df,
        x="Strategy",
        y="RSR (%)",
        hue="Strategy",
        legend=False,
        palette="viridis",
        edgecolor="black",
        linewidth=1.2
    )
    plt.title("Figure 1: Recovery Success Rate (RSR) Across 5-Way Ablation", pad=15, fontweight="bold")
    plt.ylabel("Recovery Success Rate (%)", fontweight="bold")
    plt.xlabel("")
    plt.ylim(0, 105)
    plt.xticks(rotation=15, ha="right")

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}%",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    xytext=(0, 4), textcoords='offset points')

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "fig1_recovery_success_ablation.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"[+] Saved: {fig1_path}")

    # ----------------------------------------------------
    # Figure 2: Pareto Frontier (CNRE vs Token Overhead Ratio)
    # ----------------------------------------------------
    plt.figure(figsize=(8.5, 6), dpi=300)
    for i, row in summary_df.iterrows():
        plt.scatter(
            row["TOR (%)"],
            row["CNRE"],
            s=220,
            label=row["Strategy"],
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5
        )
        # Text annotation
        plt.annotate(
            row["Strategy"],
            (row["TOR (%)"], row["CNRE"]),
            textcoords="offset points",
            xytext=(10, -5),
            fontsize=10,
            fontweight="bold"
        )

    plt.title("Figure 2: Pareto Efficiency (CNRE vs. Token Overhead Ratio)", pad=15, fontweight="bold")
    plt.xlabel("Token Overhead Ratio (TOR %) [Lower is Cheaper]", fontweight="bold")
    plt.ylabel("Cost-Normalized Recovery Efficiency (CNRE) [Higher is Better]", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "fig2_pareto_frontier_cnre.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"[+] Saved: {fig2_path}")

    # ----------------------------------------------------
    # Figure 3: Diagnostic Root Cause Accuracy (RCA)
    # ----------------------------------------------------
    drac_trials = raw_df[raw_df["strategy_name"] == "DRAC Full System"]
    plt.figure(figsize=(8, 5.5), dpi=300)
    rca_by_domain = drac_trials.groupby("ground_truth_domain")["diagnosis_correct"].mean() * 100.0
    rca_df = rca_by_domain.reset_index()
    rca_df.columns = ["Domain", "Accuracy"]

    ax = sns.barplot(
        data=rca_df,
        x="Domain",
        y="Accuracy",
        hue="Domain",
        legend=False,
        palette="crest",
        edgecolor="black",
        linewidth=1.2
    )
    plt.title("Figure 3: DRAC Root-Cause Diagnostic Accuracy (RCA) by Domain", pad=15, fontweight="bold")
    plt.ylabel("Diagnostic Accuracy (%)", fontweight="bold")
    plt.xlabel("Ground Truth Fault Domain", fontweight="bold")
    plt.ylim(0, 110)
    plt.axhline(53.0, color="crimson", linestyle="--", linewidth=2, label="AgentChaos Baseline (<53%)")
    plt.legend(loc="lower right", frameon=True)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}%",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    xytext=(0, 4), textcoords='offset points')

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_diagnostic_confusion_matrix.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"[+] Saved: {fig3_path}")

    # ----------------------------------------------------
    # Figure 4: SRE Metrics (Cascade Containment Factor CCF)
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    ax = sns.barplot(
        data=summary_df,
        x="Strategy",
        y="CCF (%)",
        hue="Strategy",
        legend=False,
        palette="magma",
        edgecolor="black",
        linewidth=1.2
    )
    plt.title("Figure 4: Cascade Containment Factor (CCF) Across Recovery Paradigms", pad=15, fontweight="bold")
    plt.ylabel("Cascade Containment Factor (%)", fontweight="bold")
    plt.xlabel("")
    plt.ylim(0, 110)
    plt.xticks(rotation=15, ha="right")

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}%",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    xytext=(0, 4), textcoords='offset points')

    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "fig4_sre_metrics_comparison.png")
    plt.savefig(fig4_path)
    plt.close()
    print(f"[+] Saved: {fig4_path}")

    print("[*] All publication figures successfully generated!")

if __name__ == "__main__":
    generate_all_plots()
