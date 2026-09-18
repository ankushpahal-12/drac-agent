"""
DRAC Theoretical Diagrams Generator.
Generates 300 DPI publication figures for:
1. Figure 5: The Recovery Trilemma & Attention Contamination Space
2. Figure 6: The B-POMDP State Transition & Budget Frontier
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

os.makedirs("experiments/plots", exist_ok=True)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 1.2

def generate_trilemma_figure(output_path="experiments/plots/fig5_recovery_trilemma_formalization.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    fig.patch.set_facecolor("#f8f9fa")
    
    # ----------------------------------------------------
    # Subplot 1: Geometric Trilemma Space
    # ----------------------------------------------------
    ax1.set_facecolor("#ffffff")
    ax1.set_title("A. The Recovery Trilemma & DRAC Solution", fontsize=13, fontweight="bold", pad=15, color="#1a252f")
    
    # Coordinates for triangle
    A = np.array([0.15, 0.82])  # Context Contamination
    B = np.array([0.85, 0.82])  # Rollback Amnesia
    C = np.array([0.50, 0.15])  # In-band Reflection
    center = np.array([0.50, 0.58]) # DRAC DNCS
    
    # Triangle boundary
    tri = patches.Polygon([A, B, C], closed=True, fill=True, facecolor="#edf2f7", edgecolor="#cbd5e0", linewidth=2.5, linestyle="--")
    ax1.add_patch(tri)
    
    # Corners
    ax1.scatter([A[0], B[0], C[0]], [A[1], B[1], C[1]], s=250, c=["#e53e3e", "#dd6b20", "#805ad5"], zorder=5)
    
    # Corner text
    ax1.text(A[0]-0.02, A[1]+0.06, "Context Contamination\n(Naive Retry)\nRSR: 36.2% | Primed Errors", 
             ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c53030")
    ax1.text(B[0]+0.02, B[1]+0.06, "Rollback Amnesia\n(Pure Rollback)\nRSR: 50.0% | Greedy Recurrence", 
             ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c05621")
    ax1.text(C[0], C[1]-0.08, "Monolithic In-Band Reflection\n(Reflexion)\nRSR: 73.8% | TOR: 107.5% | CNRE: 0.053", 
             ha="center", va="top", fontsize=10, fontweight="bold", color="#6b46c1")
    
    # Center DRAC Solution
    ax1.scatter([center[0]], [center[1]], s=600, c="#319795", zorder=6, edgecolors="#234e52", linewidth=2.5)
    ax1.text(center[0], center[1], "DRAC\nDNCS", ha="center", va="center", fontsize=11, fontweight="bold", color="#ffffff")
    ax1.text(center[0], center[1]-0.15, "Pareto Solution (DNCS):\n* RSR: 100.0%\n* TOR: 35.6%\n* CNRE: 20.00\n* Zero Contamination & Zero Amnesia", 
             ha="center", va="top", fontsize=9.5, fontweight="bold", color="#234e52",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6fffa", edgecolor="#319795", linewidth=1.5))
    
    # Arrows from corners to center
    for pt, col in [(A, "#e53e3e"), (B, "#dd6b20"), (C, "#805ad5")]:
        ax1.annotate("", xy=(center[0], center[1]), xytext=(pt[0], pt[1]),
                     arrowprops=dict(arrowstyle="->", color=col, lw=2, ls=":", mutation_scale=15))
        
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axis("off")
    
    # ----------------------------------------------------
    # Subplot 2: Attention Probability Distribution (Theorem 1)
    # ----------------------------------------------------
    ax2.set_facecolor("#ffffff")
    ax2.set_title("B. Attention Weight Priming: P(Valid Action | Context)", fontsize=13, fontweight="bold", pad=15, color="#1a252f")
    
    categories = ["Valid Next Action\na_valid", "Error Rationalization\na_defensive", "Repeat Flawed Action\na_fail", "Hallucinated Syntax\na_syntax"]
    p_clean = [0.92, 0.03, 0.03, 0.02]
    p_polluted = [0.34, 0.38, 0.20, 0.08]
    p_dncs = [0.98, 0.01, 0.00, 0.01]
    
    x = np.arange(len(categories))
    width = 0.26
    
    rects1 = ax2.bar(x - width, p_clean, width, label="Clean State (S_k)", color="#4299e1", edgecolor="#2b6cb0")
    rects2 = ax2.bar(x, p_polluted, width, label="Polluted Context (X_fail)", color="#fc8181", edgecolor="#e53e3e")
    rects3 = ax2.bar(x + width, p_dncs, width, label="DRAC (S_k + c_dncs)", color="#38b2ac", edgecolor="#234e52")
    
    ax2.set_ylabel("Conditional Sampling Probability", fontsize=11, fontweight="bold", color="#2d3748")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9.5, fontweight="bold", color="#2d3748")
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#edf2f7", edgecolor="#cbd5e0", fontsize=10)
    
    # Add annotation for Theorem 1
    ax2.annotate("Theorem 1: Negative Priming\nP(a_valid) drops from 92% to 34%", 
                 xy=(0 - width/2, 0.35), xytext=(0.4, 0.65),
                 arrowprops=dict(facecolor="#c53030", arrowstyle="->", lw=1.5),
                 fontsize=9.5, fontweight="bold", color="#c53030",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f5", edgecolor="#feb2b2"))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {output_path}")

def generate_bpomdp_figure(output_path="experiments/plots/fig6_bpomdp_belief_state_space.png"):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    ax.set_title("DRAC Budget-Constrained POMDP Utility Surface: U(a) = \u03b3 P(Succ) - \u03bb_c (Cost/C_rem) - \u03bb_t (Lat/T_rem)",
                 fontsize=12, fontweight="bold", pad=15, color="#1a252f")
    
    c_rem = np.linspace(50, 2000, 100)
    
    # Utility curves for different actions
    # ROLLBACK_WITH_DNCS: Cost=110, Lat=0.01, P=0.94
    u_dncs = (0.94 * 1.0) - (0.25 * (110.0 / c_rem)) - 0.02
    
    # RETRY: Cost=120, Lat=0.01, P=0.36
    u_retry = (0.36 * 1.0) - (0.25 * (120.0 / c_rem)) - 0.02
    
    # REFLEXION: Cost=430, Lat=0.80, P=0.74
    u_refl = (0.74 * 1.0) - (0.25 * (430.0 / c_rem)) - (0.15 * (0.80 / 10.0))
    
    # ALTERNATE_MODEL: Cost=350, Lat=2.0, P=0.95
    u_alt = (0.95 * 1.0) - (0.25 * (350.0 / c_rem)) - (0.15 * (2.0 / 10.0))
    
    ax.plot(c_rem, u_dncs, label="Action: ROLLBACK_WITH_DNCS (DRAC)", color="#319795", lw=3.0)
    ax.plot(c_rem, u_refl, label="Action: REFLEXION (In-band verbal)", color="#805ad5", lw=2.0, linestyle="--")
    ax.plot(c_rem, u_alt, label="Action: ALTERNATE_MODEL", color="#d69e2e", lw=2.0, linestyle="-.")
    ax.plot(c_rem, u_retry, label="Action: RETRY (Naive retry)", color="#e53e3e", lw=2.0, linestyle=":")
    
    # Failsafe escalation boundary
    ax.axvline(x=100, color="#e53e3e", lw=2, linestyle="-", label="Failsafe Threshold (C_rem < 100 -> HUMAN_ESCALATION)")
    ax.fill_betweenx([-0.8, 1.1], 0, 100, color="#fed7d7", alpha=0.4)
    ax.text(50, 0.4, "HUMAN\nESCALATION\nZONE", ha="center", va="center", fontsize=9, fontweight="bold", color="#9b2c2c", rotation=90)
    
    ax.set_xlabel("Remaining Token Budget C_rem (tokens)", fontsize=11, fontweight="bold", color="#2d3748")
    ax.set_ylabel("Expected B-POMDP Policy Utility U(a)", fontsize=11, fontweight="bold", color="#2d3748")
    ax.set_xlim(0, 2000)
    ax.set_ylim(-0.4, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", facecolor="#edf2f7", edgecolor="#cbd5e0", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {output_path}")

if __name__ == "__main__":
    generate_trilemma_figure()
    generate_bpomdp_figure()
