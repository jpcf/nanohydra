import pandas as pd
import numpy  as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

CSV_PATH = "./data/results_ours_k8_g8.csv"

if __name__ == "__main__":

    csv = pd.read_csv(CSV_PATH)

    # Positive values means Hydra is better
    diffs = 100*(csv['Hydra_Binomial'] - csv['Hydra'])

    # Count Wins/Draws/Losses. Within -0.5pp is considered draw
    wins   = (np.greater_equal(diffs, 1) == True).sum()
    draws  = (np.isclose(100*csv['Hydra_Binomial'], 100*csv['Hydra'], atol=0.5) == True).sum()
    losses = (np.less(diffs, -1) == True).sum()
    total = wins+draws+losses

    # WDL Info. Print and display on histogram graph
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
    textstr = f"W/D/L\n{100*wins/total:.02f}%/{100*draws/total:.02f}%/{100*losses/total:.02f}%"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    fig, ax = plt.subplots()
    plt.title("Hydra Binomial (k=8, g=8) vs Hydra Gaussian. Positive means Binomial > Gaussian")
    plt.hist(diffs)
    plt.xlabel("Percent point difference")
    plt.ylabel("# Datasets")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.show()