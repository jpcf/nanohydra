import pandas as pd
import numpy  as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
import warnings
import os
import pickle

CSV_PATH = "./data/results_ours_k8_g8.csv"
UCR_DATA_SET_FACTS = "./data/ucr_props.bin"

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
    plt.savefig("./data/fig1_histogram_dev.png", dpi=200, bbox_inches='tight')

    # Write back results to another CSV. x
    csv_out = pd.DataFrame(csv['dataset'])
    csv_out['Hydra'] = csv['Hydra']
    csv_out['Hydra_Binomial'] = csv['Hydra_Binomial']
    csv_out['Differences'] = diffs

    # Print out the worse ones
    csv_out.sort_values(by='Differences', ascending=True, inplace=True)
    print(f"****** The 15 worse-performing datasets are:\n{csv_out['dataset'][0:15]}\n")
    
    # Print out the best ones
    csv_out.sort_values(by='Differences', ascending=False, inplace=True)
    print(f"****** The 15 best-performing datasets are:\n{csv_out['dataset'][0:15]}\n")

    # Append other relevant information
    csv_out['Input Length']     = np.nan
    csv_out['Training Samples'] = np.nan
    csv_out['Output Classes']   = np.nan
    csv_out['Outlen/Train']     = np.nan
    csv_out['Imbalance']        = np.nan

    if(not os.path.exists("./data/ucr_props.bin")):
        props = {}
        print("First execution. It might take some time...")

        for idx,row in csv_out.iterrows():
            X,Y = load_ucr_ds(row['dataset'], split="test", return_type="numpy2d")
            input_len  = X.shape[1]
            numex_len  = X.shape[0]
            output_cls = len(np.unique(Y))
            
            # Evaluate Histogram and Class Imbalance 
            Y = Y.astype(np.uint8)
            hist,_ = np.histogram(Y, bins=output_cls)
            argmax,histmax  = np.argmax(hist), np.max(hist)
            histmean = np.mean(sorted(hist)[:-1])

            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                csv_out['Input Length'][idx]     = input_len 
                csv_out['Training Samples'][idx] = numex_len
                csv_out['Output Classes'][idx]   = output_cls
                csv_out['Outlen/Train'][idx]     = numex_len/output_cls
                csv_out['Imbalance'][idx]        = histmax/histmean
            props[row['dataset']] = {'input_len': input_len, 'numex_len': numex_len, 'output_cls': output_cls, 'imbalance': histmax/histmean}

        with open(UCR_DATA_SET_FACTS, "wb") as f:
            pickle.dump(props, f)
    else:
        print("Using precomputed values...")
        with open(UCR_DATA_SET_FACTS, "rb") as f:
            props = pickle.load(f)
            for idx,row in csv_out.iterrows():
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore')
                    csv_out['Input Length'][idx]     = props[row['dataset']]['input_len']
                    csv_out['Training Samples'][idx] = props[row['dataset']]['numex_len']
                    csv_out['Output Classes'][idx]   = props[row['dataset']]['output_cls']
                    csv_out['Outlen/Train'][idx]     = props[row['dataset']]['output_cls']/props[row['dataset']]['numex_len']
                    csv_out['Imbalance'][idx]        = props[row['dataset']]['imbalance']

    print(csv_out)

    # Scatter plot of accuracy difference vs input length
    plt.figure(2)
    plt.title("nanoHydra Differences vs Input Length")
    plt.scatter(csv_out['Differences'], csv_out['Input Length'])
    plt.xlabel("Accuracy Deviation of nanoHydra")
    plt.ylabel("Input Sequence Length (in samples)")
    plt.grid(True)
    plt.savefig("./data/fig2_dev_vs_input_len.png", dpi=200,  bbox_inches='tight')
    
    # Scatter plot of accuracy difference vs training set size
    plt.figure(3)
    plt.title("nanoHydra Differences vs Training Samples")
    plt.scatter(csv_out['Differences'], csv_out['Training Samples'])
    plt.xlabel("Accuracy Deviation of nanoHydra")
    plt.ylabel("Training Set Size (in number of training sequence examples)")
    plt.grid(True)
    plt.savefig("./data/fig3_dev_vs_training_sz.png", dpi=200,  bbox_inches='tight')

    # Scatter plot of accuracy difference vs original accuracy
    plt.figure(4)
    plt.title("nanoHydra Differences vs Original Accuracy")
    plt.scatter(csv_out['Differences'], 100*csv_out['Hydra'])
    plt.xlabel("Accuracy Deviation of nanoHydra")
    plt.ylabel("Accuracy of original Hydra Model")
    plt.grid(True)
    plt.savefig("./data/fig4_dev_vs_orig_acc.png",  dpi=200, bbox_inches='tight')

    # Scatter plot of accuracy difference vs output length
    plt.figure(5)
    plt.title("nanoHydra Differences vs Output Length")
    plt.scatter(csv_out['Differences'], csv_out['Output Classes'])
    plt.xlabel("Accuracy Deviation of nanoHydra")
    plt.ylabel("# Output Classes")
    plt.grid(True)
    plt.savefig("./data/fig5_dev_vs_orig_acc.png",  dpi=200, bbox_inches='tight')

    # Scatter plot of accuracy difference vs output length / num training examples ratio
    plt.figure(6)
    plt.title("nanoHydra Differences vs Output Length")
    plt.scatter(csv_out['Differences'], csv_out['Outlen/Train'])
    plt.xlabel("Accuracy Deviation of nanoHydra")
    plt.ylabel("# Output Classes /# Training examples")
    plt.grid(True)
    plt.savefig("./data/fig6_dev_vs_outtrain_rat.png",  dpi=200, bbox_inches='tight')

    # Scatter plot of accuracy difference vs output length / num training examples ratio
    plt.figure(7)
    plt.title("nanoHydra Differences vs Output Length")
    plt.scatter(csv_out['Differences'], csv_out['Imbalance'])
    plt.xlabel("Accuracy Deviation of nanoHydra")
    plt.ylabel("Class Imbalance Ratio")
    plt.grid(True)
    plt.savefig("./data/fig7_imbalance.png",  dpi=200, bbox_inches='tight')
    # Save to CSV
    csv_out.to_csv("./data/results_vs_data_props.csv")

    plt.show()

    




