import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def calc_single_path(init_port_val: float, ann_withdraw: float, 
                    inf_rate: float, mkt_ret_mean: float,  
                    mkt_ret_std: float, num_yrs: int) -> list:
    # 1 portfolio over certain years, market returns, inflation, withdrawals, initial value
    
    port_hist = [init_port_val] 
    curr_val = init_port_val
    curr_withdraw = ann_withdraw

    for yr in range(num_yrs):

        mkt_ret = np.random.normal(loc=mkt_ret_mean, scale=mkt_ret_std)

        curr_val = (curr_val * (1 + mkt_ret)) - curr_withdraw

        if curr_val < 0:
            curr_val = 0

        port_hist.append(curr_val)

        curr_withdraw = curr_withdraw * (1 + inf_rate)

    return port_hist
"""
# EX calc_single_path: the inputs are given and the outcomes are printed.
# we can see that the portfolio value fluctuates for each test, showing volatility 
if __name__ == "__main__":
    
    single_sim = calc_single_path(
        init_port_val=1000000,
        ann_withdraw=40000,
        inf_rate=0.025,
        mkt_ret_mean=0.07,
        mkt_ret_std=0.15,
        num_yrs=30
    )

    print("Year 0 to 30 Portfolio Values:", [round(val, 2) for val in single_sim[:30]])

    print("Final Value (Year 30):", round(single_sim[-1], 2))
"""

def run_monte_carlo(num_sims: int, init_port_val: float, 
                    ann_withdraw: float, inf_rate: float, 
                    mkt_ret_mean: float, mkt_ret_std: float, 
                    num_yrs: int) -> np.ndarray:
  # 2 vectorized operation for monte carlo simulation of 10,000 paths 
    port_matrix = np.zeros((num_yrs + 1, num_sims))
    
    port_matrix[0, :] = init_port_val
    
    ret_matrix = np.random.normal(loc=mkt_ret_mean, scale=mkt_ret_std, size=(num_yrs, num_sims))
    
    curr_withdraw = ann_withdraw
    
    for yr in range(num_yrs):

        prev_vals = port_matrix[yr, :]
        
        mkt_rets_this_yr = ret_matrix[yr, :]
        
        new_vals = (prev_vals * (1 + mkt_rets_this_yr)) - curr_withdraw

        new_vals = np.maximum(new_vals, 0)
        
        port_matrix[yr + 1, :] = new_vals
        
        curr_withdraw = curr_withdraw * (1 + inf_rate)
        
    return port_matrix

'''
if __name__ == "__main__":
    # test for 10000 simulations, peek at first 5, check runtime
    print("Loading 10000 timelines...")
    
    start_time = time.time()
    
    mc_results = run_monte_carlo(
        num_sims=10000,
        init_port_val=1000000,
        ann_withdraw=40000,
        inf_rate=0.025,
        mkt_ret_mean=0.07,
        mkt_ret_std=0.15,
        num_yrs=30
    )
    
    end_time = time.time()
    
    print(f"Simulation Matrix Shape: {mc_results.shape}")
    print(f"Execution Time: {round(end_time - start_time, 4)} seconds")
    formatted_vals = [f"${val:,.2f}" for val in mc_results[-1, :5]]
    print(f"Final Values of Sims 1-5: {formatted_vals}")
    '''


def plot_mc_results(port_matrix: np.ndarray):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    #1: Spaghetti Plot 
    num_yrs = port_matrix.shape[0] - 1
    yrs_arr = np.arange(num_yrs + 1)
    
    ax1.plot(yrs_arr, port_matrix[:, :500], color='blue', alpha=0.05)
    ax1.set_title("First 500 Simulated Portfolio Paths")
    ax1.set_xlabel("Years in Retirement")
    ax1.set_ylabel("Portfolio Value ($)")
    
    ax1.set_xlim(0, num_yrs) 
    ax1.set_ylim(bottom=0) 
    
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, loc: f"${x/1e6:,.1f}M"))
    ax1.grid(True, alpha=0.3)

    # 2: Distribution Histogram of Final Values
    final_vals = port_matrix[-1, :]
    
    ax2.hist(final_vals, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax2.set_title("Distribution of Final Values (Year 30)")
    ax2.set_xlabel("Final Portfolio Value ($)")
    ax2.set_ylabel("Frequency (Number of Sims)")
    
    ax2.set_xlim(left=0) 
    
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, loc: f"{int(x):,}"))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, loc: f"${x/1e6:,.1f}M"))
    
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label="Bankruptcy ($0)")

    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0) 
    plt.show()

import pandas as pd

def calc_risk_metrics(port_matrix: np.ndarray, target_yr: int = 30):
    # 1. Extract the specific row for our target year
    # If the user asks for Year 30, we pull index 30. 
    target_yr_vals = port_matrix[target_yr, :]

    # 2. Convert the raw NumPy array into a Pandas Series for easy analysis
    df_vals = pd.Series(target_yr_vals)

    # 3. Calculate Probability of Ruin
    # df_vals == 0 creates a True/False list. sum() counts the Trues.
    num_ruined = (df_vals == 0).sum()
    total_sims = len(df_vals)
    prob_ruin = (num_ruined / total_sims) * 100

    # 4. Quantile Analysis (Best, Median, and Worst case scenarios)
    # 0.05 means only 5% of outcomes were worse than this number
    worst_5_pct = df_vals.quantile(0.05)
    median_val = df_vals.quantile(0.50)
    best_5_pct = df_vals.quantile(0.95)

    # 5. Print the formatted Risk Report
    print(f"\n--- Risk Analytics Report (Year {target_yr}) ---")
    print(f"Total Simulations: {total_sims:,}")
    print(f"Probability of Ruin: {prob_ruin:.2f}% ({num_ruined} paths failed)")
    print("-" * 40)
    print(f"Worst 5% Case:  ${worst_5_pct:,.2f}")
    print(f"Median Outcome: ${median_val:,.2f}")
    print(f"Top 5% Case:    ${best_5_pct:,.2f}")
    print("-" * 40)

if __name__ == "__main__":
    import time
    print("Spinning up 10,000 parallel universes...")
    
    start_time = time.time()
    
    mc_results = run_monte_carlo(
        num_sims=10000,
        init_port_val=1000000,
        ann_withdraw=40000,
        inf_rate=0.025,
        mkt_ret_mean=0.07,
        mkt_ret_std=0.15,
        num_yrs=30
    )
    
    end_time = time.time()
    print(f"Simulation completed in {round(end_time - start_time, 4)} seconds.")
    
    print("Generating charts...")
    plot_mc_results(mc_results)
    
    print("Calculating risk metrics...")
    calc_risk_metrics(mc_results, target_yr=30)
    
    print("Done")