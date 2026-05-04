import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()

def run_monte_carlo(num_sims: int, init_port_val: float, ann_withdraw: float, inf_rate: float, mkt_ret_mean: float, mkt_ret_std: float, num_yrs: int) -> np.ndarray:
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

def plot_mc_results(port_matrix: np.ndarray):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
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

    final_vals = port_matrix[-1, :]
    ax2.hist(final_vals, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax2.set_title(f"Distribution of Final Values (Year {num_yrs})")
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

def calc_risk_metrics(port_matrix: np.ndarray, target_yr: int):

    target_yr_vals = port_matrix[target_yr, :]
    df_vals = pd.Series(target_yr_vals)

    num_ruined = (df_vals == 0).sum()
    total_sims = len(df_vals)
    prob_ruin = (num_ruined / total_sims) * 100

    worst_5_pct = df_vals.quantile(0.05)
    median_val = df_vals.quantile(0.50)
    best_5_pct = df_vals.quantile(0.95)
    best_outcome = df_vals.max()

    report = (
        f"[bold cyan]Total Simulations:[/bold cyan] {total_sims:,}\n"
        f"[bold red]Probability of Ruin:[/bold red] {prob_ruin:.2f}% ({num_ruined} paths failed)\n"
        f"----------------------------------------\n"
        f"[bold]Worst 5% Case:[/bold]  ${worst_5_pct:,.2f}\n"
        f"[bold]Median Outcome:[/bold] ${median_val:,.2f}\n"
        f"[bold]Top 5% Case:[/bold]    ${best_5_pct:,.2f}"


        
        f"\n[bold]Best Outcome:[/bold]   ${best_outcome:,.2f}"
    )
    console.print(Panel(report, title=f"Risk Analytics Report (Year {target_yr})", expand=False))


def get_valid_input(prompt_msg: str, default_val: float, is_int: bool = False) -> float:
    
    while True:
        try:
            user_input = Prompt.ask(prompt_msg, default=str(default_val))
            
            if is_int:
                val = int(user_input)
            else:
                val = float(user_input)
            
            if val < 0:
                console.print("[bold red]Error:[/bold red] Value cannot be negative. Try again.")
                continue
                
            return val
            
        except ValueError:
            console.print("[bold red]Error:[/bold red] Invalid input. Please enter numbers only (no commas or symbols).")

def main_cli():
    console.clear()
    console.print(Panel.fit("[bold green]Monte Carlo Retirement Simulator[/bold green]\nPress Enter to accept defaults, or type your own values."))
    
    while True:

        console.print("\n[bold]Step 1: Set Parameters[/bold]")
        init_port = get_valid_input("Initial Portfolio Value ($)", 1000000)
        ann_draw = get_valid_input("Annual Withdrawal ($)", 40000)
        inf_r = get_valid_input("Inflation Rate (Decimal, e.g., 0.025)", 0.025)
        mkt_mean = get_valid_input("Market Return Mean (Decimal, e.g., 0.07)", 0.07)
        mkt_std = get_valid_input("Market Return Volatility (Decimal, e.g., 0.15)", 0.15)
        yrs = int(get_valid_input("Years to Simulate", 30, is_int=True))
        sims = int(get_valid_input("Number of Simulations", 10000, is_int=True))

        console.print("\n[bold]Step 2: Simulating...[/bold]")
        with console.status("[cyan]Calculating parallel universes..."):
            mc_results = run_monte_carlo(sims, init_port, ann_draw, inf_r, mkt_mean, mkt_std, yrs)
        
        console.print("[bold green]Simulation Complete![/bold green]\n")
        calc_risk_metrics(mc_results, yrs)
        plot_mc_results(mc_results)

        run_again = Prompt.ask("\nWould you like to run another simulation?", choices=["y", "n"], default="n")
        if run_again.lower() == 'n':
            console.print("[bold blue]Exiting simulator. Happy investing![/bold blue]")
            break
        else:
            console.clear()

if __name__ == "__main__":
    try:
        main_cli()
    except KeyboardInterrupt:
        console.print("\n[bold red]Simulator terminated by user.[/bold red]")