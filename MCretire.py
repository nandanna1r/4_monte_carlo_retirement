import numpy as np

def calc_single_path(init_port_val: float, ann_withdraw: float, inf_rate: float, mkt_ret_mean: float, mkt_ret_std: float, num_yrs: int) -> list:
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

'''
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
    '''