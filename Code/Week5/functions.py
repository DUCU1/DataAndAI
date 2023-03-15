import pandas as pd

# Nu merge pe tabele care au frequences
def all_freq(x):
    t_abs = x.value_counts(dropna=False).sort_index()
    t_rel = (x.value_counts(dropna=False, normalize=True).sort_index()*100).round(1)
    t_abs_cum = x.value_counts(dropna=False).sort_index().cumsum()
    t_rel_cum = (x.value_counts(dropna=False, normalize=True).sort_index().cumsum()*100).round(1)
    return pd.DataFrame({'abs freq':t_abs,'rel freq':t_rel,'abs cum freq':t_abs_cum,'relcum freq':t_rel_cum})


## Comulatitve percentage
x = [4,5,6,7,8,9,10]
freq = [2,3,5,6,4,2,3]
df = pd.DataFrame({'freq': freq }, columns = ['freq'], index=x)
# <---- Here ---->
cum_perc = df.cumsum()/df.sum()*100

## Media pentru categorical data ( Procesor name de exmeplu )
import math as m
def median_categorical(data):
    d = data.dropna()
    n = len(d)
    middle = m.floor(n/2)
    return d.sort_values().reset_index(drop=True)[middle]
