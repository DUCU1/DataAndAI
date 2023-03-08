import pandas as pd


def all_freq(x):
    t_abs = x.value_counts(dropna=False).sort_index()
    t_rel = (x.value_counts(dropna=False, normalize=True).sort_index()*100).round(1)
    t_abs_cum = x.value_counts(dropna=False).sort_index().cumsum()
    t_rel_cum = (x.value_counts(dropna=False, normalize=True).sort_index().cumsum()*100).round(1)
    return pd.DataFrame({'abs freq':t_abs,'rel freq':t_rel,'abs cum freq':t_abs_cum,'relcum freq':t_rel_cum})


