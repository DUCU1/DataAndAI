import pandas as pd
import math
import statistics as stat

# Missing values for NAs
missing_values = ['n/a', 'na', 'nan', 'N/A', 'NA', 'NaN', 'NAN', '--', 'Missing']


def nb_of_fields(name_csv_file, delim):
    import csv
    fields=[]
    with open(name_csv_file, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=delim)
        fields = next(data)
        min_counter = len(fields)
    max_counter = min_counter
    for row in data:
        nb_fields = len (row)
        if nb_fields > max_counter:
            max_counter = nb_fields
        elif nb_fields < min_counter: min_counter = nb_fields
    return([min_counter, max_counter])


# Nu merge pe tabele care au frequences
def all_freq(x):
    t_abs = x.value_counts(dropna=False).sort_index()
    t_rel = (x.value_counts(dropna=False, normalize=True).sort_index() * 100).round(1)
    t_abs_cum = x.value_counts(dropna=False).sort_index().cumsum()
    t_rel_cum = (x.value_counts(dropna=False, normalize=True).sort_index().cumsum() * 100).round(1)
    return pd.DataFrame({'abs freq': t_abs, 'rel freq': t_rel, 'abs cum freq': t_abs_cum, 'relcum freq': t_rel_cum})


## Comulatitve percentage
x = [4, 5, 6, 7, 8, 9, 10]
freq = [2, 3, 5, 6, 4, 2, 3]
df = pd.DataFrame({'freq': freq}, columns=['freq'], index=x)
# <---- Here ---->
cum_perc = df.cumsum() / df.sum() * 100

## Median pentru categorical data ( Procesor name de exmeplu ) da tre facute categorie si sortate
# laptops = [0, 1, 2, 3, 4, 5, 6]
import math as m
# cpuGenerationLevels = ['Sandy Bridge', 'Ivy Bridge', 'Haswell', 'Broadwell', 'Skylake', 'Kabylake']
# laptops.cpuGeneration = pd.Categorical(laptops.cpuGeneration, ordered=True, categories=cpuGenerationLevels)
def median_categorical(data):
    d = data.dropna()
    n = len(d)
    middle = m.floor(n / 2)
    return d.sort_values().reset_index(drop=True)[middle]


# Weighted mean
def weighted_mean(credits, score):
    return sum(credits * score) / sum(credits)


# Geometric mean
def geo_mean(delta):
    import numpy as np
    return np.exp(np.mean(np.log(delta)))


# Harmonic mean
def harmonic_mean(data):
    from scipy import stats
    return stats.hmean(data)


## Functie care face toate mediile
def central(cacat):
    data = cacat.dropna()
    mode = data.mode()
    med = data.median()
    mean = data.mean()
    geo = geo_mean(data)
    harm = harmonic_mean(data)
    return pd.DataFrame(
        {'Mode': mode, 'Median': med, 'Mean': mean.round(2), 'G-mean': geo.round(2), 'H-mean': harm.round(2)})


## Remove outliers funtion
def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    I = Q3 - Q1
    low = Q1 - 1.5 * I
    high = Q3 + 1.5 * I
    return [data[~data.between(low, high)]]

## Get Outliers
def outlier_boundaries(x):
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    I = Q3 - Q1
    return [Q1-1.5*I, Q3+1.5*I]

## Cate clase ne trebe
#data = [1,2,3]
# n=len(data)
# math.ceil(1+math.log2(n)) # Sturges
# b=3.5*stat.stdev(data)/(n**(1/3))
# math.ceil((data.max()-data.min())/b) # Scott
# math.ceil(math.sqrt(n))

#Standard deviation
#data.std()
# sau
def standard_dev(data):
 n = len(data)
 differences = (data - data.mean())**2
 return m.sqrt(differences.mean()*n/(n-1))

#Statistical dispersion
def dispersion(series):
    s = series.dropna()
    import numpy as np
    if type(s[0]) == np.int32 or type(s[0]) == np.float32 or type(s[0]) == np.float64 or type(s[0]) == np.int64:
        from scipy import stats
        d_range = s.max() - s.min()
        d_IQR = s.quantile(0.75) - s.quantile(0.25)
        d_mad = s.mad()
        d_var = s.var()
        d_std = s.std()
        print(" Range IQR MAD Var std")
        print('%5.2f' % d_range + ' ' + '%5.2f' % d_IQR + ' '+'%5.2f' % d_mad + ' ' + '%5.2f' % d_var + ' ' + '%5.2f' % d_std)
    else:
        print('Range = ' + s.min() + ' ' + s.max())