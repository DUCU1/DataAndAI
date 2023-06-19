# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

revenues = np.array([20, 100, 175, 13, 37, 136, 245, 26, 75, 155, 326, 48, 92, 202, 384, 82, 176, 282, 445, 181],
                    dtype=float)


# %% prediction is last value
def naive(y: np.array):
    if y.size > 0:
        return y[-1]
    return np.nan


# %% prediction is average of all previous values
def average(y: np.array):
    if y.size < 1:
        return np.nan

    return y.mean()


# %% Prediction is moving average of M previous values
def moving_average(y: np.array, m=4):
    if y.size < m:
        return np.nan

    return np.mean(y[-m:])


# %% prediction is a linear combination
def calculate_weights(y: np.array, m: int):
    n = y.size  # n is number of elements
    if n < 2 * m:  # We need > 2 m of elements
        return np.nan
    M = y[-(m + 1):-1]  #   Select the last elements
    for i in range(1, m):  # create a matrix M of coefficients
        M = np.vstack([M, y[-(m + i + 1):-(i + 1)]])

    v = np.flip(y[-m:])  # select de known values
    return np.linalg.solve(M, v)  # solve the system of m equations


def linear_combination(y: np.array, m=4) -> np.ndarray:
    n = y.size
    #  check for at least 2m data points
    if n < 2 * m:
        return np.nan
    # calculate the weights
    a = calculate_weights(y, m)
    # Calculate the predicted value and return the predicted value
    return np.sum(y[-m:] * a)


# %% prediction generator
def predictor(y: np.array, f, *argv):
    i = 0
    while True:
        if i <= y.size:
            yield f(y[:i], *argv)
        else:
            y = np.append(y, f(y, *argv))
            yield f(y, *argv)
        i += 1


# %% utility function
def predict(y: np.array, start, end, f, *argv):
    generator = predictor(y, f, *argv)
    predictions = np.array([next(generator) for _ in range(end)])
    predictions[:start] = np.nan
    return predictions


# %% general regression function
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score as r2score


class GeneralRegression:
    def __init__(self, degree=1, exp=False, log=False):
        self.degree = degree
        self.exp = exp
        self.log = log
        self.model = None
        self.x_orig = None
        self.y_orig = None
        self.X = None
        self.y = None

    def fit(self, x: np.array, y: np.array):
        self.x_orig = x
        self.y_orig = y
        self.X = x.reshape(-1, 1)

        if self.exp:
            self.y = np.log(y)

        else:
            self.y = y

        if self.log:
            self.X = np.log(self.X)

        self.model = make_pipeline(PolynomialFeatures(degree=self.degree), LinearRegression())
        self.model.fit(self.X, self.y)

    def predict(self, x: np.array):
        X = x.reshape(-1, 1)

        if self.exp:
            return np.exp(self.model.predict(X))

        if self.log:
            return self.model.predict(np.log(X))

        return self.model.predict(X)

    @property
    def r2_score(self):
        return r2score(self.y_orig, self.predict(self.x_orig))

    @property
    def se_(self):
        if self.exp:
            return mean_squared_error(self.predict(self.X), np.exp(self.y), squared=False)
        if self.log:
            return mean_squared_error(self.predict(self.X), np.log(self.y), squared=False)
        return mean_squared_error(self.predict(self.X), self.y, squared=False)

    @property
    def coef_(self):
        return self.model.steps[1][1].coef_

    @property
    def intercept_(self):
        return self.model.steps[1][1].intercept_

    def get_feature_names(self):
        return self.model.steps[0][1].get_feature_names()


# %% trend model
def create_trend_model(y: np.array):
    X = np.arange(0, y.size)  # We build a linear regression model
    model = GeneralRegression()
    model.fit(X, y)

    return lambda x: model.predict(np.array(x).reshape(-1, 1))  # We return a predictor function


# %%
def forecast_errors(x: np.array, f: np.array, method: str):
    e = x - f
    mae = np.nanmean(np.abs(e))
    rmse = np.sqrt(np.nanmean(e ** 2))
    mape = np.nanmean(np.abs(e / x))
    return pd.DataFrame({'MAE': [mae], 'RMSE': [rmse], 'MAPE': [mape]}, index=[method])


# %% autocorrelate period
def find_period(y: np.array, maxlags=10, top_n=1) -> int:
    # autocorrelate at both sides
    acfs = np.correlate(y, y, mode='full') / np.sum(y ** 2)
    # define the middle
    middle = acfs.size // 2
    # reverse argsort from (middle + 1) to maxlags + top selection
    return (np.argsort(-1 * acfs[middle + 1: middle + maxlags]) + 1)[:top_n]


# %% smoother
def smooth(y: np.array, m: int):
    result = np.empty(0)
    for i in range(y.size - m + 1):
        result = np.append(result, [np.mean(y[i:i + m])])

    return result


# %% double filter function
def find_trend(y: np.array, m: int):
    result = smooth(y, m)
    nan = [np.nan] * int(m / 2)
    if m % 2 == 0:
        result = smooth(result, 2)
        result = np.hstack([nan, result, nan])

    return result


# %%Calculate seasonal averages
def find_seasons(y: np.array, m: int, method='additive'):
    if method == 'multiplicative':
        seasonal_noise = y / find_trend(y, m)
    else:
        seasonal_noise = y - find_trend(y, m)

    n = seasonal_noise.size

    seasonal_pattern = np.empty(0)
    for i in range(m):  # m groups of means that are always m steps apart
        seasonal_pattern = np.append(seasonal_pattern, np.nanmean(seasonal_noise[np.arange(i, n, m)]))

    # repeat pattern over full period of time
    return np.tile(seasonal_pattern, n // m)  # n // m is the number of seasons.


# %% find regression models
def find_regression_models(z: np.array, m: int, degree=1, exp=False):
    reg_models = []

    for i in range(z.size // m - 1):
        x = np.arange(i, revenues.size, m).reshape(-1, 1)
        y = z[x]
        reg_models.append(GeneralRegression(degree, exp))
        reg_models[i].fit(x, y)

    return reg_models


# %% forecasting with seasonal decomposition components
def seasonal_decomposition_forecast(reg_model: GeneralRegression, sd_model, start, end, method='additive', m=None):
    if not m:
        m = find_period(sd_model.observed)[0]

    # Repeat seasons beyond 'end'
    seasonal = np.tile(sd_model.seasonal[0:m], end // m + 1)
    if method.startswith('m'):
        return reg_model.predict(np.arange(start, end)) * seasonal[start:end]
    else:
        return reg_model.predict(np.arange(start, end)) + seasonal[start:end]


# %% seasonal_trend_forecast
def create_seasonal_trend_forecast(z: np.array, m: int, degree=1, exp=False):
    reg_models = find_regression_models(z, m, degree, exp)

    def forecast(x: np.array):
        predictions = np.empty(0)

        for i in range(x.size):
            y = reg_models[i % m].predict(x[i].reshape(1, -1))
            predictions = np.append(predictions, y)

        return predictions

    return forecast


# %% compare original and forecast
def plot_trends(y1: np.array, y2=None, title='',sub_title=None, label1='given', label2='predicted', color='C0', ax=None):
    if y2 is not None:
        n = max(y1.size, y2.size)
    else:
        n = y1.size

    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    if sub_title:
        fig.suptitle(sub_title, y=1.02)

    ax.set_title('revenues past 5 year')
    ax.set_xlabel('quarter')
    ax.set_ylabel('revenue (€)')
    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(['Q{}'.format(j % 4 + 1) for j in range(n)])

    ax.set_xticks(range(n))
    ax.plot(y1, label=label1, color=color, marker='o')
    if y2 is not None:
        ax.plot(y2, label=label2, color='C1', marker='^')
    for i in range(0, n, 4):
        ax.axvline(i, color='gray', linewidth=0.5)

    ax.legend()

# %% seasonal decomposition plotten
def plot_seasonal_decompositon(model, title: str, figsize=(8, 8)):
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=figsize)

    axes[0].plot(model.observed, 'o-', label='observed')
    axes[0].set_ylabel('observed')
    axes[0].set_title(title)
    axes[0].legend()

    axes[1].plot(model.trend, 'o-', color='orange', label='trend')
    axes[1].set_ylabel('trend')
    axes[1].legend()

    axes[2].plot(model.seasonal, 'o-', color='green', label='seasonal')
    axes[2].set_ylabel('season')
    axes[2].legend()

    axes[3].scatter(range(model.nobs[0]), model.resid, color='darkgrey', label='noise')
    axes[3].set_ylabel('residue')
    axes[3].set_xlabel('kwartaal')
    axes[3].legend()

def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error = np.abs(y_true - y_pred)
    mae = np.mean(error)
    return mae

def rmse(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error = y_true - y_pred
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    return rmse

def mape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(error) * 100
    return mape

def reliabilityTable (past, period):
    overview_1 = []
    overview_2 = []
    overview_3 = []
    overview_4 = []
    result_1 = mae(past, naive)
    result_2 = rmse(past, naive)
    result_3 = mape(past, naive)
    overview_1 = np.append(overview_1, result_1)
    overview_1 = np.append(overview_1, result_2)
    overview_1 = np.append(overview_1, result_3)
    result_1 = mae(past, average)
    result_2 = rmse(past, average)
    result_3 = mape(past, average)
    overview_2 = np.append(overview_2, result_1)
    overview_2 = np.append(overview_2, result_2)
    overview_2 = np.append(overview_2, result_3)
    result_1 = mae(past, moving_average(period))
    result_2 = rmse(past, moving_average(period))
    result_3 = mape(past, moving_average(period))
    overview_3 = np.append(overview_3, result_1)
    overview_3 = np.append(overview_3, result_2)
    overview_3 = np.append(overview_3, result_3)
    result_1 = mae(past, linear_combination(period))
    result_2 = rmse(past, linear_combination(period))
    result_3 = mape(past, linear_combination(period))
    overview_4 = np.append(overview_4, result_1)
    overview_4 = np.append(overview_4, result_2)
    overview_4 = np.append(overview_4, result_3)
    rel_table=pd.DataFrame (data=np.array([overview_1, overview_2, overview_3, overview_4]), columns=['MAE', 'RSME', 'MAPE'])
    rel_table.index = ['Naive', 'Average', 'Mov. Av.', 'lin. comb.']
    return[rel_table]

past = [20, 100, 175, 13, 37, 136, 245, 26, 75, 155, 326, 48, 92, 202, 384, 82, 176, 282, 445, 181]
reliabilityTable(past,4)
