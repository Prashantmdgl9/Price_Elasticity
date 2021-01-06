# Imports
from __future__ import print_function
import statsmodels.graphics.api as smg
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from statsmodels.graphics.regressionplots import plot_partregress_grid
import datetime as dt
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import turicreate as tc
from statsmodels.compat import lzip
import statsmodels.api as sm
from statsmodels.formula.api import ols
%matplotlib inline
sf = tc.SFrame.read_csv("avocado.csv")
sf.print_rows(2)


'''
Small/Medium Hass Avocado (~3-5oz avocado) | #4046

Large Hass Avocado (~8-10oz avocado) | #4225

Extra Large Hass Avocado (~10-15oz avocado) | #4770

Hass Avocado Bags | Size varies

'''
tc.visualization.set_target(target='browser')

sf = sf.rename({'Total Volume': 'Volume'})
sf = sf.rename({'Total Bags': 'Bags'})
sf = sf.rename({'4225': 'twent_fv_Av'})
sf = sf.rename({'4046': 'for_si_Av'})
sf = sf.rename({'4770': 'sev_sev_Av'})

sf.print_rows(3)
sf.show()


qtr = []
for item in sf['Date']:
    date_i = dt.datetime.strptime(item, '%Y-%m-%d')
    qtr.append((date_i.month + 2) // 3)
sf['qtr'] = qtr


sf_g = sf.groupby(['year', 'qtr'], tc.aggregate.MEAN(
    'Volume'), tc.aggregate.MEAN('AveragePrice'))
sf_g = sf_g.sort(['year', 'qtr'])

sf_g = sf_g.rename({'Avg of Volume': 'Volume', 'Avg of AveragePrice': 'Price'})

sf_g

tc.show(sf_g['Price'], sf_g['Volume'], xlabel="Price in $",
        ylabel="Demand", title="Demand-Supply Curve")


df_g = sf_g.to_dataframe()


def plt_x():
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, subplot_kw=dict(
        frameon=False), figsize=(15, 8))

    plt.subplots_adjust(hspace=.01)
    ax1.grid()
    ax2.grid()

    ax1.plot(df_g['Price'], color='g')
    ax2.plot(df_g['Volume'], color='b')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Volume')


plt_x()





# model


# Turicreate Model

model_1 = tc.regression.create(
    sf, target='AveragePrice', features=['twent_fv_Av', 'Bags', 'region', 'type', 'year'])

predictions = model_1.predict(sf)
result = model_1.evaluate(sf)
print(result)


# OLS model
df = sf.to_dataframe()

X = df[['twent_fv_Av', 'Bags','type', 'qtr']]
y = df['AveragePrice']
X = pd.get_dummies(X, prefix=["type"], columns=["type"], drop_first = True)

mod = sm.OLS(y, X).fit()
mod.summary()

# Other model

model_2 = ols(
    " AveragePrice ~ twent_fv_Av + Bags + type + qtr+ region", data=df).fit()
print(model_2.summary())


fig = plt.figure(figsize=(8, 8))
fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
fig



def plt_reg():
    fig, scatter = plt.subplots(figsize=(15, 7))
    sns.set_theme(color_codes=True)
    sns.regplot(x=df_g['Price'], y=df_g['Volume'])


plt_reg()

'''
PED = dQ/dP * (P/Q)
'''
PED = ((0.7 - 1.1)/(1.2-1.7)) * (1.7/0.7)
PED
