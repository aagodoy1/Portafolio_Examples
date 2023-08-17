import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm



df=pd.read_csv('shampoo.csv')


df.columns=["Month","Sales"]
df.head()
df.describe()
df.set_index('Month',inplace=True)

rcParams['figure.figsize'] = 7, 4
df.plot()

plt.show()

test_result=adfuller(df['Sales'])


def adfuller_test(result, sales):
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis (Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis, indicating it is non-stationary")



adfuller_test(test_result, df['Sales'])

df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)
df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)
print(df.head())

#adfuller_test(df['Seasonal First Difference'].dropna())


first_dif =df['Seasonal First Difference'].dropna()


adfuller_test(adfuller(first_dif), first_dif)
first_dif.plot()
plt.show()


autocorrelation_plot(df['Sales'])
plt.show()

#fig = plt.figure(figsize=(12,8))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(first_dif,lags=40,ax=ax1)
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(first_dif,lags=40,ax=ax2)


print('fin de del codigo')