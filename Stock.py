import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpl_finance as mpf
import codecs as cd
import numpy as np

df = pd.read_csv("4641.csv")
#print(df.head())



from sklearn import linear_model
clf = linear_model.LinearRegression()

df = df.drop("Date", axis=1)

df_except_close = df.drop("Close", axis=1)

X = df_except_close.values

Y = df['Close'].values

clf.fit(X[0:(len(X)-150-15)], Y[15:(len(X)-150)])

# 偏回帰係数
print(pd.DataFrame({"Name": df_except_close.columns,
                    "Coefficients": clf.coef_}).sort_values(by='Coefficients'))

# 切片 (誤差)
print(clf.intercept_)

vector=range(0,len(df))

plt.plot(vector, clf.predict(X),"x-",label="prediction")
plt.plot(vector,Y,label="Real price")

plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

plt.grid(True)
plt.show()
