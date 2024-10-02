# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('Housing.csv')

df.shape
df.head()
df.info()

sns.pairplot(df)

sns.heatmap(df.corr(),annot=True)

for i in df.index:
  df['mainroad'][i] = 1 if df['mainroad'][i] == 'yes' else 0
  df['guestroom'][i] = 1 if df['guestroom'][i] == 'yes' else 0
  df['basement'][i] = 1 if df['basement'][i] == 'yes' else 0
  df['hotwaterheating'][i] = 1 if df['hotwaterheating'][i] == 'yes' else 0
  df['airconditioning'][i] = 1 if df['airconditioning'][i] == 'yes' else 0
  df['prefarea'][i] = 1 if df['prefarea'][i] == 'yes' else 0

  match df['furnishingstatus'][i]:
    case 'unfurnished':
      df['furnishingstatus'][i] = 1
    case 'semi-furnished':
      df['furnishingstatus'] = 2
    case 'furnished':
      df['furnishingstatus'][i] = 3

df.head()
# df['mainroad'] = np.where(df['mainroad'] == 'yes', 1, 0)
# df['guestroom'] = np.where(df['guestroom'] == 'yes', 1, 0)
# df['mainroad'] = .where((df['mainroad'] == 'yes') & (df['price'] == 12215000 ), 1)
# df.loc[df['mainroad'] == 'no','mainroad'] = 0


X=df[['area', 'bedrooms', 'bathrooms', 'stories','parking','mainroad',
      'guestroom', 'basement', 'hotwaterheating', 'airconditioning','parking', 'prefarea']]
y=df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)
X_train.shape,X_test.shape
y_train.shape, y_test.shape
X_train.dtypes
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
coeff_df=pd.DataFrame(lm.coef_,X.columns,columns=['coeffiecient'])
coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.displot((y_test-predictions),bins=50);







