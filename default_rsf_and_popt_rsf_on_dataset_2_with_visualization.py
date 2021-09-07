from google.colab import files
uploaded = files.upload()

pip install lifelines

import pandas as pd
import numpy as np

from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import seaborn as sn
import matplotlib.pyplot as plt

#Read the data file:
data = pd.read_csv("/content/patients_data.csv")
data.head()

#Columns of dataset:
data.columns = data.columns.str.replace(' ', '')
data.columns

#Drop rows with null values:

data= data.dropna(subset=['DateAnnounced', 'AgeBracket', 'Gender',
       'DetectedState', 'CurrentStatus', 'StatusChangeDate', 'Duration'])

data.head()

#Organize the data:

data.loc[data.Gender == 'M', 'Sexint'] = 1
data.loc[data.Gender == 'F', 'Sexint'] = 2
data.loc[data.CurrentStatus == 'Recovered', 'Death'] = 0
data.loc[data.CurrentStatus == 'Hospitalized', 'Death'] = 0
data.loc[data.CurrentStatus == 'Deceased', 'Death'] = 1
data.head()

data.loc[data.CurrentStatus == 'Recovered', 'event'] = False
data.loc[data.CurrentStatus == 'Hospitalized', 'event'] = False
data.loc[data.CurrentStatus == 'Deceased', 'event'] = True
data.head()

#data= data[['AgeBracket', 'Duration', 'Sexint', 'Death']]
data= data[['AgeBracket', 'Duration', 'Sexint', 'Death','event']]

X = data[['AgeBracket', 'Duration', 'Sexint']]
y= data['Death']
#y= data['event']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
data1=pd.concat([X_train, y_train], axis=1, join='inner')
data2=pd.concat([X_test, y_test], axis=1, join='inner')

i=20
while(i<=1000):
  clf = RandomForestRegressor(n_estimators=i)
  clf.fit(X_train,y_train)
  y_pred = clf.predict(X_test)
  print("For no. of trees ",i)
  print(f'Concordance index: {concordance_index(y_test, y_pred)}')
  i=i+40

i=2
while(i<=30):
  clf = RandomForestRegressor(n_estimators=660, max_depth=i)
  clf.fit(X_train,y_train)
  y_pred = clf.predict(X_test)
  print("For Max Depth ",i)
  print(f'Concordance index: {concordance_index(y_test, y_pred)}')
  i=i+1

i=2
while(i<=50):
  clf = RandomForestRegressor(n_estimators=660,max_depth=3,min_samples_split=i)
  clf.fit(X_train,y_train)
  y_pred = clf.predict(X_test)
  print("For Minimum Sample split ",i)
  print(f'Concordance index: {concordance_index(y_test, y_pred)}')
  i=i+1

j=1
while(j<=3):
  clf = RandomForestRegressor(n_estimators=660,max_depth=3,min_samples_split=8,max_features=j)
  clf.fit(X_train,y_train)
  y_pred = clf.predict(X_test)
  print("For Max Features ",j)
  print(f'Concordance index: {concordance_index(y_test, y_pred)}')
  j=j + 1

clf = RandomForestRegressor(n_estimators=660,max_depth=3,min_samples_split=8,max_features=3)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f'Concordance index: {concordance_index(y_test, y_pred)}')

clr = RandomForestRegressor()
clr.fit(X_train,y_train)
y_pred3 = clr.predict(X_test)
print(f'Concordance index: {concordance_index(y_test, y_pred3)}')

featureImportances = pd.Series(clr.feature_importances_)
print(featureImportances)

sn.barplot(x=featureImportances, y=X.columns)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

"""**2nd library for RSF to plot Graph**"""

pip install -U pip setuptools

pip install scikit-survival

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

a=data[['AgeBracket', 'Sexint','Duration']]
b= data[["event","Duration"]]

tuples = [tuple(x) for x in b.to_numpy()]

dt=np.array(tuples, dtype=[('cens', '?'), ('time', '<f8')])

X_train, X_test, y_train, y_test = train_test_split(
    a, dt, test_size=0.25, random_state=0)

rsf = RandomSurvivalForest(n_estimators=660,max_depth=3,min_samples_split=8,max_features=3)
rsf.fit(X_train,y_train)

def_rsf = RandomSurvivalForest()
def_rsf.fit(X_train,y_train)

"""# **Performance**"""

def perf_measure1(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    k=0
    Xtp1 = pd.DataFrame(columns=['AgeBracket', 'Duration', 'Sexint'])
    Xtn1 = pd.DataFrame(columns=['AgeBracket', 'Duration', 'Sexint'])
    Xfp1 = pd.DataFrame(columns=['AgeBracket', 'Duration', 'Sexint'])
    Xfn1 = pd.DataFrame(columns=['AgeBracket', 'Duration', 'Sexint'])
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
           Xtp=X_test[i:i+1]
           Xtp1 = Xtp1.append(Xtp, ignore_index=True)
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
           Xfp=X_test[i:i+1]
           Xfp1 = Xfp1.append(Xfp, ignore_index=True)
        if y_actual[i]==y_hat[i]==0:
           TN += 1
           Xtn=X_test[i:i+1]
           Xtn1 = Xtn1.append(Xtn, ignore_index=True)
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
          Xfn=X_test[i:i+1]
          Xfn1 = Xfn1.append(Xfn, ignore_index=True)
          FN += 1

    return(TP, FP, TN, FN, Xtp1,Xfp1,Xtn1,Xfn1)

actual = np.where(y_test < 0.48, 1,0)
predicted_def = np.where(y_pred3 < 0.48, 1,0)
predicted_our=np.where(y_pred < 0.48, 1,0)

tp1,fp1,tn1,fn1,Xtp1,Xfp1,Xtn1,Xfn1=perf_measure1(actual,predicted_def)
tp2,fp2,tn2,fn2,Xtp2,Xfp2,Xtn2,Xfn2=perf_measure1(actual,predicted_our)

acc1=(tp1+tn1)/(tp1+fp1+tn1+fn1)
acc2=(tp2+tn2)/(tp2+fp2+tn2+fn2)
print("Accuracy of the default RSF:",acc1)
print("Accuracy of Our RSF:",acc2)

"""# **Paper 1 Works:**"""

surv1 = def_rsf.predict_survival_function(Xtp1, return_array=True)
survAvg1=surv1.mean(axis=0)
surv5 = rsf.predict_survival_function(Xtp2, return_array=True)
survAvg5=surv5.mean(axis=0)

conc_arr1 = np.append([survAvg1], [survAvg5], axis=0)

labelhead=["Default RSF", "Popt-RSF"]

plt.figure(figsize=(8,8))
plt.title("True Positive")
for i, s in enumerate(conc_arr1):
    plt.step(def_rsf.event_times_, s, where="post", label=labelhead[i])
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)

surv2 = def_rsf.predict_survival_function(Xfp1, return_array=True)
survAvg2=surv2.mean(axis=0)

surv6 = rsf.predict_survival_function(Xfp2, return_array=True)
survAvg6=surv6.mean(axis=0)

conc_arr2 = np.append([survAvg2], [survAvg6], axis=0)

labelhead=["Default RSF", "Popt-RSF"]

plt.figure(figsize=(8,8))
plt.title("False Positive")
for i, s in enumerate(conc_arr2):
    plt.step(def_rsf.event_times_, s, where="post", label=labelhead[i])
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)

surv3 = def_rsf.predict_survival_function(Xtn1, return_array=True)
survAvg3=surv3.mean(axis=0)
surv7 = rsf.predict_survival_function(Xtn2, return_array=True)
survAvg7=surv7.mean(axis=0)

conc_arr3 = np.append([survAvg3], [survAvg7], axis=0)

labelhead=["Default RSF", "Popt-RSF"]

plt.figure(figsize=(8,8))
plt.title("True Negative")
for i, s in enumerate(conc_arr3):
    plt.step(def_rsf.event_times_, s, where="post", label=labelhead[i])
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)

surv4 = def_rsf.predict_survival_function(Xfn1, return_array=True)
survAvg4=surv4.mean(axis=0)
surv8 = rsf.predict_survival_function(Xfn2, return_array=True)
survAvg8=surv8.mean(axis=0)

conc_arr4 = np.append([survAvg4], [survAvg8], axis=0)

labelhead=["Default RSF", "Popt-RSF"]

plt.figure(figsize=(8,8))
plt.title("False Negative")
for i, s in enumerate(conc_arr4):
    plt.step(def_rsf.event_times_, s, where="post", label=labelhead[i])
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)