import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

#import zillow data 
real_estate=pd.read_csv("city_final.csv")
real_estate1=pd.concat([real_estate,year],axis=1)
real_estate1=real_estate1[["City","CountyName","value","new_date"]]
real_estate1['new_date']=pd.to_datetime(real_estate1['new_date'])

#average metrics 
#from 1996 (apr)-2018 (dec)
avg_market=real_estate1.groupby('CountyName')['value'].mean()
avg_market.sort_values(ascending=False) 
avg_market=pd.DataFrame(avg_market)
avg_market.reset_index(level=0,inplace=True)

#1996-2007 
pre_market=pre.groupby('CountyName')['value'].mean()
pre_market.sort_values(ascending=False) 
pre_market=pd.DataFrame(pre_market)
pre_market.reset_index(level=0,inplace=True)
pre_market.columns=['CountyName','value_pre']

#2008-current 
post_market=post.groupby('CountyName')['value'].mean()
post_market.sort_values(ascending=False) 
post_market=pd.DataFrame(post_market)
post_market.reset_index(level=0,inplace=True)
post_market.columns=['CountyName','value_post']

#biggest change average 1996-2007 
pre=real_estate1[0:500010]
post=real_estate1[500010:]

#differences between the two eras 
comps=pd.merge(pre_market,post_market,on="CountyName")
comps['diff']=comps['value_post']=comps['value_pre']
comps.sort_values('diff',ascending=False)[['CountyName','diff']]

# time series modeling 
for i in range(6,25):
    oc_avg["lag_{}".format(i)]=oc_avg.value.shift(i)

#ts cross-validated set 5 folds 
tscv=TimeSeriesSplit(n_splits=5)

#train and test data 
X=oc_avg.iloc[:,1:oc_avg.shape[1]]
y=oc_avg.iloc[:,0]

#remove missing values
y=oc_avg.dropna()
X=X.dropna() 
X_train,X_test,y_train,y_test=train_test_split(X,y)

#model results 
model_oc=LinearRegression().fit(X_train,y_train)
predict_oc=model_oc.predict(X_test)

#cross-validation scores 
cv=cross_val_score(model_oc,X_train,y_train,cv=tscv,scoring="neg_mean_squared_error")
mae=cv.mean()*(-1)

#upper and lower bound bounds 
deviation=np.sqrt(cv.std()) #sd 
scale=1.96 
lower_bound=predict_oc-(scale*deviation)
upper_bound=predict_oc+(scale*deviation)

#model coefficients 
coefs=pd.DataFrame(model_oc.coef_)
coefs.columns=X_train.columns 
coefs_avg=coefs.iloc[:,0:coefs.shape[1]].mean()
coefs_avg=pd.DataFrame(coefs_avg)
coefs_avg.columns=["abs"]
coefs_avg=coefs_avg.sort_values(by="abs",ascending=False) #lag 6 most sign for predicting time series value at time t

#a. Plot predicted vs. actual real estate value
predict_oc=pd.DataFrame(predict_oc)
predict_oc_l6=predict_oc.iloc[:,6]
predict_oc_l6=pd.DataFrame(predict_oc_l6)
predict_oc_l6.columns=['lag_6_predict']
p6=predict_oc_l6

test=y_test 
test=test["value"]
test=pd.DataFrame(test)
test.columns=['acutal_value']
test.reset_index(level=0,inplace=True)

combo_real=pd.concat([test,p6],axis=1)

plt.figure(figsize=(15,8))
plt.plot(combo_real['acutal_value'],"g",label="actual value",linewidth=2.0)
plt.plot(combo_real['lag_6_predict'],"r",label="predicted value",linewidth=2.0)
plt.show() 

#b. plot coefficients
plt.figure(figsize=(15, 7))
coefs_avg['abs'].plot(kind='bar')
plt.grid(True, axis='y')
plt.hlines(y=0, xmin=0, xmax=len(coefs_avg), linestyles='dashed')
plt.show() 











































