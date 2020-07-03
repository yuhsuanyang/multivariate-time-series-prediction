import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from matplotlib import pyplot



def mae(pred, truth, target):
    ans=0
    size=pred.shape[0]
    for i in range(size):
        diff=abs(pred[i,:,target]-truth[i,:,target])
        ans+=diff.mean()
    return ans/size

def smape(pred, truth, M, m, target):
    ans=0
    size=pred.shape[0]
    for i in range(size):
        diff=2*abs(pred[i,:,target]-truth[i,:,target])*M
        numer=abs(pred[i,:,target]*M+m)+abs(truth[i,:,target]*M+m)
        ans+=(diff/numer).mean()
    return ans/size


#-------exchange rate------------------------

exchange=pd.read_csv('exchange_rate.csv')
del exchange['China']

data=(exchange-exchange[:-2000].min())/(exchange[:-2000].max()-exchange[:-2000].min())
scale=[exchange[:-2000].max().values,exchange[:-2000].min().values]

data1=data.diff(1).dropna()

#train=exchange[:-2000].values
#test=exchange[-2000:].values

train=data1[:-2000].values
test=data1[-2000:].values



model=VAR(train)
results = model.fit(14)

predictions=[]
truth=[]

for i in range(len(test)-21):
    truth.append(data[5587+i+14:5587+i+21].values)
    predict=results.forecast(test[i:i+14,], 7)
    last_step=data.loc[5586+i+14].values
    predict[0,:]=predict[0,:]+last_step
    for j in range(1,7):
        predict[j,:]=predict[j,:]+predict[j-1,:]
    predictions.append(predict)


predictions=np.array(predictions)
truth=np.array(truth)



mae(predictions,truth)
smape(predictions,truth,(scale[0]-scale[1]),scale[1])


 
maes=[]
for i in range(7):
    maes.append(mae(predictions,truth,i))
    
smapes=[]
for i in range(7):
    smapes.append(smape(predictions,truth,(scale[0][i]-scale[1][i]),scale[1][i],i))
    
   
np.mean(smapes)
np.std(smapes)   
np.mean(maes)
np.std(maes)

#-------air quality-------------------------
air=pd.read_csv('air_quality_processed.csv')
air=air.drop(columns=['Date','Time'])
air=air.dropna()

data=(air-air[:7154].min())/(air[:7154].max()-air[:7154].min())
scale=[air[:7154].max().values,air.min()[:7154].values]

train=data[:7154].values
test=data[7154:].values


model=VAR(train)
results = model.fit(24)

predictions=[]
truth=[]


for i in range(len(test)-36):
    truth.append(test[i+24:i+36,:].tolist())
    predictions.append(results.forecast(test[i:i+24,], 12).tolist())
    
predictions=np.array(predictions)
truth=np.array(truth)


 
maes=[]
for i in range(9):
    maes.append(mae(predictions,truth,i))
    
smapes=[]
for i in range(9):
    smapes.append(smape(predictions,truth,(scale[0][i]-scale[1][i]),scale[1][i],i))
    
   
np.mean(smapes)
np.std(smapes)   
np.mean(maes)
np.std(maes)

#----------household comsumption-------------------
electricity=pd.read_csv('household_consumption.csv').loc[3186:13097]
del electricity['Datetime']

data=(electricity-electricity.loc[3186:10001].min())/(electricity.loc[3186:10001].max()-electricity.loc[3186:10001].min()) 
scale=[electricity.loc[3186:10001].max().values,electricity.loc[3186:10001].min().values]

train=data[:-3096].values
test=data[-3096:].values



model=VAR(train)
results = model.fit(24)

predictions=[]
truth=[]


for i in range(len(test)-36):
    truth.append(test[i+24:i+36].tolist())
    predictions.append(results.forecast(test[i:i+24,], 12).tolist())
    
predictions=np.array(predictions)
truth=np.array(truth)


 
maes=[]
for i in range(4):
    maes.append(mae(predictions,truth,i))
    
smapes=[]
for i in range(4):
    smapes.append(smape(predictions,truth,(scale[0][i]-scale[1][i]),scale[1][i],i))
    
   
np.mean(smapes)
np.std(smapes)   
np.mean(maes)
np.std(maes)


#---------stock-------------------------
stock=pd.read_csv('stock_preprocess.csv')
data=(stock-stock[:-2993].min())/(stock[:-2993].max()-stock[:-2993].min())
scale=[stock[:-2993].max().tolist(), stock[:-2993].min().tolist()]

data1=data.diff(1).dropna()

#train=exchange[:-2000].values
#test=exchange[-2000:].values

train=data1[:-2993].values
test=data1[-2993:].values



model=VAR(train)
results = model.fit(20)

predictions=[]
truth=[]

for i in range(len(test)-30):
    truth.append(data[6000+i+20:6000+i+30].values)
    predict=results.forecast(test[i:i+20,], 10)
    last_step=data.loc[5999+i+10].values
    predict[0,:]=predict[0,:]+last_step
    for j in range(1,9):
        predict[j,:]=predict[j,:]+predict[j-1,:]
    predictions.append(predict)


predictions=np.array(predictions)
truth=np.array(truth)

maes=[]
for i in range(9):
    maes.append(mae(predictions,truth,i))
    
smapes=[]
for i in range(9):
    smapes.append(smape(predictions,truth,(scale[0][i]-scale[1][i]),scale[1][i],i))
    



   
np.mean(smapes)*100
np.std(smapes)*100   
np.mean(maes)
np.std(maes)


#-----performaace of first and last-----------------
maes=[]
for i in range(4):
    maes.append(mae(predictions[:,0:1,:],truth[:,0:1,:],i))
    
smapes=[]
for i in range(4):
    smapes.append(smape(predictions[:,0:1,:],truth[:,0:1,:],(scale[0][i]-scale[1][i]),scale[1][i],i))




#-----use the last stamp as prediction--------------
prediction=[]
truth=[]

for i in range(len(test)-21):
    truth.append(test[i+14:i+21])
    prediction.append(np.repeat(test[i+14],7).reshape(7,7).transpose())


prediction =np.array(prediction)
truth=np.array(truth)


 
maes=[]
for i in range(7):
    maes.append(mae(prediction,truth,i))
    
smapes=[]
for i in range(7):
    smapes.append(smape(prediction,truth,(scale[0][i]-scale[1][i]),scale[1][i],i))
    
   
np.mean(smapes)
np.std(smapes)   
np.mean(maes)
np.std(maes)

