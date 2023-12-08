from matplotlib import pyplot
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM,Activation,Dropout
from sklearn.metrics import mean_squared_error
from keras import backend as k
df = pd.read_csv('nyc-weather.csv')
df.drop(['name','conditions','description','icon','stations','sunrise','sunset'],axis=1,inplace=True)
df.head()
df.corr(numeric_only=True)
df.drop(['snow','snowdepth','severerisk'],axis=1,inplace=True)
df.corr(numeric_only=True)
sn.heatmap(df.corr(numeric_only=True))
df.preciptype.unique()
df.head()
df.drop(['preciptype','windgust','solarenergy','solarradiation','uvindex'],axis=1,inplace=True)
df.corr(numeric_only=True)

# convert series to supervised learning 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True): 
    n_vars = 1 if type(data) is list else data.shape[1] 
    df = pd.DataFrame(data) 
    cols, names = list(), list() 
    # input sequence (t-n, ... t-1) 
    for i in range(n_in, 0, -1):  
        cols.append(df.shift(i))  
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)] 
    # forecast sequence (t, t+1, ... t+n) 
    for i in range(0, n_out):  
        cols.append(df.shift(-i))  
        if i == 0:   
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]  
        else:   
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)] 
    # put it all together 
    agg = pd.concat(cols, axis=1) 
    agg.columns = names 
    # drop rows with NaN values 
    if dropnan:  
        agg.dropna(inplace=True) 
    return agg 

df = df.set_index(pd.DatetimeIndex(df['datetime']))
df=df.drop('datetime',axis=1)
df.index.name='datetime'
data = df.values
data=data.astype('float64')
# normalize features
scaler=MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(data)
# specify no. of lag days
n_days = 5
n_features = 17
# frame as supervised learning
reframed = series_to_supervised(scaled,n_days,1)
reframed
values = reframed.values

train = values[:int(len(values)*0.8),:]
test = values[int(len(values)*0.8):,:]

# split into input and outputs
n_obs = n_days * n_features
train_X,train_Y = train[:,:n_obs],train[:,-15]   # as at 15th no. from last is temp. our o/p var
test_X,test_Y = test[:,:n_obs],test[:,-15]
print(test_Y)
print(train_X.shape,len(train_X),train_Y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0],n_days,n_features))
test_X = test_X.reshape((test_X.shape[0],n_days,n_features))
print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)

hidden_nodes = int(2/3 * (train_X.shape[1] * train_X.shape[2]))
hidden_nodes

# design network
model = Sequential()
model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2])))    # input_shape=(no. of i/p, dimension), result=(1,50)
# test
model.add(Dense(256,name='FC1'))  #256
model.add(Activation('relu'))
model.add(Dropout(0.2))
# end
model.add(Dense(1,name='out_layer'))
model.compile(loss='mean_squared_error',optimizer='adam', metrics=['mean_squared_error'])

model.summary()

# fit network
history = model.fit(train_X, train_Y, epochs=100, batch_size=128, validation_data=(test_X, test_Y), verbose=2, shuffle=False)  #, verbose=2, shuffle=False
model.save('LSTM.h5')
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# make a prediction
test_Y_predicted = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0],n_days*n_features))


# invert scaling for forecast
vir_array = np.concatenate((test_X[:, -17:-15],test_Y_predicted), axis=1)
inv_test_Y_predicted = np.concatenate((vir_array,test_X[:, -14:]), axis=1)
inv_test_Y_predicted = scaler.inverse_transform(inv_test_Y_predicted)
inv_test_Y_predicted = inv_test_Y_predicted[:,-15]
inv_test_Y_predicted

# invert scaling for actual
test_Y = test_Y.reshape((len(test_Y), 1))
vir_array2 = np.concatenate((test_X[:, -17:-15],test_Y), axis=1)
inv_test_Y = np.concatenate((vir_array2,test_X[:, -14:]), axis=1)
inv_test_Y = scaler.inverse_transform(inv_test_Y)
inv_test_Y = inv_test_Y[:,-15]
inv_test_Y

pyplot.plot(inv_test_Y[:100], label='Actual')
pyplot.plot(inv_test_Y_predicted[:100], label='Predicted')


# pyplot.legend()
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_test_Y, inv_test_Y_predicted))
print('Test RMSE: %.3f' % rmse)
# comparing actual temperature and predicted temperature predicted using data of previous 3 days
df_result = pd.DataFrame({'Actual':inv_test_Y,'Predicted':inv_test_Y_predicted})

df_result
json_result = df_result.to_json(orient='records', lines=True)

print(json_result)


test_dates = df.index[int(len(values)*0.8) + n_days:]
test_dates

aligned_dates = test_dates[500:600]

pyplot.figure(figsize=(10, 6))
pyplot.plot(aligned_dates, inv_test_Y[500:600], label='Actual')
pyplot.plot(aligned_dates, inv_test_Y_predicted[500:600], label='Predicted')
pyplot.xticks(rotation=45)
pyplot.legend()
pyplot.show()