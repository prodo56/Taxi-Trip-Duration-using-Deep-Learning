import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime,time
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
'''
df = pd.read_csv("train.csv")


def get_part_of_day(time_now):
    """Return part of day depending on time_now and the user's timzone
    offset value.

    From  -  To  => part of day
    ---------------------------
    00:00 - 04:59 => midnight
    05:00 - 06:59 => dawn
    07:00 - 10:59 => morning
    11:00 - 12:59 => noon
    13:00 - 16:59 => afternoon
    17:00 - 18:59 => dusk
    19:00 - 20:59 => evening
    21:00 - 23:59 => night
    """
    format = '%Y-%m-%d %H:%M:%S'
    x= datetime.strptime(time_now, format)
    user_hour = x.hour
    month = x.month
    day = x.day
    if user_hour < 5:
        part_day = 'midnight'
    elif user_hour < 7:
        part_day =  'dawn'
    elif user_hour < 11:
        part_day = 'morning'
    elif user_hour < 13:
        part_day = 'noon'
    elif user_hour < 17:
        part_day = 'afternoon'
    elif user_hour < 19:
        part_day = 'dusk'
    elif user_hour < 21:
        part_day = 'evening'
    else:
        part_day = 'night'
    day_of_week =x.isoweekday()
    isWeekend = 0
    if day_of_week >=6:
        isWeekend = 1
    return part_day,month,day,day_of_week,isWeekend


df= df.drop(labels=['id', 'store_and_fwd_flag'],axis=1)
print df.columns
pickupdate = []
dropoffdate =[]
pickup_days=[]
pickup_day_week = []
pickup_weekend=[]
pickup_month=[]

dropoff_days=[]
dropoff_day_week = []
dropoff_weekend=[]
dropoff_month=[]

for index, row in df.iterrows():
    part_day, month, day, day_of_week, isWeekend = get_part_of_day(row['pickup_datetime'])
    pickupdate.append(part_day)
    pickup_days.append(day)
    pickup_day_week.append(day_of_week)
    pickup_weekend.append(isWeekend)
    pickup_month.append(month)
    part_day, month, day, day_of_week, isWeekend = get_part_of_day(row['dropoff_datetime'])
    dropoffdate.append(part_day)
    dropoff_days.append(day)
    dropoff_day_week.append(day_of_week)
    dropoff_weekend.append(isWeekend)
    dropoff_month.append(month)

df= df.drop(labels=['dropoff_datetime', 'pickup_datetime'],axis=1)
df['pickup_part_of_day'] = pickupdate
df['pickup_day_of_week'] = pickup_day_week
df['pickup_month'] = pickup_month
df['pickup_isWeekend'] = pickup_weekend
df['pickup_day'] = pickup_days


df['dropoff_part_of_day'] = dropoffdate
df['dropoff_day_of_week'] = dropoff_day_week
df['dropoff_month'] = dropoff_month
df['dropoff_isWeekend'] = dropoff_weekend
df['dropoff_day'] = dropoff_days


print df.columns

df.to_csv("train_new.csv")
'''
df = pd.read_csv("train_new.csv")



y_train = df.iloc[:,6].values
df = df.drop(labels=['trip_duration'],axis=1)
X= df.iloc[:,1:].values
encoder = LabelEncoder()
encoder2 = LabelEncoder()
X[:,6] = encoder.fit_transform(X[:,6])
X[:,11] = encoder2.fit_transform(X[:,11])
#onehot = OneHotEncoder(categorical_features=[6,11])
#X = onehot.fit_transform(X).toarray()

scaler = StandardScaler()
X_train = scaler.fit_transform(X)

print type(X_train)

'''classifier = Sequential()
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1))
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])
classifier.fit(X_train,y_train,batch_size=32,epochs=50)'''






def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1))
    classifier.compile(optimizer = optimizer, loss = 'mean_squared_error')
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [64,300,400],
              'epochs': [25, 50],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print best_accuracy
print best_parameters

