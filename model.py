# -*- coding: utf-8 -*-

# Import libraries.
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


## Creating the model.
model = Sequential()

model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

## Defining the optimiser and loss function.
model.compile(optimizer='adam',loss='mse')

## Training the model.
model.fit(x=X_train, y=Y_train,
          validation_data=(X_test,Y_test),
          bathh_size=128,epochs=400)

