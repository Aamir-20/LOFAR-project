# -*- coding: utf-8 -*-

import time


def main():
    t1 = time.time()
    
    ##Import the libraries.
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input,Dense
    
    ## Creating the layers.
    input_layer = Input(shape=(3,))
    Layer_1 = Dense(4, activation="relu")(input_layer)
    Layer_2 = Dense(4, activation="relu")(Layer_1)
    output_layer= Dense(1, activation="linear")(Layer_2)
    
    ## Defining the model by specifying the input and output layers.
    model = Model(inputs=input_layer, outputs=output_layer)
    
    ## Defining the optimiser and loss function.
    model.compile(optimizer='adam',
                  loss='mse')
    
    ## Training the model.
    X_train, X_test, Y_train, Y_test = data()
    model.fit(x=X_train, y=Y_train,
              validation_data=(X_test,Y_test),
              bathh_size=128,epochs=400)
    
    t2 = time.time()
    print("Runtime (seconds): ", t2-t1)
    
    
    
def data():
    t1 = time.time()
    
    import pandas as pd    
    
    # Read the CSV file.
    telescope_data = pd.read_csv("test_train_data.csv")
    
    # Splitting the data into an 80/20 split.
    split_1 = telescope_data[0:8000]
    split_2 = telescope_data[8000:10000]
    
    # Setting the training data.
    X_train = split_1[["id","phi_0","chi_0","P_0"]]
    X_test = split_1[["id","Q","U"]]
    
    # Setting the testing data.
    Y_train = split_2[["id","phi_0","chi_0","P_0"]]
    Y_test = split_2[["id","Q","U"]]
    
    # print(X_train)
    # print(Y_train)
    # print(X_test)
    # print(Y_test)
    
    t2 = time.time()
    print("Runtime (seconds): ", t2-t1)
    
    return X_train, X_test, Y_train, Y_test
    
    
    
if __name__ == "__main__":
    data()
