import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys



keras.utils.set_random_seed(42)
sys.stdout = open("../outputs/nn.txt", "w")



def plot_loss_curves(history):
    plt.clf()
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1,len(loss_values)+1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_accuracy_curves(history):
  plt.clf()
  history_dict = history.history
  acc = history_dict["accuracy"]
  val_acc = history_dict["val_accuracy"]
  epochs = range(1, len(acc) + 1)
  plt.plot(epochs, acc, "bo", label="Training acc")
  plt.plot(epochs, val_acc, "b", label="Validation acc")
  plt.title("Training and validation accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()



df = pd.read_csv("../DL_DataFrame.csv")
features = ['density','plasma_current','elongation','minor_radius','toroidal_B_field','triangularity']
print(f"Training on {len(features)} features")

X = df[features].values
y = df['density_limit_phase'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled  = np.expand_dims(X_train_scaled , -1)
X_test_scaled = np.expand_dims(X_test_scaled, -1)

input = keras.Input(shape=(6,1))

x = keras.layers.Conv1D(32,                         #layer 1                                
                        kernel_size=(2, 2),    
                        activation="relu",     
                        name="Conv_1")(input)
x = keras.layers.MaxPool2D()(x)

x = keras.layers.Conv2D(32,                         #layer 2                                
                        kernel_size=(2, 2),    
                        activation="relu",     
                        name="Conv_2")(input)
x = keras.layers.MaxPool2D()(x)

x = keras.layers.Flatten()(x) #flatten
x = keras.layers.Dense(256, activation="relu")(x) #fully-connected dense ReLu
output = keras.layers.Dense(10, activation="softmax")(x) 
model = keras.Model(input, output)
print(model.summary())
keras.utils.plot_model(model, show_shapes=True)