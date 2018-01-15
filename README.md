
# Artificial Intelligence Nanodegree

## Convolutional Neural Networks

---

In this notebook, we train an MLP to classify images from the MNIST database.

### 1. Load MNIST Database


```python
from keras.datasets import mnist

# use Keras to import pre-shuffled MNIST database
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("The MNIST database has a training set of %d examples." % len(X_train))
print("The MNIST database has a test set of %d examples." % len(X_test))
```

    The MNIST database has a training set of 60000 examples.
    The MNIST database has a test set of 10000 examples.
    

### 2. Visualize the First Six Training Images


```python
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.cm as cm
import numpy as np

# plot first six training images
fig = plt.figure(figsize=(20,20))
for i in range(6):
    ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(str(y_train[i]))
```


![png](output_3_0.png)


### 3. View an Image in More Detail


```python
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
visualize_input(X_train[0], ax)
```


![png](output_5_0.png)


### 4. Rescale the Images by Dividing Every Pixel in Every Image by 255


```python
# rescale [0,255] --> [0,1]
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255 
```

### 5. Encode Categorical Integer Labels Using a One-Hot Scheme


```python
from keras.utils import np_utils

# print first ten (integer-valued) training labels
print('Integer-valued labels:')
print(y_train[:10])

# one-hot encode the labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# print first ten (one-hot) training labels
print('One-hot labels:')
print(y_train[:10])
```

    Integer-valued labels:
    [5 0 4 1 9 2 1 3 1 4]
    One-hot labels:
    [[ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
     [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
     [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
     [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]]
    

### 6. Define the Model Architecture


```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# define the model
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# summarize the model
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_2 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 512)               401920    
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 512)               262656    
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 669,706
    Trainable params: 669,706
    Non-trainable params: 0
    _________________________________________________________________
    

### 7. Compile the Model


```python
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])
```

### 8. Calculate the Classification Accuracy on the Test Set (Before Training)


```python
# evaluate test accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)
```

    Test accuracy: 10.4700%
    

### 9. Train the Model


```python
from keras.callbacks import ModelCheckpoint   

# train the model
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', 
                               verbose=1, save_best_only=True)
hist = model.fit(X_train, y_train, batch_size=128, epochs=10,
          validation_split=0.2, callbacks=[checkpointer],
          verbose=1, shuffle=True)
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    47744/48000 [============================>.] - ETA: 0s - loss: 0.2862 - acc: 0.9113Epoch 00001: val_loss improved from inf to 0.13708, saving model to mnist.model.best.hdf5
    48000/48000 [==============================] - 12s 248us/step - loss: 0.2856 - acc: 0.9116 - val_loss: 0.1371 - val_acc: 0.9577
    Epoch 2/10
    47872/48000 [============================>.] - ETA: 0s - loss: 0.1159 - acc: 0.9643Epoch 00002: val_loss improved from 0.13708 to 0.10251, saving model to mnist.model.best.hdf5
    48000/48000 [==============================] - 11s 222us/step - loss: 0.1160 - acc: 0.9643 - val_loss: 0.1025 - val_acc: 0.9702
    Epoch 3/10
    47744/48000 [============================>.] - ETA: 0s - loss: 0.0820 - acc: 0.9751Epoch 00003: val_loss improved from 0.10251 to 0.09112, saving model to mnist.model.best.hdf5
    48000/48000 [==============================] - 11s 230us/step - loss: 0.0819 - acc: 0.9751 - val_loss: 0.0911 - val_acc: 0.9748
    Epoch 4/10
    47744/48000 [============================>.] - ETA: 0s - loss: 0.0641 - acc: 0.9809Epoch 00004: val_loss improved from 0.09112 to 0.08469, saving model to mnist.model.best.hdf5
    48000/48000 [==============================] - 11s 229us/step - loss: 0.0639 - acc: 0.9810 - val_loss: 0.0847 - val_acc: 0.9778
    Epoch 5/10
    47872/48000 [============================>.] - ETA: 0s - loss: 0.0523 - acc: 0.9843Epoch 00005: val_loss did not improve
    48000/48000 [==============================] - 11s 237us/step - loss: 0.0522 - acc: 0.9844 - val_loss: 0.0875 - val_acc: 0.9780
    Epoch 6/10
    47744/48000 [============================>.] - ETA: 0s - loss: 0.0440 - acc: 0.9866Epoch 00006: val_loss did not improve
    48000/48000 [==============================] - 12s 240us/step - loss: 0.0441 - acc: 0.9866 - val_loss: 0.1056 - val_acc: 0.9764
    Epoch 7/10
    47872/48000 [============================>.] - ETA: 0s - loss: 0.0384 - acc: 0.9882Epoch 00007: val_loss did not improve
    48000/48000 [==============================] - 11s 236us/step - loss: 0.0383 - acc: 0.9882 - val_loss: 0.0988 - val_acc: 0.9773
    Epoch 8/10
    47744/48000 [============================>.] - ETA: 0s - loss: 0.0353 - acc: 0.9890Epoch 00008: val_loss did not improve
    48000/48000 [==============================] - 11s 238us/step - loss: 0.0354 - acc: 0.9890 - val_loss: 0.0950 - val_acc: 0.9812
    Epoch 9/10
    47744/48000 [============================>.] - ETA: 0s - loss: 0.0296 - acc: 0.9907Epoch 00009: val_loss did not improve
    48000/48000 [==============================] - 12s 242us/step - loss: 0.0295 - acc: 0.9908 - val_loss: 0.1026 - val_acc: 0.9797
    Epoch 10/10
    47872/48000 [============================>.] - ETA: 0s - loss: 0.0275 - acc: 0.9915Epoch 00010: val_loss did not improve
    48000/48000 [==============================] - 11s 232us/step - loss: 0.0275 - acc: 0.9915 - val_loss: 0.1113 - val_acc: 0.9792
    

### 10. Load the Model with the Best Classification Accuracy on the Validation Set


```python
# load the weights that yielded the best validation accuracy
model.load_weights('mnist.model.best.hdf5')
```

### 11. Calculate the Classification Accuracy on the Test Set


```python
# evaluate test accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)
```

    Test accuracy: 97.9500%
    
