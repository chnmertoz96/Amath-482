#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Part 1
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm


# In[2]:


train_df = pd.read_csv('~/Downloads/fashionmnist/fashion-mnist_train.csv')
test_df =  pd.read_csv('~/Downloads/fashionmnist/fashion-mnist_test.csv')


# In[3]:


train_data = np.array(train_df, dtype='float32')
test_data = np.array(test_df, dtype='float32')

x_train = train_data[:, 1:] / 255
y_train = train_data[:, 0]

x_test = test_data[:, 1:] / 255
y_test = test_data[:, 0]


# In[5]:


x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train, test_size=0.08335, random_state=12345,
)


# In[6]:


im_rows = 28
im_cols = 28
batch_size = 512
im_shape = (im_rows, im_cols, 1)

x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *im_shape)

print('x_train shape: {}'.format(x_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('x_validate shape: {}'.format(x_validate.shape))


# In[7]:


model = Sequential([
    Flatten(),
    Dense(400, activation='relu'),
    Dense(50, activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='softmax')
])


# In[8]:


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)


# In[11]:


model_fit = model.fit(
    x_train, y_train, batch_size=batch_size,
    epochs=10, verbose=1,
    validation_data=(x_validate, y_validate),
)


# In[12]:


score = model.evaluate(x_test, y_test, verbose=0)

print('test loss: {:.4f}'.format(score[0]))
print(' test acc: {:.4f}'.format(score[1]))


# In[15]:


pd.DataFrame(model_fit.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# In[16]:


predict_y = model.predict_classes(x_train)


# In[23]:


conf = cm(y_train,predict_y)
print(conf)


# In[ ]:




