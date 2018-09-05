
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy, imageio
import keras
from keras.utils import np_utils
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils 

from sklearn.metrics import confusion_matrix

from matplotlib import cm
get_ipython().run_line_magic('matplotlib', 'inline')

import itertools
from sklearn.model_selection import train_test_split
df=pd.read_csv('fusion_20180902.csv')


# In[ ]:


df_sample=pd.DataFrame.copy(df)
Ind=df.groupby('latex').count().index
for i in Ind:
    N=df.groupby('latex').size()[i]
    if (N>160):
        L=np.random.choice(df_sample[df_sample['latex']==i].index,N-int(np.log(N/160+1)/np.log(2)*160),replace=False)
        df_sample=df_sample.drop(L,axis=0)


# In[ ]:


df_sample.index=range(df_sample.shape[0])
data_test=np.zeros([df_sample.shape[0],1,32,32])
for i in range(df_sample.shape[0]):
    if len(plt.imread(df_sample['path'][i]).shape)==3:
        data_test[i,0,:,:]=plt.imread(df_sample['path'][i])[:,:,0]
    else:
        data_test[i,0,:,:]=plt.imread(df_sample['path'][i])[:,:]


# In[ ]:


target_test=df_sample['latex']


# In[ ]:


Index_train=[]
for i in Ind:
    Index_train.extend(random.sample(list(df_sample[df_sample['latex']==i].index),int(0.8*df_sample[df_sample['latex']==i].shape[0])))
Index_test=list(df_sample.drop(Index_train,axis=0).index)


# In[ ]:


X_train=data_test[Index_train,:,:,:]
X_test=data_test[Index_test,:,:,:]
y_train=target_test[Index_train]
y_test=target_test[Index_test]


# In[ ]:


y_train_2=pd.get_dummies(y_train)
y_test_2=pd.get_dummies(y_test)


# In[ ]:


Filters1=[30]
Filters2=[15]
Dropo=[0.25]
Dense1=[500]
Dense2=[300]

nb_epoch1=[12]
batch_si=[200]
d_score_5=dict()
for nb_f1,nb_f2,dropo,dense1,dense2,nb_ep,batch_si in itertools.product(Filters,Filters2,Dropo,Dense1,Dense2,nb_epoch1,batch_si):
    model=Sequential()
    model.add(Conv2D(nb_filter=nb_f1, nb_row=5, nb_col=5, border_mode="valid", input_shape=(1,32,32),activation='relu',data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(nb_filter=nb_f2, nb_row=3, nb_col=3, border_mode='valid', input_shape=(1, 32, 32), activation='relu', data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropo))
    model.add(Flatten())
    model.add(Dense(dense1,activation='relu'))
    model.add(Dense(dense2,activation='relu'))
    
    model.add(Dense(df.groupby('latex').count().sort_values('path').shape[0],activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train_2, nb_epoch = nb_ep, batch_size = batch_si, verbose = 2)
    pred=model.predict(X_test)
    scores = model.evaluate(X_test, y_test_2, verbose=0)
    print("Perte: %.2f Erreur: %.2f%%" % (scores[0], 100-scores[1]*100))
    d_score_5[(nb_f,dropo,dense1,dense2,nb_ep,batch_si)]=scores[1]*100

