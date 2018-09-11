# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
from skimage.io import imread
import pylab as plt
from matplotlib import cm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# In[ ]:


os.chdir('C:/Users/Moi/HIRES/csv')

# In[ ]:


df = pd.read_csv('fusion_20180902.csv')
nb_labels = len( set(df.latex)  )
nb_pixels = 32*32

# In[ ]:


os.chdir('C:/Users/Moi/HIRES/datasets')

# In[ ]:


df_train = pd.DataFrame({'path':[],'latex':[]})
df_test = pd.DataFrame({'path':[],'latex':[]})

borne=100

for i in set(df.latex):
    nb_images = df[df.latex==i].shape[0]
    if nb_images < 2*borne+1:
        lignes = np.random.choice(df[df.latex == i].index, nb_images,replace=False)
        lignes_train = lignes[: nb_images//2]
        lignes_test = lignes[nb_images//2 :]
        df_train = pd.concat([df_train, df.iloc[lignes_train]])
        df_test = pd.concat([df_test, df.iloc[lignes_test]])
    else:
        lignes = np.random.choice(df[df.latex == i].index, 2*borne ,replace=False)
        lignes_train = lignes[:borne]
        lignes_test = lignes[borne:]
        df_train = pd.concat([df_train, df.iloc[lignes_train]])
        df_test = pd.concat([df_test, df.iloc[lignes_test]])

# In[ ]:


nb_images_train = df_train.shape[0]
nb_images_test = df_test.shape[0]
df_train.index = range(nb_images_train)
df_test.index = range(nb_images_test)

# In[ ]:


data_train = np.zeros([nb_images_train,1,32,32])
for i in df_train.index:
    if len(plt.imread(df_train['path'][i]).shape)==3:
        data_train[i,0,:,:]=plt.imread(df_train['path'][i])[:,:,0]
    else:
        data_train[i,0,:,:]=plt.imread(df_train['path'][i])[:,:]

# In[ ]:


data_test = np.zeros([nb_images_test,1,32,32])
for i in df_test.index:
    if len(plt.imread(df_test['path'][i]).shape)==3:
        data_test[i,0,:,:]=plt.imread(df_test['path'][i])[:,:,0]
    else:
        data_test[i,0,:,:]=plt.imread(df_test['path'][i])[:,:]

# In[ ]:


data_train = data_train.reshape(nb_images_train,nb_pixels)
target_train = pd.get_dummies(df_train.latex)
data_test = data_test.reshape(nb_images_test,nb_pixels)
target_test = pd.get_dummies(df_test.latex)

# In[ ]:


model = Sequential()
model.add( Dense( units = nb_pixels , input_dim = nb_pixels , kernel_initializer = 'normal' , activation='relu' ) )
model.add( Dropout(0.3) )
model.add( Dense( units = nb_pixels , input_dim = nb_pixels , kernel_initializer = 'normal' , activation='relu' ) )
model.add( Dense( units = nb_labels , kernel_initializer = 'normal' , activation='softmax'   )  )
model.compile( loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy']  )

# In[ ]:


model.fit(data_train, target_train, epochs = 25, batch_size = 200, verbose = 2)
pred = model.predict(data_test)
scores = model.evaluate(data_test, target_test, verbose=0)
print("Perte: %.2f Erreur: %.2f%%" % (scores[0], 100-scores[1]*100))
