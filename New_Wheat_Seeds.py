#!/usr/bin/env python
# coding: utf-8

# In[9]:


from pycaret.datasets import get_data
dataset = get_data('Wheat-Seeds')


# In[2]:


import pandas as pd
dataset = pd.read_csv(r'C:\Users\DELL\Desktop\seeds_data.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.shape


# In[5]:


data = dataset.sample(frac= 0.95, random_state = 786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace = True, drop = True)
print('Date for modeling ' + str(data.shape))
print('Unseen data for for Predictions: ' + str (data_unseen.shape))


# In[7]:


from pycaret.classification import *


# In[ ]:


exp_clf101 = setup(data= data, target = 'Class', session_id = 123)


# In[ ]:


# Comparing all models
best = compare_models()


# In[ ]:


# create a model and decision tree classifier
dt = creat_model('dt')


# In[ ]:


#trained model object is stored in the variable 'dt'
print(dt)


# In[ ]:


# Neighbors Classifier
knn = create_model('knn')


# In[ ]:


#Logistic Regression
lr = create_model('lr')


# In[ ]:


#Tune a Model and Decision Tree Classifier
tuned_dt = tune_model(dt)


# In[ ]:


#tuned model object is stored in the variable 'tuned_dt'
print(tuned_dt)


# In[ ]:


#Decision Tree Classifier
import numpy as np
tuned_knn = tune_model(knn, custom_grid = {'n_neighbors' : np.arange(0,50,1)})


# In[ ]:


#Logistic Regression
tuned_lr = tune_model(lr)


# In[ ]:


#Plot a Model
#confusion matrix
plot_model(tuned_knn, plot = 'confusion_matrix')


# In[ ]:


#Classification Report
plot_model(tuned_knn, plot = 'class_report')


# In[ ]:


#Decision Boundary Plot
plot_model(tuned_knn, plot= 'boundary')


# In[ ]:


#Prediction Error Plot
plot_model(tuned_knn, plot = 'error')


# In[ ]:


evaluate_model(tuned_knn);


# In[ ]:


#Predict on test / hold-out Sample
predict_model(tuned_knn);


# In[ ]:


#Finalize Model for Deployment
final_knn= finalize_model(tuned_knn)


# In[ ]:


#Final K Nearest Neighbour parameters for deployments
print(final_knn)


# In[ ]:


#Predict on unseen data
unseen_predictions= predict_model(final_knn, data = data_unseen)
unseen_predictions.head()


# In[ ]:


#Saving the model
save_model(final_knn, 'Final KNN Model Feb 13 2023')


# In[ ]:


#Loading the saved model
saved_final_knn = load_model('Final KNN Model Feb 13 2023')


# In[ ]:


#getting the new prediction models
new_prediction = predict_model(saved_final_knn, data = data_unseen)


# In[ ]:


new_prediction.head()

