#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


function = lambda x: x**2


# In[4]:


x = np.linspace(-5,5,100)


# In[5]:


plt.plot(x, function(x))


# In[6]:


def deriv(x):
    return 2*x


# In[9]:


def step_function(x_new, x_prev, precision, learning_rate):
    x_list, y_list = [x_new], [function(x_new)]
    
    while abs(x_new - x_prev) > precision:
        x_prev = x_new
        deriv_x = - deriv(x_prev)
        x_new = x_prev + (learning_rate * deriv_x)
        x_list.append(x_new)
        y_list.append(function(x_new))

    print ("Minimum occurs at: "+ str(x_new))
    print ("Number of iterations: " + str(len(x_list)))
    
    plt.subplot(1,2,2)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    plt.plot(x,function(x), c="r")
    plt.title("Gradient descent")
    plt.show()


# In[10]:


step_function(4.5, 0, 0.01, 0.05)


# In[11]:


step_function(4.5, 0, 0.00001, 0.00005)


# In[12]:


step_function(-4, 0, 0.001, 0.005)


# In[ ]:




