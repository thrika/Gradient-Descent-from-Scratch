#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[ ]:


function = lambda x: np.sin(10*(math.pi)*x) + ((x-1)**4)


# In[ ]:


x = np.linspace(-0.5,2.5,500)


# In[ ]:


plt.plot(x, function(x))


# In[ ]:


def deriv(x):
    return (((5*math.pi)*(np.cos(10*math.pi*x)))/x) - ((math.sin(10*math.pi*x))/(2*(x**2))) + (4*((x-1)**3))


# In[ ]:


def step_function(x_new, x_prev, precision, learning_rate):
    x_list, y_list = [x_new], [function(x_new)]
    
    while abs(x_new - x_prev) > precision:
        x_prev = x_new
        deriv_x = - deriv(x_prev)
        x_new = x_prev + (learning_rate * deriv_x)
        x_list.append(x_new)
        y_list.append(function(x_new))

    print ("Minimum occurs at: "+ str(x_new))
    print ("Number of steps: " + str(len(x_list)))
    
    plt.subplot(1,2,2)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    plt.plot(x,function(x), c="r")
    plt.title("Gradient descent")
    plt.show()


# In[ ]:


step_function(2.4, 0, 0.001, 0.001)


# In[ ]:


step_function(2.4, 0, 0.000000001, 0.000000001)


# In[ ]:


step_function(1.8, 0, 0.000000001, 0.000000001)


# In[ ]:


step_function(1.5, 0, 0.000000001, 0.000000001)


# In[ ]:


step_function(0.5, 0, 0.000000001, 0.000000001)


# In[ ]:




