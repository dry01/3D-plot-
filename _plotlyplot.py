#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pydicom


# In[1]:


pip install plotly


# In[1]:


pip install pynrrd


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import nrrd

import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
init_notebook_mode(connected=True) 


# In[14]:


cd ..


# In[15]:


ls


# In[17]:


data_path = "Training 2/"
output_path = working_path = "Training 2/"
g = glob(data_path + '/*.nrrd')

# Print out the first 5 file names to verify we're in the right folder.
print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
print ('\n'.join(g[:5]))


# In[18]:


def return_nrrd(file_path):
    # Read all NRRD (annotation) files within a directory
    out_nrrd = {}
    for dirName, subdirList, fileList in os.walk(file_path):
        for filename in fileList:
            if ".nrrd" in filename.lower():
                name = filename.split('_')[0] 
                name = name.split('.')[0] # Get annotation name and store in dictionary
                out_nrrd[name] = os.path.join(dirName,filename)
    return out_nrrd


# In[19]:


def get_dataset(anns_dir):
    
    data_out = []
    
    d_nrrd = return_nrrd(data_path)
    for i in d_nrrd:
        seg, opts = nrrd.read(d_nrrd[i])
        
        
            
        # Saves data
        data_out.append((seg))
    return data_out


# In[20]:


train = get_dataset (data_path)


# In[59]:


print(train[35].shape)
print(type(train[3]))
imgs_to_process = train[3]


# In[ ]:


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

print "Shape before resampling\t", imgs_to_process.shape
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print "Shape after resampling\t", imgs_after_resamp.shape


# In[48]:


def make_mesh(image, step_size=1):

    #print ("Transposing surface")
    #p = image.transpose(2,1,0)
    
    print ("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes_lewiner(image, step_size=step_size, allow_degenerate=True) 
    return verts, faces


# In[23]:


pip install chart-studio


# In[24]:


import chart_studio.plotly as py
import plotly.figure_factory as FF


# In[49]:


import plotly
v, f = make_mesh(imgs_to_process, 2)


# In[50]:


def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    print ("Drawing")
    
    
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = FF.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    iplot(fig)


# In[51]:


plotly_3d(v, f)


# In[ ]:




