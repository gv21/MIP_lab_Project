import os
import numpy as np
import re
from nltools.stats import isc


def split_path_to_data(film, name_comparison="TC_14"):
    """Load data and separate it in 2
        return an array of all subject with their datapoints for all brain areas """
    file_name = os.listdir('data/'+film)
    all_path_to_14=[]
    all_path_to_400=[]
    for i in range(len(file_name)):
        path=(os.path.join('data/'+film, file_name[i]))
        condition=re.search(name_comparison,path)
        if (condition == None):
            all_path_to_400.append(path)
        else:
            all_path_to_14.append(path)
    return all_path_to_14, all_path_to_400

def load_data(name_comparison="TC_14", film='Rest', TR=460 ): #mettre une taille définie par TR par film (faudra stocker une liste de film et de taille correspondante à un moment)
    path14, path400 = split_path_to_data(film, name_comparison=name_comparison)
    data_all_subjects=[]
    for n in range (len(path14)):
        data14=np.genfromtxt(path14[n], delimiter=",")
        data400=np.genfromtxt(path400[n], delimiter=",")
        data_one_subject=np.concatenate([data400[:TR], data14[:TR]], axis=1)
        data_all_subjects.append(data_one_subject)
    return data_all_subjects





### Windows function, not tested on Linux yet
def compute_ISC(all_subjects_film, min_len):
    """Compute Pairwise Intersubject Correlations """
    #transform the data in a rectangular (same shape) array
    all=[]
    for i in range (len(all_subjects_film)):
        all.append(all_subjects_film[i][:min_len][:])

    
    tensor_subjects_film= tf.convert_to_tensor(all)
    #swap axes to be able to loop over brain regions
    tensor_subjects_film2=np.swapaxes(tensor_subjects_film,2,0)
    
    brain_region_ISC=[]
    for roi in range (400):
        brain_region_ISC.append(isc(tensor_subjects_film2[roi])['isc'])
    
    return brain_region_ISC

def compute_ISC_remove_noise(all_subjects_film, min_len):
    """Compute Pairwise Intersubject Correlations """
    #transform the data in a rectangular (same shape) array
    all=[]
    diff = min_len-min_len
    max_ = min_len - 75
    for i in range (len(all_subjects_film)):
        all.append(all_subjects_film[i][75:max_][:])
        
    tensor_subjects_film= tf.convert_to_tensor(all)
    #swap axes to be able to loop over brain regions
    tensor_subjects_film2=np.swapaxes(tensor_subjects_film,2,0)
    
    brain_region_ISC=[]
    for roi in range (400):
        brain_region_ISC.append(isc(tensor_subjects_film2[roi])['isc'])
    
    return brain_region_ISC