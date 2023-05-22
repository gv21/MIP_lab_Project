import numpy as np
import pandas as pd
import pingouin as pg
import pickle
from itertools import product
import time
from multiprocessing import Pool
import multiprocessing
import json
from scipy import signal

#taken without change from Peter's code
# Function to transform similarity matrices within each dictionary key to arrays
def transform_rdm(dictionary_data):
    arrays = {}

    # Create boolean matrix and only retain upper triangle. Uses first key, so assumes all keys have same size
    mask = np.ones(dictionary_data[list(dictionary_data.keys())[0]].shape, dtype='bool')
    #print(f"1{mask}")
    mask[np.tril_indices(len(dictionary_data[list(dictionary_data.keys())[0]]))] = False
    #print(f"2{mask}")
    for dict_idx, data_type in enumerate(list(dictionary_data.keys())):
        arrays[data_type] = dictionary_data[data_type].values[mask]  # Convert dataframe to array and mask
    return arrays

def null_dist(paired_data, perms=500):
    #make a copy of the data
    shuffle_data = paired_data
    #shuffle fMRI data rows independently
    size= len(shuffle_data['behavior'])
    rng = np.random.default_rng()
    r = np.tile(rng.choice(size, size=size, replace=False), (perms,1))
    r = rng.permuted(r, axis=1)
    fMRI_shuffled = np.take(np.array(shuffle_data['fMRI']), r, axis=0)
    #repeat behvaior data for each row of fMRI data
    behavior_array= np.tile(np.array(shuffle_data['behavior']),(perms,1))
    arr=np.concatenate((fMRI_shuffled, behavior_array), axis=1)
    #compute the correlation for each pair fMRI and behavior
    shuffled_corr = np.apply_along_axis(lambda x: pg.corr(x=x[:size], y=x[size:], method='spearman')["r"][0], axis=1, arr=np.concatenate((fMRI_shuffled, behavior_array), axis=1))
    return shuffled_corr

"""# Function to generate null distribution for suspense ratings
def null_dist_pearson(x, y, perms=500): #perms = je ne sais pas trop quoi, à quoi correspond la valeur par défaut?
    rng = np.random.default_rng()
    shuffled_corr = np.full(perms, np.nan)  # Empty array for null distribution

    for perm_idx in range(0, perms):
        # Shuffle fMRI data
        sublist_a = rng.choice(len(x), size=len(x), replace=False) #generate len(x) number from 0 to len(x)
        x_new = np.array([x])[0][sublist_a]

        # Run correlation
        shuffled_corr[perm_idx] = pg.corr(x_new, y, method="pearson")["r"][0]

    return shuffled_corr"""
#Select the region of interest
region_interest = 'Hippocampus'
l_r_region_interest = ["Left Hippocampus", "Right Hippocampus"]

print('Load data')
# Load  data
pkl_file = open('Computation/Pairwise/'+region_interest+'/'+region_interest+'_Pairwise_Data_all_movie.pkl', "rb")
pairwise_data = pickle.load(pkl_file)
behavioral_arrays = transform_rdm(pairwise_data["behavior"])  # Transform behavioral/covariate data into arrays
fMRI_arrays = transform_rdm(pairwise_data["fMRI"])
pkl_file.close()

"""ATTENTION LE VRAI NOMBRE DE PERMUTATION EST 10 000"""
num_perms = 500 ###nbr of permutation to build the distribution 

def computation_movie_wise(behavior, connection):
    behavioral_data = behavioral_arrays[behavior]

    pairwise_arrays = np.vstack((behavioral_data, fMRI_arrays[connection])).transpose()
    pairwise_dataframe = pd.DataFrame(data=pairwise_arrays, columns=['behavior', 'fMRI'], dtype='float32')
    pairwise_dataframe = pairwise_dataframe.dropna().reset_index(drop=True)  
    if not ('TR-' in connection):  # if movie-wide analysis, use absolute values for all covariates/behavior  
        pairwise_dataframe.loc[:, pairwise_dataframe.columns != 'fMRI'] = \
            np.absolute(pairwise_dataframe.loc[:, pairwise_dataframe.columns != 'fMRI'])
    correlation = pg.corr(x=pairwise_dataframe['fMRI'], y=pairwise_dataframe['behavior'], method='spearman')

    null_distribution = null_dist(pairwise_dataframe, perms = num_perms)
    # Test Fisher transformed correlations. Absolute null z less then absolute observed z
    greater_less_sum = np.sum(np.absolute(np.arctanh(null_distribution)) <= np.absolute(np.arctanh(correlation.r[0])))  ###compte le nombre de fois que 
    ###la valeur que l'on veut tester est plus petite que la valeur significative
    permutation_significance = 1.0 - (greater_less_sum / num_perms)
 
    new_df = pd.DataFrame([[behavior, connection, correlation.r[0], correlation['p-val'][0],permutation_significance]],
                          columns = ["Behavior", "Connection", "r", "p", "perm_p"])
    
    return new_df


#Select only the region of interest 
print('Selection of region of interest')
ref = ['7Networks_LH_SalVentAttn_Med_1', '7Networks_RH_SalVentAttn_Med_1', '7Networks_LH_Cont_PFCl_5', '7Networks_RH_Cont_PFCl_9', '7Networks_LH_SomMot_4',
       '7Networks_RH_SomMot_6', 'Hippocampus', 'Amygdala']
ref.remove(region_interest) #avoid duplicate

fMRI_name_selected = []
for fMRI in (fMRI_arrays.keys()):
    str_fMRI = str(fMRI)
    if any(word in str_fMRI for word in ref):
        fMRI_name_selected.append(str_fMRI)

#Movie wise analysis
print('Movie wise')
connection_x_behavior = list(product(list(behavioral_arrays.keys()), fMRI_name_selected)) 
print(len(connection_x_behavior))
start_time = time.time()
nbr = multiprocessing.cpu_count()
with Pool (processes=nbr) as pool: #processes = number of thread (coeur) I use on my computer
    result = pool.starmap(computation_movie_wise, connection_x_behavior) 
statistical_tests = pd.concat((result), axis =0, ignore_index=True)
time_ = (time.time() - start_time)
heures = (time_ % (24 * 3600)) // 3600
minutes = (time_ % 3600) // 60
secondes = time_ % 60
print("Cela a pris {} heures, {} minutes, et {} secondes.".format(int(heures), int(minutes), round(secondes, 2)))

print('Save movie wise results')
pkl_file = open('Computation/Group/'+region_interest + '_all_' + region_interest+'_Movie_wise_group_analysis_500_perms.pkl', "wb")
pickle.dump(statistical_tests, pkl_file)
pkl_file.close()
statistical_tests.to_csv('Computation/Group/'+region_interest + '_all_' + region_interest+'_Movie_wise_group_analysis_500_perms.csv', index=False)


