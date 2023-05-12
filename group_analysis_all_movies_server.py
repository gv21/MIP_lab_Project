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

# Function to generate null distribution
def null_dist(paired_data, perms=500):
    #print('rng')
    rng = np.random.default_rng()
    #print('shuffle')
    shuffle_data = paired_data  # Copy to new dataframe as some functions may apply directly to original data
    #print('max')
    num_sub_comparisons = np.max(shuffle_data.index.values)  # Base comparisons off data as this varies (dropped na)
    #print('shuffled_corr')
    shuffled_corr = np.full(perms, np.nan)  # Empty array for null distribution

    for perm_idx in range(0, perms):
        #print(perm_idx)
        # Shuffle fMRI data
        sublist_a = rng.choice(num_sub_comparisons, size=num_sub_comparisons, replace=False)
        shuffle_data['fMRI'] = shuffle_data['fMRI'].iloc[sublist_a].reset_index(drop=True)

        """shuffled_corr[perm_idx] = pg.partial_corr(data=shuffle_data, x="fMRI", y="behavior",
                                                  covar=["age", "sex", "movement"], method='spearman')["r"][0]"""
        # Run correlation #He used pg.partial_corr, as I don't have covariate I will only use pg.corr
        shuffled_corr[perm_idx] = pg.corr(x=shuffle_data['fMRI'], y=shuffle_data['behavior'],
                                                  method='spearman')["r"][0] #x, y = names of column in data; covar = name of covariate in data ->remove it

    return shuffled_corr

# Function to generate null distribution for suspense ratings
def null_dist_pearson(x, y, perms=500): #perms = je ne sais pas trop quoi, à quoi correspond la valeur par défaut?
    rng = np.random.default_rng()
    shuffled_corr = np.full(perms, np.nan)  # Empty array for null distribution

    for perm_idx in range(0, perms):
        # Shuffle fMRI data
        sublist_a = rng.choice(len(x), size=len(x), replace=False) #generate len(x) number from 0 to len(x)
        x_new = np.array([x])[0][sublist_a]

        # Run correlation
        shuffled_corr[perm_idx] = pg.corr(x_new, y, method="pearson")["r"][0]

    return shuffled_corr

print('Load data')
# Load  data
pkl_file = open("Computation/Pairwise/Pairwise_Data_all_movie.pkl", "rb")
pairwise_data = pickle.load(pkl_file)
behavioral_arrays = transform_rdm(pairwise_data["behavior"])  # Transform behavioral/covariate data into arrays
fMRI_arrays = transform_rdm(pairwise_data["fMRI"])
pkl_file.close()


def computation_movie_wise(behavior, connection):
    behavioral_data = behavioral_arrays[behavior]

    pairwise_arrays = np.vstack((behavioral_data, fMRI_arrays[connection])).transpose()
    pairwise_dataframe = pd.DataFrame(data=pairwise_arrays, columns=['behavior', 'fMRI'], dtype='float32')
    pairwise_dataframe = pairwise_dataframe.dropna().reset_index(drop=True)  
    if not ('TR-' in connection):  # if movie-wide analysis, use absolute values for all covariates/behavior  
        pairwise_dataframe.loc[:, pairwise_dataframe.columns != 'fMRI'] = \
            np.absolute(pairwise_dataframe.loc[:, pairwise_dataframe.columns != 'fMRI'])
    correlation = pg.corr(x=pairwise_dataframe['fMRI'], y=pairwise_dataframe['behavior'], method='spearman')

    null_distribution = null_dist(pairwise_dataframe)
    # Test Fisher transformed correlations. Absolute null z less then absolute observed z
    greater_less_sum = np.sum(np.absolute(np.arctanh(null_distribution)) <= np.absolute(np.arctanh(correlation.r[0])))  ###compte le nombre de fois que 
    ###la valeur que l'on veut tester est plus petite que la valeur significative
    permutation_significance = 1.0 - (greater_less_sum / num_perms)
 
    new_df = pd.DataFrame([[behavior, connection, correlation.r[0], correlation['p-val'][0],permutation_significance]],
                          columns = ["Behavior", "Connection", "r", "p", "perm_p"])
    
    return new_df


# Initialize variables
subjects = len(pairwise_data['fMRI']['Left Amygdala_x_7Networks_LH_SalVentAttn_Med_1']) #take on brain region combination at random
num_perms = 10000 ###nbr of permutation to build the distribution
#statistical_tests = pd.DataFrame(columns=["Behavior", "Connection", "r", "p", "perm_p"])
suspense_tests = {}


#Select only the region of interest 
print('Selection of region of interest')
ref = ['7Networks_LH_SalVentAttn_Med_1', '7Networks_RH_SalVentAttn_Med_1', '7Networks_LH_Cont_PFCl_5', '7Networks_RH_Cont_PFCl_9', '7Networks_LH_SomMot_4',
       '7Networks_RH_SomMot_6', 'Hippocampus']
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
with Pool (processes=90) as pool: #processes = number of thread (coeur) I use on my computer
    result = pool.starmap(computation_movie_wise, connection_x_behavior[:176928]) #10% des données que j'ai
statistical_tests = pd.concat((result), axis =0, ignore_index=True)
time_ = (time.time() - start_time)
heures = (time_ % (24 * 3600)) // 3600
minutes = (time_ % 3600) // 60
secondes = time_ % 60
print("Cela a pris {} heures, {} minutes, et {} secondes.".format(int(heures), int(minutes), round(secondes, 2)))

print('Save movie wise results')
pkl_file = open("Computation/Group/amygd_Movie_wise_group_analysis.pkl", "wb")
pickle.dump(statistical_tests, pkl_file)
pkl_file.close()
statistical_tests.to_csv('amygd_Movie_wise_group_analysis.csv', index=False)

#Au cas où la deuxième partie se passe mal
"""pkl_file = open("Computation/Group/Movie_wise_group_analysis.pkl", "rb")
statistical_tests = pickle.load(pkl_file)
pkl_file.close()"""

"""print('TR wise analysis')
with open('Data/Annot_Sintel_stim.json', 'r') as f:
    # Lecture du contenu du fichier et chargement des données JSON en tant que dictionnaire
    annot = json.load(f)
index_anx = annot['Columns'].index('Anxiety')

ratings = np.genfromtxt('Data/Annot_Sintel_stim.tsv', delimiter='\t')
anx_rating = ratings[:,index_anx]

#Ressample
resampling_len = len(statistical_tests["r"])
anx_rating_ressampled= signal.resample(anx_rating, resampling_len)

r_ref_l = 'Left Amygdala'+'_x_'
r_ref_r = 'Right Amygdala'+'_x_'
r_interest_TR = [r_ref_l+'7Networks_LH_SalVentAttn_Med_1', r_ref_l+'7Networks_RH_SalVentAttn_Med_1',
                r_ref_l+'7Networks_LH_Cont_PFCl_5', r_ref_l+'7Networks_RH_Cont_PFCl_9',
                r_ref_l+'7Networks_LH_SomMot_4', r_ref_l+'7Networks_RH_SomMot_6',
                r_ref_l+'Left Hippocampus', r_ref_l+'Right Hippocampus',
                r_ref_r+'7Networks_LH_SalVentAttn_Med_1', r_ref_r+'7Networks_RH_SalVentAttn_Med_1',
                r_ref_r+'7Networks_LH_Cont_PFCl_5', r_ref_r+'7Networks_RH_Cont_PFCl_9',
                r_ref_r+'7Networks_LH_SomMot_4', r_ref_r+'7Networks_RH_SomMot_6',
                r_ref_r+'Left Hippocampus', r_ref_r+'Right Hippocampus']

behavior_data = pd.read_excel('ordered_by_sub_behavior_score.xlsx')
start_column_name = "cov_total"  # the name of the column you want to start from
#take all the behavior data from cov_total to the end = behvior of interest
name_behavior_interest = list(behavior_data.columns[behavior_data.columns.get_loc(start_column_name):])

for idx_behavior, behavior in enumerate (name_behavior_interest):
    print(f"Iteration (behavior) {idx_behavior} of {len(name_behavior_interest)}")
    for idx_region, region in enumerate(r_interest_TR):
        print(f"Iteration (region) {idx_region} of {len(r_interest_TR)}")
        amygdala = statistical_tests["Connection"].str.contains("TR-") & \
            statistical_tests["Connection"].str.contains(str(region)) & \
            statistical_tests["Behavior"].eq(str(behavior))
    
        #For amygdala
        x = np.array(statistical_tests["r"][amygdala])
        y = anx_rating_ressampled

        amyg_sr_suspense = pg.corr(x,y, method="pearson")

        null_distribution = null_dist_pearson(x, y, num_perms)
        greater_less_sum = np.sum(np.absolute(np.arctanh(null_distribution)) <= np.absolute(np.arctanh(amyg_sr_suspense.r[0])))
        amyg_sr_suspense_sig = 1.0 - (greater_less_sum / num_perms)
        
        #store it
        idx_max = len(connection_x_behavior)+1
        len_behavior = len(name_behavior_interest)
        len_region_interest = len(r_interest_TR)
        statistical_tests.loc[idx_max + len_region_interest*idx_behavior +idx_region]=[str(behavior), region, amyg_sr_suspense.r[0],
                                                amyg_sr_suspense['p-val'][0], amyg_sr_suspense_sig]

print('Save Group results')
pkl_file = open("Computation/Group/Group_Results.pkl", "wb")
pickle.dump(statistical_tests, pkl_file)
pkl_file.close()

statistical_tests.to_csv("Computation/Group/Group_Results.csv", index=False, float_format='%.7f')"""

