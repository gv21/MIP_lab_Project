from typing import Dict, Any
import pickle
import pingouin as pg
import numpy as np
from multiprocessing import Pool
from itertools import combinations, product
import pandas as pd 
import re



movies_list = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LessonLearned', 'Payload',
               'Sintel', 'Spaceman', 'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain']
TR_list_preprocessed = {'AfterTheRain':382, 'BetweenViewings':624, 'BigBuckBunny':376, 'Chatter':312, 'FirstBite':461, 
           'LessonLearned':513, 'Payload':776, 'Sintel':558, 'Spaceman':592, 'Superhero':793, 'TearsOfSteel':455, 
           'TheSecretNumber':605, 'ToClaireFromSonny':308, 'YouAgain':616}

#Gather data from different movies in once. We will have for each subject the timecorr for all movies one after the other
print('Gather data from different movies')
list_dataframe=[]
for sub in range (30):
    print(sub)
    list_dataframe_per_subject=[]
    for film in movies_list:
        film_='Computation/'+film+'_amygdala_dyn_connectivity_width_30.pkl'
        #film_ = 'data/Pairwise_Test/5_sub_' + film + '_amygdala_cortical_subcortical_dyn_connectivity_width_30.pkl'
        with open(film_, "rb") as file:
            sub_data = pickle.load(file)
        list_dataframe_per_subject.append(sub_data[sub]['fMRI']) #sub_data[sub]['fMRI'] is a Dataframe
    concat_data = pd.concat(list_dataframe_per_subject,axis=0, ignore_index=True)
    column_with_all_nan=concat_data.columns[concat_data.isna().all()]
    concat_data.drop(column_with_all_nan, axis=1, inplace=True)
    list_dataframe.append({'fMRI':concat_data})

with open('all_movies_within_subjects_analysis.pkl', 'wb') as f:
    pd.to_pickle(list_dataframe, f)


def blockwide_corr(sub_a_index, sub_b_index):
    #Load data
    with open("all_movies_within_subjects_analysis.pkl", "rb") as file:
        sub_data = pickle.load(file)
    
    #prepare dict
    dyn_conn_pairwise_corr: Dict[Any, Any] = {}
    for reg1_x_reg2 in list(sub_data[sub_a_index]['fMRI'].keys()):
        dyn_conn_pairwise_corr[reg1_x_reg2] = pg.corr(sub_data[sub_a_index]['fMRI'][reg1_x_reg2], sub_data[sub_b_index]['fMRI'][reg1_x_reg2],
                                                      method='pearson').r[0]
    return [sub_a_index, sub_b_index], dyn_conn_pairwise_corr 

# Function to output TR-wise differences in dynamic connectivity ###add region pair, ie Amygdala Right_x_dmPFC in the paper
def tr_diff(index_paire, region_pair): 
    #Load data
    with open("all_movies_within_subjects_analysis.pkl", "rb") as file:
        sub_data = pickle.load(file)
        file.close()
    #compute the difference between one region of interest (ie amygdala) and another brain region ()
    diff = np.subtract(sub_data[index_paire[0]]['fMRI'][region_pair], sub_data[index_paire[1]]['fMRI'][region_pair])
    return diff

### Initialize variables
#prepare a list of combination of all subject. Subject 1 as index 0, subject 2 index 1 and so on
sub_list = np.arange(0,29,1)
pairwise_subject_combination = list(combinations(sub_list, 2))

# Create dictionaries full of empty (nan) dataframes
pkl_file = open("all_movies_within_subjects_analysis.pkl", "rb")
prepped_labels = pickle.load(pkl_file)
connection_labels = list(prepped_labels[0]['fMRI'].keys())
pkl_file.close()

pairwise_fMRI = {key: pd.DataFrame(columns=sub_list, index=sub_list) for key in connection_labels}
TR_all_movies = 7371
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
print('Initialisation TR')
for r in r_interest_TR:
    for TR in range(1, TR_all_movies + 1):
            pairwise_fMRI[r+f"_TR-{TR}"] = pd.DataFrame(columns=sub_list, index=sub_list)

print('Blockwide computation')
with Pool (processes=90) as pool: #processes = number of thread (coeur) I use on my computer
    dyn_conn_pairwise_corr_array = pool.starmap(blockwide_corr, pairwise_subject_combination)
print('End Blockwide computation')

print('Assign dyn corr values')
for index_data_array in dyn_conn_pairwise_corr_array:
    pairwise_index = index_data_array[0]
    for connection in list(index_data_array[1].keys()):
        pairwise_fMRI[connection].loc[pairwise_index[0], pairwise_index[1]] = index_data_array[1][connection] 

#for sub_paire in pairwise_subject_combination:
print('TR diff')
for sub_paire in pairwise_subject_combination:
    for region_paire in r_interest_TR:
        TR_diff = tr_diff(sub_paire, region_paire)
        for TRindex in range (0, TR_all_movies):
            pairwise_fMRI[region_paire+f"_TR-{TRindex+1}"].loc[sub_paire[0], sub_paire[1]]=TR_diff[TRindex]

print('Preprocess behavioral data')
behavior_data = pd.read_excel('Behavioural_PSY_scored.xlsx')
#behavior_data = pd.read_excel('Data/Behavioural_PSY_scored.xlsx')
#create a new excel file with the subjects ordered by increasing number. Subject S12 and S18 were excluded fron the study, then we have 30 subjects with numbers
#going from S01 to S32
col_names = behavior_data.columns[1:] #do not need Unamed 0 since it will not be the correct number when subjects will be ordered correctly
sub_ordered_behavior = pd.DataFrame(columns=col_names, index = np.arange(0,32,1))
#loop over subject number we want to find
for sub in range (1,33):
    #by default, we take the subject number alone ie 11, 23...
    s='' 
    #if the subject number is smaller than 10, we need to add a 0 before so that we only end up with one subject. 
    # We only want subject 01 for 1 and not subject 1,10,11,12...
    if sub < 10:
        s='0'
    ref = s+str(sub)
    #loop to fill in the new dataframe with the ordered subjects and their corresponding value
    for i in range (30):
        initial_sub = behavior_data['ID'][i]
        condition = re.search(ref,initial_sub)
        if condition!= None:
            new = behavior_data.iloc[i].values[1:]
            sub_ordered_behavior.iloc[(sub-1)]= new
sub_ordered_behavior.to_excel('ordered_by_sub_behavior_score.xlsx', index=True)

#sub_ordered_behavior = pd.read_excel('ordered_by_sub_behavior_score.xlsx')

start_column_name = "cov_total"  # the name of the column you want to start from
#take all the behavior data from cov_total to the end
name_behavior_interet = ['ID']
name_behavior_interet2 = list(sub_ordered_behavior.columns[sub_ordered_behavior.columns.get_loc(start_column_name):])
name_behavior_interet.extend(name_behavior_interet2)
behavior_data_interest = sub_ordered_behavior[name_behavior_interet]

#five_sub_behavior = behavior_data_interest[:5]
#we exclude the subject number for the analysis -> start at 1
print('Behavioral computation')
labels = name_behavior_interet[1:]
#sub_list = [0,1,2,3,4] #from before
pairwise_behavior = {key: pd.DataFrame(columns=sub_list, index=sub_list) for key in labels}
for index_arrax in pairwise_subject_combination:
    Sub_A = behavior_data_interest.iloc[index_arrax[0]]
    Sub_B = behavior_data_interest.iloc[index_arrax[1]]
    #Sub_A = five_sub_behavior.iloc[index_arrax[0]]
    #Sub_B = five_sub_behavior.iloc[index_arrax[1]]
    for label in labels:
        pairwise_behavior[label].loc[index_arrax[0], index_arrax[1]] = (Sub_A[label]- Sub_B[label])


print('Store the data')
pairwise_data = {"fMRI": pairwise_fMRI, "behavior": pairwise_behavior}

# Save dictionary using pickle
pkl_file = open("Pairwise_Data_all_movie.pkl", "wb")
pickle.dump(pairwise_data, pkl_file)
pkl_file.close()



