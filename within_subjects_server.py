import os
import re
import numpy as np
import timecorr as tc
from itertools import product
import scipy.stats
import pandas as pd
#from functions import load_data_server

#Function definition
def compute_timecorr(brain_regions):
    timecorr_br = tc.timecorr(brain_regions, weights_function=gaussian['weights'], weights_params=gaussian['params'])[:,2]
    timecorr_br = np.arctanh(timecorr_br)  # Fisher transform
    timecorr_br = scipy.stats.zscore(timecorr_br)  # Z-score
    return timecorr_br

def split_path_to_data_server(film, name):
    film_comparison = film + '.csv'
    file_name = os.listdir('Data')
    pattern_str = fr"^{name}.*{film_comparison}$"
    pattern = re.compile(pattern_str)
    match = list(filter(pattern.match, file_name))
    return sorted(match)

def load_data_server(film='Rest', TR=460, preprocess = True ): #mettre une taille définie par TR par film (faudra stocker une liste de film et de taille correspondante à un moment)
    print(f"Preprocess {preprocess}")
    path14 = split_path_to_data_server(film,'TC_14')
    path400 = split_path_to_data_server(film,'TC_400')
    print(path14)
    print(path400)
    data_all_subjects=[]
    begin = 0
    end = TR
    if preprocess: 
        begin = 76
        end = (TR-76)
    path_to_folder = 'Data/'
    for n in range (len(path14)):
        data14=np.genfromtxt(path_to_folder + path14[n], delimiter=",")
        data400=np.genfromtxt(path_to_folder + path400[n], delimiter=",")
        data_one_subject=np.concatenate([data400[begin:end], data14[begin:end]], axis=1)
        data_all_subjects.append(data_one_subject)
    return np.array(data_all_subjects)

def compute_dyn_corr(data_film, TR, combination_brain_regions, cort400_name, subcort14_name):
    reg1_x_reg2 = [r[0] + "_x_" + r[1] for r in combination_brain_regions]
    all_sb_dynamic_connectivity=[]
    for sb in range (data_film.shape[0]):
        print(f"Subject {sb}")
        dynamic_connectivity = {"fMRI": pd.DataFrame(columns=reg1_x_reg2, index=range(0, TR))} #à mettre en paramètre quelque part = propore à chaque film
        for r in range (len(combination_brain_regions)):
            #extract data points of pair of brain regions for subject sb
            index_subcort = 400 + subcort14_name.index(combination_brain_regions[r][0]) #brain regions from 400 to 414
            r_subcort = data_film[sb,:,(index_subcort)]

            index_cort_sub=0
            if subcort14_name.count(combination_brain_regions[r][1])!=0:
                index_cort_sub = 400 + subcort14_name.index(combination_brain_regions[r][1])
            else : index_cort_sub = cort400_name.index(combination_brain_regions[r][1]) #brain regions from 0 to 400
            r_cort_sub = data_film[sb,:,index_cort_sub]
            subcort_corSubcort = np.vstack((r_subcort, r_cort_sub)).transpose()
            #compute timecorr for one subject and a pair brain regions
            timecorr = compute_timecorr(subcort_corSubcort)

            #fill in the dataframe
            dynamic_connectivity['fMRI'][combination_brain_regions[r][0] + '_x_' + combination_brain_regions[r][1]] = timecorr
        all_sb_dynamic_connectivity.append(dynamic_connectivity)
    return all_sb_dynamic_connectivity

#Load data
"""data=load_data_server(film='Sintel', TR=710)
data_Sintel=np.array(data)
print('Data loaded')"""

movies_list = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LessonLearned', 'Payload',
               'Sintel', 'Spaceman', 'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain']
TR_list = {'AfterTheRain':534, 'BetweenViewings':776, 'BigBuckBunny':528, 'Chatter':464, 'FirstBite':613, 
           'LessonLearned':665, 'Payload':928, 'Sintel':710, 'Spaceman':744, 'Superhero':945, 'TearsOfSteel':607, 
           'TheSecretNumber':757, 'ToClaireFromSonny':460, 'YouAgain':768}
TR_list_preprocessed = {'AfterTheRain':382, 'BetweenViewings':624, 'BigBuckBunny':376, 'Chatter':312, 'FirstBite':461, 
           'LessonLearned':513, 'Payload':776, 'Sintel':558, 'Spaceman':592, 'Superhero':793, 'TearsOfSteel':455, 
           'TheSecretNumber':605, 'ToClaireFromSonny':308, 'YouAgain':616}


width = 30 
gaussian = {'name': 'Gaussian', 'weights': tc.gaussian_weights, 'params': {'var': width}}
amygdala = ["Left Amygdala", "Right Amygdala"]

#load cortical brain region names into a list
#path_cort400_names = 'Data/TC_cort400_labels.csv'
path_cort400_names = 'Data/TC_cort400_labels.csv'
with open(path_cort400_names, encoding='utf-8-sig') as f:
    cort400_Name = np.genfromtxt(f, dtype=str, delimiter=',').tolist()

#load subcortical brain region names into a list
#path_subcort14_name = 'Data/TC_sub14_labels.csv'
path_subcort14_name = 'Data/TC_sub14_labels.csv'
with open(path_subcort14_name, encoding='utf-8-sig') as f:
    subcort14_Name = np.genfromtxt(f, dtype=str, delimiter=',').tolist()

#Concatenate both
cort_sub_414_name = cort400_Name + subcort14_Name

combination_amygdala_x_cort_sub = list(product(amygdala, cort_sub_414_name))

print("Compute Dynamic Correlation")
for film in (movies_list):
    print(film)
    data = load_data_server(film=film, TR=TR_list[str(film)])
    print('Data Loaded')
    dyn_corr_film = compute_dyn_corr(data, TR_list_preprocessed[str(film)], combination_amygdala_x_cort_sub, cort400_Name, subcort14_Name)

    #Save the results 
    name = 'Computation/' + film +'_amygdala_dyn_connectivity_width_30.pkl'
    #name =  film +'_Test.pkl'
    with open(name, 'wb') as f:
        pd.to_pickle(dyn_corr_film, f)


"""#Compute dynamic correlation for each subjects on all combination of brain area
Amygdala_x_cort = [r[0] + "_x_" + r[1] for r in combination_amygdala_x_cort_sub]
all_sb_dynamic_connectivity=[]
for sb in range (data_Sintel.shape[0]):
    dynamic_connectivity = {"fMRI": pd.DataFrame(columns=Amygdala_x_cort, index=range(0, 558))} #à mettre en paramètre quelque part = propore à chaque film
    for r in range (len(combination_amygdala_x_cort_sub)):
        #extract data points of pair of brain regions for subject sb
        index_subcort = 400 + subcort14_Name.index(combination_amygdala_x_cort_sub[r][0]) #brain regions from 400 to 414
        r_subcort = data_Sintel[sb,:,(index_subcort)]

        index_cort_sub=0
        if subcort14_Name.count(combination_amygdala_x_cort_sub[r][1])!=0:
            index_cort_sub = 400 + subcort14_Name.index(combination_amygdala_x_cort_sub[r][1])
        else : index_cort_sub = cort400_name.index(combination_amygdala_x_cort_sub[r][1]) #brain regions from 0 to 400
        r_cort_sub = data_Sintel[sb,:,index_cort_sub]
        subcort_corSubcort = np.vstack((r_subcort, r_cort_sub)).transpose()
        #compute timecorr for one subject and a pair brain regions
        timecorr = compute_timecorr(subcort_corSubcort)

        #fill in the dataframe
        dynamic_connectivity['fMRI'][combination_amygdala_x_cort_sub[r][0] + '_x_' + combination_amygdala_x_cort_sub[r][1]] = timecorr
    all_sb_dynamic_connectivity.append(dynamic_connectivity)"""










