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

print('Resampling')
movies_list = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LessonLearned', 'Payload',
               'Sintel', 'Spaceman', 'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain']
TR_list_preprocessed = {'AfterTheRain':382, 'BetweenViewings':624, 'BigBuckBunny':376, 'Chatter':312, 'FirstBite':461, 
           'LessonLearned':513, 'Payload':776, 'Sintel':558, 'Spaceman':592, 'Superhero':793, 'TearsOfSteel':455, 
           'TheSecretNumber':605, 'ToClaireFromSonny':308, 'YouAgain':616}

def resampling_movie_annot (movie): 
    name_json = 'Data/Annot_'+movie+'_stim.json'
    with open(name_json, 'r') as f:
        # Lecture du contenu du fichier et chargement des données JSON en tant que dictionnaire
        annot = json.load(f)
    index_anx = annot['Columns'].index('Anxiety')

    name_tsv = 'Data/Annot_'+movie+'_stim.tsv'
    ratings = np.genfromtxt(name_tsv, delimiter='\t')
    anx_rating = ratings[:,index_anx]

    #Ressample
    resampling_len = TR_list_preprocessed[str(movie)]
    anx_rating_ressampled= signal.resample(anx_rating, resampling_len)
    return {movie: anx_rating_ressampled}

result = list(map(resampling_movie_annot, movies_list))

anx_movies_rating = np.concatenate((result[0]['AfterTheRain'], result[1]['BetweenViewings'], result[2]['BigBuckBunny'], result[3]['Chatter'],
                                    result[4]['FirstBite'], result[5]['LessonLearned'], result[6]['Payload'], result[7]['Sintel'], result[8]['Spaceman'],
                                    result[9]['Superhero'], result[10]['TearsOfSteel'], result[11]['TheSecretNumber'], result[12]['ToClaireFromSonny'],
                                    result[13]['YouAgain']))

print('Load Movie wise stat data')
pkl_file = open("Computation/Group/all_amygd_Movie_wise_group_analysis_500_perms.pkl", "rb")
movie_wise_stat = pickle.load(pkl_file)
pkl_file.close()

print('TR_wise analysis')
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
num_perms = 10000
pair_args = list(product(r_interest_TR, name_behavior_interest))

def null_dist_pearson(x, y, perms=500): #perms = je ne sais pas trop quoi, à quoi correspond la valeur par défaut?
    rng = np.random.default_rng()
    r = np.tile(rng.choice(len(x), size=len(x), replace=False), (perms,1))
    r = rng.permuted(r, axis=1)
    x_new = np.take(np.array(x), r, axis=0)
    y_new = np.tile(np.array(y),(perms,1))
    shuffled_corr = np.apply_along_axis(lambda x_: pg.corr(x=x_[:len(x)], y=x_[len(x):], method='pearson')["r"][0], axis=1, arr=np.concatenate((x_new, y_new), axis=1))
    return shuffled_corr

def compute_TRs_analysis(region, behavior):
    amygdala_mask = movie_wise_stat["Connection"].str.contains("TR-") & \
                movie_wise_stat["Connection"].str.contains(region) & \
                movie_wise_stat["Behavior"].str.contains(behavior)
    #For amygdala
    x = np.array(movie_wise_stat["r"][amygdala_mask])
    y = anx_movies_rating
    amyg_sr_suspense = pg.corr(x,y, method="pearson")
    null_distribution = null_dist_pearson(x, y, num_perms)
    greater_less_sum = np.sum(np.absolute(np.arctanh(null_distribution)) <= np.absolute(np.arctanh(amyg_sr_suspense.r[0])))
    amyg_sr_suspense_sig = 1.0 - (greater_less_sum / num_perms)

    new_df = pd.DataFrame([[behavior, region, amyg_sr_suspense.r[0],amyg_sr_suspense['p-val'][0], amyg_sr_suspense_sig]],
                          columns = ["Behavior", "Connection", "r", "p", "perm_p"])
    return new_df

start_time = time.time()
nbr = multiprocessing.cpu_count()
with Pool (processes=nbr) as pool: #processes = number of thread (coeur) I use on my computer
    result = pool.starmap(compute_TRs_analysis, pair_args) 
statistical_tests = pd.concat((result), axis =0, ignore_index=True)
time_ = (time.time() - start_time)
heures = (time_ % (24 * 3600)) // 3600
minutes = (time_ % 3600) // 60
secondes = time_ % 60
print("Cela a pris {} heures, {} minutes, et {} secondes.".format(int(heures), int(minutes), round(secondes, 2)))

print('Save Group results')
pkl_file = open("Computation/Group/TRs_wise_amydgala_group_results_10000_perms.pkl", "wb")
pickle.dump(statistical_tests, pkl_file)
pkl_file.close()

statistical_tests.to_csv("Computation/Group/TRs_wise_amydgala_group_results_10000_perms.csv", float_format='%.7f')
