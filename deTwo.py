import pandas as pd
import numpy as np
import gzip
from sklearn import linear_model
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

'''
Function to return a dictionary with row.name as keys
and its word_vector and rating as values given data_table
and all_words
'''
def calculate_keyphrase_vectors(data_table, all_words, training_length,\
                                min_impression, min_clicks):

    # Initilize return dictionary
    keyphrase_vectors  = dict()

    # Initlize variables to calculate avarage CTR
    sum_ctr = int()
    training_samples = int()

    # Iterate training_length row of data_table
    for index, row in data_table.tail(training_length).iterrows():

        # Found ads whos impression > 100.0
        if row['clicks'] >= min_clicks and row['impression'] >= min_impression:

            # Turn words in found ad into list
            word_list = row['keyphrase'].split()

            # Initilize word_vector of found ad
            word_vector = list()

            # Iterate over all_words
            for word in all_words:

                # If word is present in found ad keyphrase
                # append 1 into word_vector, 0 otherwise
                if word in word_list: word_vector.append(1);
                else: word_vector.append(0)

            # Calculate CTR of found ad
            CTR = float(row['clicks']) / float(row['impression'])

            # Increment training_samples and add CTRs up
            sum_ctr += CTR
            training_samples += 1

            # Add ad into keyphrase_vectors with row.name as key
            # and its word_vector and CTR as value
            keyphrase_vectors[row.name] = (word_vector, CTR)

    # Calculate avarage CTR
    avarage_ctr = float(sum_ctr) / float(training_samples)

    return keyphrase_vectors, avarage_ctr

'''
Function to train and return the model given data_table and all_words as list
'''
def train_model(data_table, all_words, training_length,\
                min_impression, min_clicks):

    # Calculate keyphrase vectors via
    # calculate_keyphrase_vectors function
    keyphrase_vectors, avarage_ctr = calculate_keyphrase_vectors(data_table,\
        all_words, training_length, min_impression, min_clicks)

    # Initlize list required for regression model
    words_vectors = list()
    ratings = list()

    # Iterate over found keyphrase_vectors
    for key in keyphrase_vectors:

        # Put word vectors and ratings into lists from keyphrase_vectors
        word_vector = keyphrase_vectors[key][0]
        rating = keyphrase_vectors[key][1]
        words_vectors.append(word_vector)
        ratings.append(rating)

    # Initilize scikit-learn model
    # clf = linear_model.LinearRegression()
    clf = linear_model.SGDRegressor(loss='squared_loss')

    # Traing it with words_vectors as input and ratings as labels
    clf.fit(words_vectors, ratings)

    return clf, avarage_ctr

def test_model(dt, clf, all_words, test_length, min_impression_test,\
               min_clicks_test, avarage_ctr):

    # Variables to calculate accuracy
    square_erros = 0
    square_erros_avarage = 0
    test_samples_num = 0

    # Iterate first test_length rows of data_table
    for index, row in dt.head(test_length).iterrows():

        # Choose ads whos impression > 100.0
        if row['clicks'] >= min_clicks_test and\
           row['impression'] >= min_impression_test:

            # Set found row as new_keyphrase
            new_keyphrase = row

            # Put words of new_keyphrase into list
            keyphrase_words = new_keyphrase['keyphrase'].split()

            # Initilize keyphrase_vector for new_keyphrase
            keyphrase_vector = list()

            # Iterate over all_words
            for word in all_words:

                # Build keyphrase_vector of new_keyphrase
                if word in keyphrase_words: keyphrase_vector += [1]
                else: keyphrase_vector += [0]

            # Predict CTR by using scikit-learn
            predicted_ctr = clf.predict(keyphrase_vector)[0]

            # Calculate real CTR of new_keyphrase
            real_ctr = float(new_keyphrase['clicks']) /\
                       float(new_keyphrase['impression'])

            # Sum up square of errors
            square_erros += (predicted_ctr - real_ctr) ** 2
            square_erros_avarage += (avarage_ctr - real_ctr) ** 2

            # Increment number of found test samples
            test_samples_num += 1

            # Display some infromations
            print('Predicted CTR: %f - Real CTR: %f for n: %d' %\
                  (predicted_ctr, real_ctr, test_samples_num))


    print("\nMean Square Error: %f\n" % (float(square_erros) /\
                                         float(test_samples_num)))
    print("w/ Avarage CTR prediction %f\n" % (float(square_erros_avarage) /\
                                              float(test_samples_num)))

################################################################################

def main():

    # Define required variables
    data_name = 'dataset.gz';

    # Define required variables 
    training_length = 100000; 
    test_length = 50000;  
    min_impression = 100;
    min_clicks = 0; 
    min_impression_test = 100;
    min_clicks_test = 0;

    # Read data_name with read_table function
    dt = pd.read_table(gzip.open('dataset.gz'))

    # Read key_name with read_table function
    dt_keys = pd.read_csv('keyset.txt')

    # Rename columns as follows
    dt.columns = ['day', 'account_id', 'rank', 'keyphrase', 'avg_bid',\
                  'impression', 'clicks']
    dt_keys.columns = ['keys']

    # Shuffle the datas
    dt = dt.iloc[np.random.permutation(len(dt))]

    # Initilize a list to hold all words
    all_words = list()

    # Iterate over data_table_keys, put each word into all_words list
    for index, row in dt_keys.iterrows(): all_words.append(row['keys'])

    # Find fit a model by using train_model
    clf, avarage_ctr = deTwo.train_model(dt, all_words, training_length,\
                                         min_impression, min_clicks)

    # Calculate mean square error of our model
    test_model(dt, clf, all_words, test_length, min_impression_test,\
               min_clicks_test)
