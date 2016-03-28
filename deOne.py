import pandas as pd
import gzip
import numpy as np

'''
Function to return dictionary which is like keyphrase_ratings
dictionary but instead of list of words as the first elements of
the value which is a tuple, there is similarity score of an
old keyphrase with the new_keyphrase
'''
# Calculate similarity of new key phrase w/ old key phrases
def calculate_similarities(new_keyphrase, keyphrase_ratings):

    # Initilize return dictionary
    similarities = dict()

    # Get the length of the new_keyphrase with respect to words in it
    new_keyphrase_length = len(new_keyphrase['keyphrase'].split())

    # Iterate over each word of new_keyphrase
    for i in range(new_keyphrase_length):

        # Iterate over each old keyphrase in keyphrase_ratings dictionary
        for key in keyphrase_ratings:

            # Initilize variable to hold the value of number of common words
            # between the new_keyphrase and one of old keyphrases
            common_word_count = 0

            # Get the length of an old keyphrase with respect to words in it
            old_keyphrase_length = len(keyphrase_ratings[key][0])

            # Iterate over each word of old keyphrase
            for j in range(old_keyphrase_length):

                # Compare each word in new and old keyphrase
                if keyphrase_ratings[key][0][j] == \
                   new_keyphrase['keyphrase'].split()[i]:

                    # If there is a common word then increase count
                    # of common_word_count
                    common_word_count += 1

            # Calculate similarity new and an old keyphrase percentage-wise
            # similarity =\
            #  ((float(common_word_count)/float(old_keyphrase_length)) +\
            #  (float(common_word_count)/float(new_keyphrase_length))) / 2.0

            similarity = common_word_count

            # If similarity not 0 then put found old keyphrase into
            # similarities dictionary
            if similarity != 0:
                similarities[key] = (similarity, keyphrase_ratings[key][1])

    # This return empty dictionary if there is no similar old ad
    return similarities

'''
Function to return percentage-wise CTR prediction, given the new_keyphrase and
its similarities dictionary which contains similarities of already known old
keyphrases with the new_keyphrase and ratings of that old keyphrases as values
and old keyphrases row name as keys
'''
def calculate_ctr(similarities):

    if len(similarities) != 0:

        # Initilize needed variabels for calculation
        sum_sim_and_rating = 0
        sum_sim = 0

        # Iterate over similarities dictionary
        for key in similarities:

            # Multiply similarity score with rating for each old keyphrase
            # and add them up
            sum_sim_and_rating += similarities[key][0] * similarities[key][1]

            # Add similarity score of keywords
            sum_sim += similarities[key][0]

        return (float(sum_sim_and_rating)/float(sum_sim))

    else: return None

'''
Function to return dictionary containing all the ads
whos impression is greater than 100 with
their keywords and CTRs as values, their row name as keys
'''
def calculate_keyphrase_ratings(data_table, traing_length,
                                min_impression, min_clicks):

    # Initilize return dictionary
    keyphrase_ratings = dict()

    # Initlize variables to calculate avarage CTR
    sum_ctr = int()
    training_samples = int()

    # Iterate last traing_length rows of data_table
    for index, row in data_table.tail(traing_length).iterrows():

        # In case we want to specify how much sample to choose
        # if training_samples == traing_length: break

        # Calculate rating for keyphrases whos impression > 100.0
        if row['clicks'] >= min_clicks and row['impression'] >= min_impression:

            # Put words of found keyphrase into a list
            words_in_keyphrase = row['keyphrase'].split()

            # Calculate CTR of found keyphrase
            CTR = float(row['clicks']) / float(row['impression'])

            # Increment training_samples and add CTRs up
            sum_ctr += CTR
            training_samples += 1

            # Put words and CTR of found keyphrase into dictionary
            # by setting its key as its name
            keyphrase_ratings[row.name] = (words_in_keyphrase, CTR)

    # Calculate avarage CTR
    avarage_ctr = float(sum_ctr) / float(training_samples)

    return keyphrase_ratings, avarage_ctr


def test_model(data_table, test_length, min_impression, min_clicks,
               keyphrase_ratings, avarage_ctr):

        """Function to calculate mean square error of our model given
        data_table, test_length, min_impression and min_clicks to choose new ad
        to put on testing, keyphrase_ratings which contains with words and
        ratings of old ads and avarage_ctr of training set"""

        # Variables to calculate accuracy
        square_erros = 0
        square_erros_avarage = 0
        test_samples_num = 0

        # Iterate first test_length rows of data_table
        for index, row in data_table.head(test_length).iterrows():

            # In case we want to specify how much sample to choose
            # if test_samples_num == test_length: break

            # Choose ads whos impression > min_impression
            if row['clicks'] >= min_clicks and\
               row['impression'] >= min_impression:

                # Set found row as new_keyphrase
                new_keyphrase = row

                # Calculate similarities dictionary for new_keyphrase
                similarities = calculate_similarities(new_keyphrase,
                                                      keyphrase_ratings)

                # Predict CTR by using calculate_ctr
                predicted_ctr = calculate_ctr(similarities)

                # Calculate real CTR of new_keyphrase
                real_ctr = float(new_keyphrase['clicks']) /\
                    float(new_keyphrase['impression'])

                # Mean square error calculation
                if predicted_ctr is not None:

                    # Sum up square of errors
                    square_erros += (predicted_ctr - real_ctr) ** 2
                    square_erros_avarage += (avarage_ctr - real_ctr) ** 2

                    # Increment number of found test samples
                    test_samples_num += 1

                    # Display some infromations
                    print('Predicted CTR: %f - Real CTR: %f for sample: %d'
                          % (predicted_ctr, real_ctr, test_samples_num))

                else:
                    print('No similarity found...')

        print("\nMean Square Error: %f\n" % (float(square_erros) /
                                             float(test_samples_num)))
        print("w/ Avarage CTR prediction %f\n" % (float(square_erros_avarage)/
                                                  float(test_samples_num)))

###############################################################################

def main():

    # Set data_name
    data_name = 'dataset.gz'

    # Define required variables
    data_name = 'dataset.gz'
    training_length = 10000000
    test_length = 1000000
    min_impression = 1000
    min_clicks = 1
    min_impression_test = 1000
    min_clicks_test = 1

    # Read data_name with pandas read_table function
    dt = pd.read_table(gzip.open(data_name))

    # Rename columns as follows
    dt.columns = ['day', 'account_id', 'rank', 'keyphrase', 'avg_bid',
                  'impression', 'clicks']

    # Shuffle the data table
    dt = dt.iloc[np.random.permutation(len(dt))]

    # Calculate keyphrase CTRs for training_length ads whos impression greater
    # than min_impression keyphrase_ratings is a dictionary containing
    # keyphrase words and ratings of old ads
    keyphrase_ratings, avarage_ctr = calculate_keyphrase_ratings(dt,\
        training_length, min_impression, min_clicks)

    # Calculate mean square error of our model
    test_model(dt, test_length, min_impression_test, min_clicks_test,\
        keyphrase_ratings, avarage_ctr)
