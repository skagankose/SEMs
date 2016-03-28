import pandas as pd
import numpy as np
import gzip
from sklearn import linear_model
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


def calculate_keyphrase_vectors(dt, dt_keyphrases, training_length):
    """Function to return a dictionary with row.name as keys 
    and word_vectors and ratings as values """

    keyphrase_vectors  = dict()
    keyphrase_words = dict()
    sum_ctr = int()
    train_count = int()

    for index, row in dt.tail(training_length).iterrows():

        old_vector = list()
        old_words = row['keyphrase'].split() 

        for word_index, word_row in dt_keyphrases.iterrows():  
            if word_row['keys'] in old_words: old_vector.append(1)  
            else: old_vector.append(0)

        ctr_real = float(row['clicks']) / float(row['impression'])

        keyphrase_vectors[row.name] = (old_vector, ctr_real)
        keyphrase_words[row.name] = (old_words, ctr_real)

        sum_ctr += ctr_real
        train_count += 1

        print('Training sample: %d', train_count)
            
    avarage_ctr = float(sum_ctr) / float(train_count)

    return keyphrase_vectors, keyphrase_words, avarage_ctr


def train_regression(keyphrase_vectors):
    """Function to train regression model"""

    clf = linear_model.SGDRegressor(loss='squared_loss')
    old_vectors = list()                                
    ratings = list()                                   

    for key in keyphrase_vectors:

        old_vectors.append(keyphrase_vectors[key][0])
        ratings.append(keyphrase_vectors[key][1])
        
    clf.fit(old_vectors, ratings)

    return clf


def train_collaborative(new_keyphrase, keyphrase_words):
    """Function to train collaborative model"""

    similarities = dict()

    new_keyphrase_length = len(new_keyphrase['keyphrase'].split())

    for i in range(new_keyphrase_length):

        for key in keyphrase_words:

            common_count = 0
            old_keyphrase_length = len(keyphrase_words[key][0])

            for j in range(old_keyphrase_length):

                if keyphrase_words[key][0][j] == new_keyphrase['keyphrase'].split()[i]:

                    common_count += 1

            similarity =\
                        ((float(common_count)/float(old_keyphrase_length)) +\
                         (float(common_count)/float(new_keyphrase_length))) / 2.0

            if similarity != 0: similarities[key] = (similarity, keyphrase_words[key][1])

    return similarities


def predict_regression(clf, new_keyphrase, dt_keyphrases):
    """Function to return regression CTR prediction"""

    new_vector = list()
    new_words = new_keyphrase['keyphrase'].split() 
                                                          
    for word_index, word_row in dt_keyphrases.iterrows():

        if word_row['keys'] in new_words: new_vector.append(1)  
        else: new_vector.append(0)

    return clf.predict(new_vector)[0]


def predict_collaborative(similarities):
    """Function to return collaborative CTR prediction"""

    if len(similarities) != 0:

        s1 = 0; s2 = 0

        for key in similarities:

            s1 += similarities[key][0] * similarities[key][1]
            s2 += similarities[key][0]

        return float(s1)/float(s2)


def test_model(dt, dt_keyphrases, keyphrase_vectors, keyphrase_words, ctr_avarage, test_length):
    """Function to display accuracy"""

    errors_collaborative = 0
    errors_regression = 0
    errors_avarage = 0
    test_count = 0

    clf = train_regression(keyphrase_vectors)

    for index, row in dt.head(test_length).iterrows():

        if row['clicks'] == 0:
            print('Yes!')

        new_keyphrase = row

        ctr_real = float(new_keyphrase['clicks']) / float(new_keyphrase['impression'])
        ctr_regression = predict_regression(clf, new_keyphrase, dt_keyphrases)

        similarities = train_collaborative(new_keyphrase, keyphrase_words)
        ctr_collaborative = predict_collaborative(similarities)

        errors_collaborative += (ctr_collaborative - ctr_real) ** 2
        errors_regression += (ctr_regression - ctr_real) ** 2
        errors_avarage += (ctr_avarage - ctr_real) ** 2

        test_count += 1

        print('Reg: %.7f |Col: %.7f |Real: %.7f |Avrg: %.7f - Sample: %d' %\
              (ctr_regression, ctr_collaborative, ctr_real, ctr_avarage, test_count))
 
    print("\nRegression: %.10f\n" % (float(errors_regression) / float(test_count)))
    print("\nCollaborative: %.10f\n" % (float(errors_collaborative) / float(test_count)))
    print("\nAvarage: %.10f\n" % (float(errors_avarage) / float(test_count)))


################################################################################################


def main():

    training_length = 10000; 
    test_length = 2000;


    dt = pd.read_table(gzip.open('dataSet.gz'))
    dt_keyphrases = pd.read_csv('keyphraseSet.txt')

    dt.columns = ['day', 'account_id', 'rank', 'keyphrase', 'avg_bid', 'impression', 'clicks']
    dt_keyphrases.columns = ['keys']

    dt = dt.iloc[np.random.permutation(len(dt))]

    rows = list()
    for index, row in dt.iterrows():
        if row['clicks'] >= 1 and row['impression'] >= 1000:
            rows.append(row)
    df = pd.DataFrame(rows)

    keyphrase_vectors, keyphrase_words, ctr_avarage =\
        calculate_keyphrase_vectors(df, dt_keyphrases, training_length)

    test_model(df, dt_keyphrases, keyphrase_vectors, keyphrase_words, ctr_avarage, test_length)

    