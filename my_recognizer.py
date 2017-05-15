import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for i, (X, lengths) in test_set.get_all_Xlengths().items():
        # Well keep trak of the probabilities in a dict
        scores = {}
        # Here is where we assing probabilities to each word
        for word, model in models.items():
            try:
                # We get the probabilities score for this word from the current_model
                scores[word] = model.score(X, lengths)
            except:
                #Somthing went wrong lest assing the lowest probabilite
                scores[word] = float("-inf")
                pass

        probabilities.append(scores)
        # we Recognize the word base on the max value in the scores dict
        guess_word = max(scores, key= scores.get)
        guesses.append(guess_word)

    return probabilities,guesses
