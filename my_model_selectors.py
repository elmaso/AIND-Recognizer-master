import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_model = self.base_model(self.n_constant)
        best_score = float('inf')
        n_features = len(self.X[0])

        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                # We train the model using GaussianHMM if the logL is maximized then we set this as the best_score
                current_model = GaussianHMM(n_components=n_components, random_state=self.random_state, n_iter=1000).fit(self.X, self.lengths)
                logL = current_model.score(self.X, self.lengths)
                p = n_components * n_components + 2 * n_features*n_components - 1
                N = len(self.X)
                #We aplay beysian BIC = -2 * logL + p * logN
                current_score = (-2)*logL+p*np.log(N)
                # We test if the score maximized
                if current_score < best_score:
                    best_score = current_score
                    best_model = current_model
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_model = self.base_model(self.n_constant)
        best_score = float('-inf')

        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                # We train the model using GaussianHMM if the logL is maximized then we set this as the best_score
                current_model = GaussianHMM(n_components=n_components, random_state=self.random_state, n_iter=1000).fit(self.X, self.lengths)
                logL = current_model.score(self.X, self.lengths)

                sum_logL = []
                for word in self.words:
                    if word == self.this_word:
                        # we skeep curent word
                        continue
                    word_n, word_len = self.hwords[word]

                    sum_logL.append(current_model.score(word_n, word_len))

                current_score = logL - np.average(sum_logL)
                # We test if the score maximized
                if current_score > best_score:
                    best_score = current_score
                    best_model = current_model
            except:
                pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        #raise NotImplementedError

        best_model = self.base_model(self.n_constant)
        best_score = float('-inf')

        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                # Split dataset into k consecutive folds
                split_method = KFold(n_splits=min(3,len(self.lengths)))
                logL = []
                for train_id, test_id in split_method.split(self.sequences):
                    train_n,train_len = combine_sequences(train_id, self.sequences)
                    test_n,test_len = combine_sequences(test_id, self.sequences)
                    # We train the model using GaussianHMM if the logL is maximized then we set this as the best_score
                    current_model = GaussianHMM(n_components=n_components, random_state=self.random_state, n_iter=1000).fit(train_n, train_len)
                    logL.append(current_model.score(test_n, test_len))
                current_score = np.average(logL)
                #print("Word: {} Num States: {} LogL: {}".format(self.this_word,n_components,current_score))
                # We test if the score maximized
                if current_score > best_score:
                    best_score = current_score
                    best_model = current_model
            except:
                pass

        return best_model
