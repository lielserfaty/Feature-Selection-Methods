import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif
from skfeature.function.statistical_based import CFS
from pyitlib import discrete_random_variable as drv

class ReduceDRF0Improve():
    """
          filter features using univariate and multivariate filter
          :param n_features_to_select - number of feature the function need to select over all the features
          :parameter features - list of the k selected features
          :parameter score - list of all the score of each feature that selected
          """
    def __init__(
            self,
            n_features_to_select,

    ):
        self.n_features_to_select = n_features_to_select
        self.features = []
        self.score = {}

    def fit(self, X, y):
        return self._fit(X, y)

    def map_local_index(self, local_index, global_index):
        """ map the local index columns of the subset partition to the dataset """
        c = [global_index[i] for i in local_index]
        return c

    def calculate_mutual_information(self, x, y):
        """ calculate I(X;Y)=H(Y)−H(Y|X)"""
        H_y = drv.entropy(y)
        cH_y_k = drv.entropy_conditional(y, x)
        return H_y - cH_y_k

    def partition_dataset(self, X, y):
        """ ranking and order the features by calculate the mutual_information of each feature"""

        mi_features_target = dict()
        for i in range(X.shape[1]):
            mi_features_target[i] = self.calculate_mutual_information(X[:, i], y)
        sorted_score = sorted(mi_features_target.items(), key=lambda x: x[1], reverse=True)
        sorted_dict = {k: v for k, v in sorted_score}
        return list(sorted_dict.keys())

    def CFS(self, X, y):
        """ calculate the correlation-based feature selection. CFS is multivariate filtering algorithm
        that ranks subsets of features according to a correlation-based heuristic evaluation function,
        thereby ensuring that there are no irrelevant features """

        idx = CFS.cfs(X, y)
        idx_list = np.ndarray.tolist(idx)
        if len(idx_list) > X.shape[1]:
            idx_list.remove(X.shape[1])
        return idx_list

    def svm(self, X, y):
        """ calculate the accuracy score of SVM classifier on subset data X  """
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=4)
        clf = SVC()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        return accuracy


    def get_information_gain_features(self, X, y):
        """ calculate the top 25% of features by information gain measure """
        selector = SelectPercentile(mutual_info_classif, percentile=25)
        X_reduced = selector.fit_transform(X, y)
        cols = selector.get_support(indices=True)
        score_select_featuers = [selector.scores_[i] for i in cols]
        return cols, score_select_featuers


    def _fit(self, X, y):
        dict_group_sub = {}
        dict_group_original_index = {}

        n_features = X.shape[1]
        # number of features in each group
        k = int(X.shape[0] / 2)
        # number of group with k features
        n = int(n_features / k)

        # stage 1- sorting the features by the value of the information from the largest to the smallest
        index_column = self.partition_dataset(X, y)

        # stage 2- dividing the dataset into n subgroups of size k features
        for i in range(n + 1):
            if i == n:
                d_i = X[:, index_column[i * k:]]
                dict_group_original_index[i] = index_column[i * k:]

            else:
                d_i = X[:, index_column[i * k:i * k + k]]
                dict_group_original_index[i] = index_column[i * k:i * k + k]
            # stage 3 - filter each subset by CFS
            dict_group_sub[i] = self.CFS(d_i, y)

        # calculate the baseline accuracy for the first subset
        global_index_select = self.map_local_index(dict_group_sub[0][0], dict_group_original_index[0])
        x_sub = X[:, global_index_select]

        # stage 4 - the first subset is including in S
        S = global_index_select

        # stage 5- calculate the accuracy score of the first subset
        baseline = self.svm(x_sub, y)
        for i in range(1, n + 1):

            # stage 6- calculate the accuracy score for each subset with the featuers in S
            # and append the subset to S set if the accuracy is improved
            s_i = dict_group_sub[i]
            global_index_select = self.map_local_index(s_i[0], dict_group_original_index[i])
            S.extend(global_index_select)
            x_sub = X[:, S]
            accuracy = self.svm(x_sub, y)
            if accuracy > baseline:
                baseline = accuracy

                # stage 7- find the most relevant features in S
                s_aux_global, scores = self.get_information_gain_features(x_sub, y)
                # stage 8 - calculate the accuracy of the new subset
                s_aux_temp = self.map_local_index(s_aux_global, S)
                x_sub = X[:, s_aux_temp]
                accuracy = self.svm(x_sub, y)

                # stage 9 - if the accuracy improved set the S to be the new subset
                if accuracy > baseline:
                    baseline = accuracy
                    S = s_aux_temp

            else:
                S = [x for x in S if x not in global_index_select]

        self.features = np.array(S)
        self.score = np.array(self.set_score(X.shape[1]))
