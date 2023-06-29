import numpy as np
from pyitlib import discrete_random_variable as drv


class mRmd():
    """
    filter features using min redundancy max dependency
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
        self.score = []

    def fit(self, X, y):
        return self._fit(X, y)

    def calculate_mutual_information(self, x, y):
        """ calculate I(X;Y)=H(Y)−H(Y|X)"""
        H_y = drv.entropy(y)
        cH_y_k = drv.entropy_conditional(y, x)
        return H_y - cH_y_k

    def calculate_new_feature_redundancy_term(self, Xk, Xj, y):
        """ calculate  I(Xj;Xk) and I(Xk;Y |Xj)"""
        # I(Xj;Xk)
        H_k = drv.entropy(Xk)
        cH_k_j = drv.entropy_conditional(Xk, Xj)
        mi_k_j = H_k - cH_k_j

        # I(Xk;Y |Xj)
        cH_y_j = drv.entropy_conditional(y, Xj)
        cH_k_y_j = drv.entropy_conditional(Xk, y, Xj)
        cmi = cH_k_j + cH_y_j - cH_k_y_j

        return mi_k_j, cmi

    def choose_feature_maximizes_mrmd(self, redundancy, mi, selected, F):
        """ calculate J(Xk)=I(Xk;Y)−1|S|∑Xj∈S{I(Xj;Xk)−I(Xk;Y|Xj)}"""
        mrmd_values = dict()
        inverse_s = 1 / len(selected)
        for x_k in F:

            # I(Xk;Y)
            mi_k_y = mi[x_k]

            # ∑Xj∈S{I(Xj;Xk)−I(Xk;Y|Xj)}
            sum_over_selected = 0
            for x_j in selected:
                mi_j_k = redundancy[(x_k, x_j)][0]  # I(Xj;Xk)
                mi_k_y_j = redundancy[(x_k, x_j)][1]  # I(Xk;Y|Xj)
                sum_over_selected += mi_j_k - mi_k_y_j
            mrmd_k = mi_k_y - (inverse_s * sum_over_selected)
            mrmd_values[x_k] = mrmd_k
        selected_feature = max(mrmd_values, key=mrmd_values.get)
        feature_score = mrmd_values[selected_feature]
        return selected_feature, feature_score

    def _fit(self, X, y):
        # stage 1 - initialization
        S = list()
        n_features = X.shape[1]
        features_index = [i for i in range(n_features)]

        # stage 2 - calculate the mutual information between the class and each candidate feature
        mi_features_target = dict()
        for i in range(n_features):
            mi_features_target[i] = self.calculate_mutual_information(X[:, i], y)

        # stage 3 - select the first feature
        S.append(max(mi_features_target, key=mi_features_target.get))
        self.score.append(mi_features_target[S[0]])

        # stage 4 - greedy selection
        feature_redundancy = dict()
        while len(S) < self.n_features_to_select:
            F = list(set(features_index) - set(S))

            # stage 4.a - calculate the new feature redundancy term
            for X_k in F:
                mi_jk, mi_kyj = self.calculate_new_feature_redundancy_term(X[:, X_k].tolist(), X[:, S[-1]].tolist(), y.tolist())
                feature_redundancy[(X_k, S[-1])] = (mi_jk, mi_kyj)
            # stage 4.b - select the next feature

            feature, score = self.choose_feature_maximizes_mrmd(feature_redundancy, mi_features_target, S, F)
            S.append(feature)
            self.score.append(score)
        self.features = np.array(S)
        self.score = np.array(self.score)