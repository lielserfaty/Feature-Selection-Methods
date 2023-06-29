import pandas as pd
import os
import numpy as np
import augment as aug

path = './output/'


def concat_all_db():
    """
    separate function to concat all the df of result
    """
    for i, filename in enumerate(os.listdir(path)):
        if i == 0:
            df = pd.read_csv(path+filename)
            continue
        df_temp = pd.read_csv(path+filename)
        df = pd.concat([df, df_temp], ignore_index=True)
    df.to_csv(path+'All DB - All Results.csv')
    df = df.loc[df['Measure Type'] == 'AUC']
    df.to_csv(path + 'All DB - AUC.csv')


def create_new_results_df():
    """
    create dataframe for each db
    :return:
    """
    df_result = pd.DataFrame(columns=['Dataset Name', 'Number of samples', 'Original Number of features',
                                  'Filtering Algorithm', 'Learning algorithm', 'Number of features selected (K)',
                                  'CV Method', 'Fold', 'Measure Type', 'Measure Value',
                                  'List of Selected Features Names', 'Selected Features scores'], dtype=object)
    return df_result


def print_best(df, db_name):
    """
    Find and print the configuration used to give the best AUC score for that db
    :param df: result df
    :param db_name: db name
    :return:
    """
    auc_df = df.loc[df['Measure Type'] == 'AUC']
    auc_df = auc_df.loc[auc_df['Measure Value'].idxmax()]
    print(f"{db_name} --> Best Configuration ----- \n\t\tFiltering Algorithm: {auc_df['Filtering Algorithm']}"
          f"\n\t\tK: {auc_df['Number of features selected (K)']}"
          f"\n\t\tLearning algorithm: {auc_df['Learning algorithm']}"
          f"\n\t\tAUC Score: {auc_df['Measure Value']}")
    df = aug.augment_db(df, db_name, auc_df['Filtering Algorithm'], auc_df['Number of features selected (K)'], auc_df['Learning algorithm'])
    return df

def save_database(X, y, db_name, X_cols, X_idx):
    """
    Save the data after pre-processing stage
    :param X:
    :param y:
    :param db_name:
    :param X_cols: feature names
    :param X_idx: data index
    :return:
    """
    path_pre = './data_process/'
    columns = list(X_cols)
    columns.extend(["Class"])
    y_reshape = y.reshape((-1, 1))
    an_array = np.append(X, y_reshape, axis=1)
    df = pd.DataFrame(data=an_array, columns=columns, index=X_idx, dtype=object)
    df.to_csv(path_pre + db_name + ".csv")


def save_result(df, db_name):
    """
    save to disk the dataframe of the result (csv format)
    :param df: df of result for the current db
    :param db_name: db name
    :return:
    """
    df = print_best(df, db_name)
    df.to_csv(path+f'{db_name}.csv')


def get_feature_names_by_idx(col_names, idx):
    return col_names[idx]


def write_result(df, db_name, sample_num, org_features, reduce_method, k, reduce_time, features_idx, score_features, models_scores):
    """
    uterate over the dictionary of result and write them to the df
    :param df: the dataframe contains the result so far
    :param db_name: db name
    :param sample_num: number of sample in db
    :param org_features: feature name before reducing
    :param reduce_method: the current filter method
    :param k: the current k
    :param reduce_time: time take for filter method
    :param features_idx: the selected feature indexes
    :param score_features: the score of the selected features
    :param models_scores: the evaluation of the ML methods
    :return:
    """
    features_num = len(org_features)
    cv_method = models_scores[list(models_scores)[0]]['cv']
    folds = models_scores[list(models_scores)[0]]['folds']
    selected_features = get_feature_names_by_idx(org_features, features_idx).tolist()
    selected_features = [str(i) for i in selected_features]
    selected_features = ';'.join(selected_features)
    score_features = [str(i) for i in score_features]
    score_features = ';'.join(score_features)
    if k == 100:
        df.loc[len(df)] = [db_name, sample_num, features_num, reduce_method, "---", k, cv_method, folds, 'reduce_time',
                        reduce_time, selected_features, score_features]
    for clf in models_scores:
        for metric in models_scores[clf]:
            if metric not in ['cv', 'folds']:
                df.loc[len(df)] = [db_name, sample_num, features_num, reduce_method, clf, k, cv_method, folds, metric,
                                  models_scores[clf][metric], selected_features, score_features]
    return df

