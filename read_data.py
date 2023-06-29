from scipy.io import arff
import pandas as pd
import scipy.io


# path to data folders
ARFF_PATH = './data/ARFF/'
BIO_PATH = './data/bioconductor/'
MISC_PATH = './data//Misc/'
SCIKIT_PATH = './data//scikit-feature datasets/'


def read_arff_files(data_name):
    """ load files from ARFF folder(arff format), split into X,y"""
    path = ARFF_PATH + data_name + '.arff'
    try:
        data = arff.loadarff(path)
        df = pd.DataFrame(data[0])
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    except Exception as e:
        print(e)
        return e
    return X, y


def read_scikit_files(data_name):
    """ load files from scikit folder (mat format), split into X,y"""
    path = SCIKIT_PATH + data_name + '.mat'
    try:
        mat = scipy.io.loadmat(path)
        y = mat['Y']
        X = mat['X']
    except Exception as e:
        print(e)
        return e
    X = pd.DataFrame(X)
    y = pd.DataFrame(y).iloc[:, 0]
    return X, y


def read_misc_files(data_name):
    """ load files from Mics folder (csv format), split into X,y"""
    path = MISC_PATH + data_name + '.csv'
    try:
        if data_name != 'GDS4824':
            df = pd.read_csv(path, index_col=0)
            df = df.transpose()
        else:
            df = pd.read_csv(path)
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    except Exception as e:
        print(e)
        return e
    return X, y


def read_bio_files(data_name):
    """ load files from bioconductor folder (csv format), split into X,y"""
    path = BIO_PATH + data_name + '.csv'
    try:
        df = pd.read_csv(path, index_col=0)
        df = df.transpose()
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    except Exception as e:
        print(e)
        return e
    return X, y


# map name of data to the function that read the data
data_name_to_function = {'CNS': read_arff_files, 'Lymphoma': read_arff_files, 'MLL': read_arff_files,
                         'Ovarian': read_arff_files, 'SRBCT': read_arff_files, 'ayeastCC': read_bio_files,
                         'bladderbatch': read_bio_files, 'CLL': read_bio_files, 'ALL': read_bio_files,
                         'leukemiasEset': read_bio_files, 'GDS4824': read_misc_files, 'khan_train': read_misc_files,
                         'ProstateCancer': read_misc_files, 'Nutt-2003-v2_BrainCancer.xlsx - Sayfa1': read_misc_files,
                         'Risinger_Endometrial Cancer.xlsx - Sayfa1': read_misc_files, 'madelon': read_scikit_files,
                         'ORL': read_scikit_files, 'Carcinom': read_scikit_files, 'USPS': read_scikit_files,
                         'Yale': read_scikit_files}

def read_data(data_name):
    """
    main function
    :param data_name: the data name to read
    :return: X, y data
    """
    if isinstance(data_name, int):
        data_name = list(data_name_to_function)[data_name - 1]
    X, y = data_name_to_function[data_name](data_name)
    return data_name, X, y
