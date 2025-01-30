import fairtl_emd as fl
import pandas as pd
import numpy as np
import re 
import kagglehub
import os
from ucimlrepo import fetch_ucirepo 
from ucimlrepo import DatasetNotFoundError 

def is_csv(path):
    _, ext = os.path.splitext(path)
    return ext.lower() == '.csv'


def find_kaggle_url(df, pattern='datasets/'):

    pattern = fr'(?<={pattern})(.*)'  # datasets/  

    urls = []
    for url in df['URL']:
        match = re.search(pattern, url)

        if not match:
            print(f'{url} match fail')
        else:
            urls.append(match.group())

    df['kaggle_download_url'] = urls
    return df


def kaggle_test(df, pattern='datasets/'):

    df = find_kaggle_url(df, pattern)

    df['kaggle_status'] = ''

    for row in df.itertuples():

        if row.is_skip == 'skip':
            continue

        kaggle_url = row.kaggle_download_url
        path = kagglehub.dataset_download(kaggle_url)

        # files = os.listdir(path)

        if len(os.listdir(path)) != 1:
            continue

        if not is_csv(os.listdir(path)[0]):
            continue

        kaggle_df = pd.read_csv(f'{path}/{os.listdir(path)[0]}')

        '----------------------------------------'

        matched = False

        for col in kaggle_df.columns:

            if col.lower() in ['sex', 'gender', 'male', 'female']:
                kaggle_df.rename(columns={col:'Gender'}, inplace=True)

                df.at[row.Index, 'kaggle_status'] = 'find gender'
                matched = True
                break

        if not matched:
            df.at[row.Index, 'kaggle_status'] = 'fail find gender'
            continue

        protected = kaggle_df['Gender'].unique()

        if len(protected) != 2:
            df.at[row.Index, 'kaggle_status'] = 'above 2 gender'
            continue

        found = False

        for i in protected:
            if str(i).lower() in ["0", "f", "female", "woman", "w", "girl", "2"]:

                kaggle_df['Gender'] = np.where(kaggle_df['Gender'] == i, 'Female', 'Male')
                found = True
                break

        if not found:
            continue

        protected = kaggle_df['Gender'].unique()

        if row.target is None:
            df.at[row.Index, 'kaggle_status'] = 'lack target attr'
            continue

        for attr in protected:

            if attr == 'Female':
                df.at[row.Index, 'EMD_female'], df.at[row.Index, 'EMD_fem_p_value'], _ = fl.EMD(kaggle_df, 'Gender', row.target, attr, 1000)

            elif attr == 'Male':
                df.at[row.Index, 'EMD_male'], df.at[row.Index, 'EMD_male_p_value'], _ = fl.EMD(kaggle_df, 'Gender', row.target, attr, 1000)

        df.at[row.Index, 'Instances'] = kaggle_df.shape[0]
        df.at[row.Index, 'Features'] = kaggle_df.shape[1]

    return df
                


def find_uci_url(url):

    match = re.search(r"/(\d+)/", url)
    if match:
        number = match.group(1)
        return number
    else:
        return False




def uci_test(df):

    df['uci_status'] = ''

    for row in df.itertuples():

        # print(row.URL)
        
        id = find_uci_url(row.URL)
        if not id:

            df.at[row.Index, 'uci_status'] = 'cannot find id'
            continue
        
        try:
            ds = fetch_ucirepo(id=int(id)) 
        except DatasetNotFoundError:

            df.at[row.Index, 'uci_status'] = 'cannot import dataset'
            continue
        

        X = ds.data.features 
        y = ds.data.targets 

        uci_df = pd.DataFrame(X)

        try:
            uci_df['target'] = y
        except ValueError:
                            
            df.at[row.Index, 'uci_status'] = 'above 1 target'
            continue

        matched = False

        for col in uci_df.columns:

            if col.lower() in ['sex', 'gender', 'male', 'female']:
                uci_df.rename(columns={col:'Gender'}, inplace=True)

                df.at[row.Index, 'uci_status'] = 'find gender'
                matched = True
                break

        if not matched:
            df.at[row.Index, 'uci_status'] = 'fail find gender'
            continue

        protected = uci_df['Gender'].unique()

        if len(protected) != 2:
            df.at[row.Index, 'uci_status'] = 'above 2 gender'
            continue

        found = False

        for i in protected:
            if str(i).lower() in ["0", "f", "female", "woman", "w", "girl", "2"]:

                uci_df['Gender'] = np.where(uci_df['Gender'] == i, 'Female', 'Male')
                found = True
                break

        if not found:
            continue

        protected = uci_df['Gender'].unique()

        for attr in protected:

            if attr == 'Female':
                df.at[row.Index, 'EMD_female'], df.at[row.Index, 'EMD_fem_p_value'], _ = fl.EMD(uci_df, 'Gender', 'target', attr, 1000)

            elif attr == 'Male':
                df.at[row.Index, 'EMD_male'], df.at[row.Index, 'EMD_male_p_value'], _ = fl.EMD(uci_df, 'Gender', 'target', attr, 1000)

        instances = uci_df.shape[0]
        features = uci_df.shape[1]
        df.at[row.Index, 'Instances'] = int(instances)
        df.at[row.Index, 'Features'] = int(features)

    return df

 





