import pandas as pd
import pickle
import numpy as np

# path_repo = 'D:\My Drive\Pessoal\Projetos\insurance_sales_predict\src' # local
path_repo = '' # cloud

class Insurance_all(object):
    def __init__(self):
        self.ss_annual_premium = pickle.load(open(path_repo + r'features/ss_annual_premium.pkl', 'rb'))
        self.mm_age = pickle.load(open(path_repo + r'features/mm_age.pkl', 'rb'))
        self.map_vehicle_age = pickle.load(open(path_repo + r'features/map_vehicle_age.pkl', 'rb'))
        self.map_gender = pickle.load(open(path_repo + r'features/map_gender.pkl', 'rb'))
        self.map_vehicle_damage = pickle.load(open(path_repo + r'features/map_vehicle_damage.pkl', 'rb'))
        self.map_region_code = pickle.load(open(path_repo + r'features/map_region_code.pkl', 'rb'))
        self.map_policy_sales_channel = pickle.load(open(path_repo + r'features/map_policy_sales_channel.pkl', 'rb'))
        self.features_drop = pickle.load(open(path_repo + r'features/features_drop.pkl', 'rb'))
        self.vintage_cicle = pickle.load(open(path_repo + r'features/vintage_cicle.pkl', 'rb'))
        self.model = pickle.load(open(path_repo + r'models/model.pkl', 'rb'))

    def data_cleaning(self, df_raw):
        # lowercase columns
        df = df_raw.copy()
        cols_lowercase = ' '.join(df.columns.to_list()).lower().split()
        df.columns = cols_lowercase
        return df


    def feature_engineering(self, df):
        return df


    def data_preparation(self, df):
        # annual_premium
        df['annual_premium'] = self.ss_annual_premium.transform(df[['annual_premium']].values)

        # age by the method minmaxscaler
        df['age'] = self.mm_age.transform(df[['age']].values)

        # vehicle_age by the method labelencoding
        df['vehicle_age'] = df['vehicle_age'].map(self.map_vehicle_age)

        # gender
        df['gender'] = df['gender'].map(self.map_gender)

        # vehicle_damage
        df['vehicle_damage'] = df['vehicle_damage'].map(self.map_vehicle_damage)

        # region_code by the method target_encoding
        df['region_code'] = df['region_code'].map(self.map_region_code)

        # policy_sales_channel
        df['policy_sales_channel'] = df['policy_sales_channel'].map(self.map_policy_sales_channel)

        # vintage
        df['vintage_sin'] = df['vintage'].apply(lambda x: np.sin(x* (2*np.pi/self.vintage_cicle)))
        df['vintage_cos'] = df['vintage'].apply(lambda x: np.cos(x* (2*np.pi/self.vintage_cicle)))
        df.drop('vintage', axis=1, inplace=True)

        # return df
        return df


    def data_filtering(self, df):
        # return df
        return df


    def feature_selection(self, df):
        df.drop(self.features_drop, axis=1, inplace=True)
        # return df
        return df


    def get_predictions(self, df, df_raw):
        # predicting proba
        df1 = df_raw.copy()
        try:
            df = df.drop(['response', 'id'], axis=1)
        except:
            df = df.drop('id', axis=1)

        df1['propensity'] = self.model.predict_proba(df.values)[:,1]

        # return df
        return df1

