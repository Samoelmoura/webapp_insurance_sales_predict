import pandas as pd
import os
from flask import Flask, request, Response
from insurance_all import Insurance_all

app = Flask(__name__)
@app.route('/propensity', methods=['POST'])

def insurance_predict():
    df_raw_json = request.get_json()

    if df_raw_json:
        if isinstance(df_raw_json, dict):
            df_raw = pd.DataFrame(df_raw_json, index=[0])

        else:
            df_raw = pd.DataFrame(df_raw_json, columns=df_raw_json[0].keys())

        #instanciate Insurance_all class
        papeline = Insurance_all()

        # data cleaning
        df = papeline.data_cleaning(df_raw)

        # feature engineering
        df = papeline.feature_engineering(df)

        # data filtering
        df = papeline.data_filtering(df)

        # data preparation
        df = papeline.data_preparation(df)

        # feature selection
        df = papeline.feature_selection(df)

        # get_predictions
        df = papeline.get_predictions(df, df_raw)

        # converting to json
        df_json = df.to_json(orient='records')

        return df_json

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    # port = 5000 # local port
    port = os.environ.get('PORT', 5000) # cloud port
    app.run('0.0.0.0', port=port)