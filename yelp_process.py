import json
import pandas as pd
import ast
from flatten_json import flatten
import re
import numpy as np
from os.path import abspath, dirname


def parser_business(line):
    data = json.loads(line)
    data = flatten(data)

    return data


def parser_amb(line):
    if isinstance(line, str) and line != 'None':
        line = ast.literal_eval(line)
    else:
        line = {'amb_empty': 1}

    return line


def businesses_to_csv(business_data_path, datadir):
    with open(business_data_path, encoding="utf8") as file:
        data = [parser_business(line) for line in file]

    df = pd.DataFrame(data)

    df['rest'] = df.categories.apply(lambda x: 1 if x is not None and 'Restaurants' in x else 0)
    df_rest = df[df.rest == 1]
    df_rest = df_rest[df_rest.review_count > 20]

    rest_columns = ['business_id', 'state', 'latitude', 'longitude', 'stars', 'review_count', 'attributes_BusinessAcceptsCreditCards',
                    'attributes_RestaurantsPriceRange2', 'attributes_GoodForKids', 'attributes_Ambience',
                    'attributes_Alcohol', 'attributes_RestaurantsReservations', 'attributes_NoiseLevel']

    df_rest = df_rest[rest_columns]
    df_rest.dropna(inplace=True)
    # states = df_rest.state.value_counts()

    df_rest = df_rest[df_rest.state == 'AZ']

    df_rest.to_csv(datadir + 'business_AZ.csv')


def process_businesses_csv(business_AZ_df_path, datadir):
    df = pd.read_csv(business_AZ_df_path)
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
    amb = pd.DataFrame([parser_amb(x) for x in df.attributes_Ambience])
    df.drop(columns=['attributes_Ambience'], inplace=True)
    df = df.join(amb)
    df = df[df.amb_empty != 1]
    df.drop(columns=['amb_empty', 'state'], inplace=True)
    binary_cols = ['attributes_GoodForKids', 'attributes_BusinessAcceptsCreditCards', 'attributes_RestaurantsReservations',
                   'romantic', 'intimate', 'classy', 'hipster', 'divey', 'touristy', 'trendy', 'upscale', 'casual']
    for col in binary_cols:
        df[col] = df[col].apply(lambda x: 1 if x else 0)

    df.replace('None', np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"len business df: {len(df)}")

    df.attributes_NoiseLevel = df.attributes_NoiseLevel.apply(lambda x: re.findall('\'.+\'', x)[0][1:-1])
    df.attributes_Alcohol = df.attributes_Alcohol.apply(lambda x: re.findall('\'.+\'', x)[0][1:-1])

    df.to_csv(datadir + 'business_filtered_AZ.csv')


def extract_stars(review_data_path, business_filtered_AZ_df_path, datadir):
    df_rest = pd.read_csv(business_filtered_AZ_df_path)
    stars_df = pd.DataFrame()
    for chunk in pd.read_json(review_data_path, lines=True, chunksize=100000, nrows=10000000):
        df_chunk = chunk[chunk.business_id.isin(df_rest.business_id)]
        stars_df = stars_df.append(df_chunk[['user_id', 'business_id', 'stars']])

    stars_df.to_csv(datadir + 'stars_df_AZ.csv')


def extract_users(user_data_path, stars_df_path, datadir):
    stars_df = pd.read_csv(stars_df_path)
    users_df = pd.DataFrame()
    for chunk in pd.read_json(user_data_path, lines=True, chunksize=100000, nrows=10000000):
        users_df = users_df.append(chunk[chunk.user_id.isin(stars_df.user_id) & (chunk.review_count > 30)])

    users_df.to_csv(datadir + 'users_df_AZ.csv')


if __name__ == "__main__":
    homedir = dirname(dirname(abspath(__file__)))
    datadir = homedir + "/datasets/yelp/"

    business_data_path = datadir + 'yelp_academic_dataset_business.json'
    review_data_path = datadir + 'yelp_academic_dataset_review.json'
    user_data_path = datadir + 'yelp_academic_dataset_user.json'

    businesses_to_csv(business_data_path, datadir)

    business_AZ_df_path = datadir + 'business_AZ.csv'
    process_businesses_csv(business_AZ_df_path, datadir)

    business_filtered_AZ_df_path = datadir + 'business_filtered_AZ.csv'
    extract_stars(review_data_path, business_filtered_AZ_df_path, datadir)

    stars_df_path = datadir + 'stars_df_AZ.csv'
    extract_users(user_data_path, stars_df_path, datadir)
