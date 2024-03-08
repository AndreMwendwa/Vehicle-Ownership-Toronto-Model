import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import geopandas as gpd
import os
import yaml
import pickle as pkl
from numba import jit

os.chdir(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\IATBR\PT_stations')
from functools import partial

# # Read data from pickled file
data = pd.read_pickle(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\Sign-Up_ordered_wsltur2.pkl')

data_orig = pd.read_excel(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\Roschlau Fellowship\THATS Survey Outputs\1. Collected Data\Sign-Up and Daily Surveys\Sign-Up Survey_September 1_modified.xlsx')
#
# # Open shapefile of POIs
# poi = gpd.read_file(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\IATBR\OSM Analysis\data_unzipped\gis_osm_pois_a_free_1.shp')
#
yaml_file = open(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\Completeness_factors_list.yml', 'r')
yaml_content = yaml.load(yaml_file, Loader=yaml.FullLoader)
#
# poi['centroids'] = poi.centroid
#
# Prepare a list of all values from the yaml file whose value is not None
yaml_list = []
for key, value in yaml_content.items():
    if value is not None:
        yaml_list.append(value)
yaml_list = [item for sublist in yaml_list for item in sublist]
#
# # Filtering out the poi's that aren't of a type that interests us, to reduce the size of the dataframe we're iterating through
# poi = poi[poi.fclass.isin(yaml_list)]




data['LocationLongitude'] = data_orig['LocationLongitude'].copy()
data['LocationLatitude'] = data_orig['LocationLatitude'].copy()

# I'm going to try a numba version of this function, because the python version is too slow (about 30 * 14 minutes by my reckoning)

# For each item in yaml_list, create a new column in data that contains the distance to the nearest POI of that type
df = data[data['LocationLongitude'].notnull()]
#
# # Pickling to save time
# df.to_pickle(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\df_persons_wsltur.pkl')
# poi.to_pickle(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\poi.pkl')

# # Unpickling to save time
# df = pd.read_pickle(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\df_persons_wsltur.pkl')
poi = pd.read_pickle(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\poi.pkl')

# Filtering the POIs to only those in the gtha
gtha = gpd.read_file(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\IATBR\OSM Analysis\GTHA_Boundary-shp\GTHA_Boundary.shp')

gtha_bbox = gtha.total_bounds

poi = poi[(poi.centroids.x > gtha_bbox[0]) & (poi.centroids.x < gtha_bbox[2]) & (poi.centroids.y > gtha_bbox[1]) & (poi.centroids.y < gtha_bbox[3])]

# Function that calculates the distance between two points using the Haversine formula
@jit(nopython=True)
def haversine(lon1, lat1, lon2, lat2):
    # Converting the coordinates to radians
    # lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])

    # Calculating the distance
    dlon = np.radians(lon2) - np.radians(lon1)
    dlat = np.radians(lat2) - np.radians(lat1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    km = 6367 * c

    return km

# df = df.sample(5)


# Add columns to poi for the latitude and longitude of the centroid
poi['latitude'] = poi.centroids.y
poi['longitude'] = poi.centroids.x

@jit(nopython=True)
def minimum_distance(np_arr: np.array, centroids: np.array):
    for i in range(np_arr.shape[0]):

        distances = np.zeros(centroids.shape[0])
        for j in range(centroids.shape[0]):
            dist = haversine(np_arr[i, 1], np_arr[i, 2], centroids[j, 1], centroids[j, 0])
            distances[j] = dist
        np_arr[i, 3] = np.min(distances)
        # print(distances)
        #     if dist < min_dist:
        #         min_dist = dist
        # np_arr[i, 3] = min_dist
    return np_arr

for item in yaml_list:
    poi_type = poi[poi.fclass == item]
    # poi_type = poi_type.sample(3)
    # print(poi_type)
    poi_type = poi_type.reset_index(drop=True)
    df[f'distance_nearest_{item}'] = np.nan
    persons = df.reset_index()[['indivID', 'LocationLongitude', 'LocationLatitude', f'distance_nearest_{item}']].to_numpy()
    poi_centroids = poi_type[['latitude', 'longitude']].to_numpy()
    tmp = pd.DataFrame(minimum_distance(persons, poi_centroids), columns=['indivID', 'LocationLongitude', 'LocationLatitude', f'distance_nearest_{item}'])
    # print('tmp= {}'.format(tmp))

    df = pd.merge(df, tmp[['indivID', f'distance_nearest_{item}']], left_index=True, right_on='indivID', how='left')
    df.drop(columns=[f'distance_nearest_{item}_x'], inplace=True)
    df.drop(columns=['indivID'], inplace=True)
    df.rename(columns={f'distance_nearest_{item}_y': f'distance_nearest_{item}'}, inplace=True)
    df.index.name = 'indivID'
    print('df= {}'.format(df.head()))

# Pickle df
df.to_pickle(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\df_persons_wsltur_with_distances09_12.pkl')


# item = 'mall'
# poi_type = poi[poi.fclass == item]
# poi_type = poi_type.sample(3)
# print(poi_type)
# poi_type = poi_type.reset_index(drop=True)
# df[f'distance_nearest_{item}'] = np.nan
# persons = df.reset_index()[['indivID', 'LocationLongitude', 'LocationLatitude', f'distance_nearest_{item}']].to_numpy()
# poi_centroids = poi_type[['latitude', 'longitude']].to_numpy()
# tmp = pd.DataFrame(minimum_distance(persons, poi_centroids), columns=['indivID', 'LocationLongitude', 'LocationLatitude', f'distance_nearest_{item}'])
# print('tmp= {}'.format(tmp))
#
# df = pd.merge(df, tmp[['indivID', f'distance_nearest_{item}']], left_index=True, right_on='indivID', how='left')
# df.drop(columns=[f'distance_nearest_{item}_x'], inplace=True)
# df.rename(columns={f'distance_nearest_{item}_y': f'distance_nearest_{item}'}, inplace=True)
# print('df= {}'.format(df))
# # df[f'distance_nearest_{item}'] = minimum_distance(persons, poi_centroids)
# #
# # print(df[f'distance_nearest_{item}'])