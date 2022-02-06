"""
Requires feature csv to have LATITUDE, LONGITUDE, address columns
"""

import argparse
import glob
import os
import logging

import pandas as pd
import geopandas as gpd
from pandarallel import pandarallel
import numpy as np
from scipy.spatial import cKDTree

pandarallel.initialize(nb_workers=16)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def ckdnearest(gdA, gdB):

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--df_folder', type=str)
    parser.add_argument('--feature_folder', type=str)
    parser.add_argument('--out_folder', type=str)
    parser.add_argument('--is_gdf', action='store_true', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    df_paths = [i for i in glob.glob(
        os.path.join(args.df_folder, "*.parquet"), recursive=True)]
    feature_paths = [i for i in glob.glob(
        os.path.join(args.feature_folder, "*.csv"), recursive=True)]
    
    for i, df_path in enumerate(df_paths):
        filename = os.path.basename(df_path).split('.')[0]
        
        if not args.is_gdf:
            df = pd.read_parquet(df_path)
            gdf = gpd.GeoDataFrame(df, 
                geometry=gpd.points_from_xy(df['rawlat'], df['rawlng']))
            logging.info(f"generated gpd for {df_path}")
        #     gdf.to_file(f'{filename}_gdf.shp')
        # else:
        #     gdf = gpd.read_file(df_path)
        
        for j, feature_path in enumerate(feature_paths):
            feat_name = os.path.basename(feature_path).split('_')[0]
            feature_df = pd.read_csv(feature_path)[['address', 'LATITUDE', 'LONGITUDE']]
            feature_gdf = gpd.GeoDataFrame(feature_df, 
                geometry=gpd.points_from_xy(feature_df.LATITUDE, feature_df.LONGITUDE))

            gdf = ckdnearest(gdf, feature_gdf)
            gdf = gdf.rename(columns={'address': f'{feat_name}',
                                      'LATITUDE': f'{feat_name}_lat', 
                                      'LONGITUDE': f'{feat_name}_lng', 
                                      'dist': f'{feat_name}_dst'})
            gdf[f'{feat_name}_dst'] = gdf[f'{feat_name}_dst'] * 111139
            logging.info(f"finish appending {feat_name} for {df_path}")
        
        gdf = gdf.drop(['geometry'], axis=1)
        gdf.to_parquet(f'{args.out_folder}/{filename}_new.parquet')
        logging.info(f'Written file {args.out_folder}/{filename}_new.parquet, {i+1}/{len(df_paths)}')
