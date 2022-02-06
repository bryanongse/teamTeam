import argparse
import glob
import os
import logging

import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt

CHUNKS = 1000
ICHUNKS = 1000j

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def get_interpolate_fn(geodf: pd.DataFrame,
                       geofeature: pd.DataFrame,
                       feature_name: str):

    minlng, maxlng = geodf['rawlng'].min(), geodf['rawlng'].max()
    minlat, maxlat = geodf['rawlat'].min(), geodf['rawlat'].max()
    print('lats limits', minlng, maxlng)
    print('lngs limits', minlat, maxlat)
    grid_x, grid_y = np.mgrid[minlng:maxlng:ICHUNKS, minlat:maxlat:ICHUNKS]
    grid = griddata(geofeature[['longitude', 'latitude']], 
                    geofeature[[feature_name]], 
                    (grid_x, grid_y), method='linear')
    
    x_idx = np.linspace(minlng, maxlng, num=CHUNKS)
    y_idx = np.linspace(minlat, maxlat, num=CHUNKS)
    f = RegularGridInterpolator((x_idx, y_idx), grid, method='linear')
    
    # np.savetxt('grid.csv', grid.squeeze())
    return grid, f


def generate_plot(grid,
                  geodf: pd.DataFrame,
                  out_path: str):
    minlng, maxlng = geodf['rawlng'].min(), geodf['rawlng'].max()
    minlat, maxlat = geodf['rawlat'].min(), geodf['rawlat'].max()
    plt.figure(figsize=(10,10))
    plt.imshow(grid, extent=(minlng, maxlng, minlat, maxlat))
    plt.savefig(out_path)


def get_feature(geodf: pd.DataFrame,
                f):
    
    def interpolate_feature(x, f):
        return f((x[1], x[0]))[0]
    
    return df[['rawlat', 'rawlng']].apply(interpolate_feature, args=(f,), axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--df_folder', type=str)
    parser.add_argument('--feature_csv', type=str)
    parser.add_argument('--out_folder', type=str)
    parser.add_argument('--feat_name', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    feature_df = pd.read_csv(args.feature_csv)
    df_paths = [i for i in glob.glob(
        os.path.join(args.df_folder, "*.parquet"), recursive=True)]
    
    for i, df_path in enumerate(df_paths):
        df = pd.read_parquet(df_path)
        grid, f = get_interpolate_fn(df, feature_df, feature_name=args.feat_name)
        generate_plot(grid, df, out_path=f'{args.out_folder}/{df_path}_heatmap.png')

        new_feature = get_feature(df, f)
        df[args.feat_name] = new_feature
        df.to_parquet(f'{args.out_folder}/{df_path}_new.parquet')
