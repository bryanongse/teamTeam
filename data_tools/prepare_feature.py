import argparse
import glob
import os
import logging

import pandas as pd
from pandarallel import pandarallel
from scipy.interpolate import griddata, RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt

CHUNKS = 1000
ICHUNKS = 1000j

pandarallel.initialize(nb_workers=16)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def get_interpolate_fn(geodf: pd.DataFrame,
                       geofeature: pd.DataFrame,
                       feature_name: str):

    minlng, maxlng = geodf['rawlng'].min(), geodf['rawlng'].max()
    minlat, maxlat = geodf['rawlat'].min(), geodf['rawlat'].max()
    logging.info(f'lats limits {minlng} {maxlng}')
    logging.info(f'lngs limits {minlat} {maxlat}')
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


def get_feature(df: pd.DataFrame, f):
    fn = lambda x: f((x[1], x[0]))[0]
    res = df[['rawlat', 'rawlng']].parallel_apply(fn, axis=1)
    return res


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

        filename = os.path.basename(df_path)
        generate_plot(grid, df, out_path=f'{args.out_folder}/{filename}_heatmap.png')

        new_feature = get_feature(df, f)
        df[args.feat_name] = new_feature
        df.to_parquet(f'{args.out_folder}/{filename}_new.parquet')
        logging.info(f'Written file {args.out_folder}/{filename}_new.parquet, {i+1}/{len(df_paths)}')
