"""
Process GPS ping data into trip information data
"""
import os
import glob
import argparse
import logging

import pandas as pd
import geopy.distance
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=16)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def read_gps_ping_data(path: str,
                       timestamp_name: str = 'pingtimestamp'):
    df = pd.read_parquet(path).dropna()
    df[timestamp_name] = pd.to_datetime(df[timestamp_name], unit='s')
    df = df.sort_values([timestamp_name], ascending=True)
    df = df.reset_index()
    return df


def plot_feature_dist(ping_df: pd.DataFrame,
                      out_path: str):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    ping_df[['population_2020']].plot.hist(bins=100).figure.savefig(f'{out_path}/pop_dist.png')
    ping_df[['real_value']].plot.hist(bins=100).figure.savefig(f'{out_path}/col_dist.png')
    ping_df[['school_dst']].plot.hist(bins=100).figure.savefig(f'{out_path}/school_dist.png')
    ping_df[['supermarket_dst']].plot.hist(bins=100).figure.savefig(f'{out_path}/supermarket_dist.png')
    ping_df[['MRT_dst']].plot.hist(bins=100).figure.savefig(f'{out_path}/mrt_dist.png')
    ping_df[['parks_dst']].plot.hist(bins=100).figure.savefig(f'{out_path}/parks_dist.png')
    ping_df[['hawker_dst']].plot.hist(bins=100).figure.savefig(f'{out_path}/hawker_dist.png')
    ping_df[['shoppingmall_dst']].plot.hist(bins=100).figure.savefig(f'{out_path}/mall_dist.png')


def get_intermediate_feature(df: pd.DataFrame):
    df['is_crowded'] = df['population_2020'] > df['population_2020'].median()
    df['is_real_value'] = df['real_value'] > df['real_value'].median()
    df['is_near_school'] = df['school_dst'] < df['school_dst'].median()
    df['is_near_supermarket'] = df['supermarket_dst'] < df['supermarket_dst'].median()
    df['is_near_mrt'] = df['MRT_dst'] < df['MRT_dst'].median()
    df['is_near_park'] = df['parks_dst'] < df['parks_dst'].median()
    df['is_near_hawker'] = df['hawker_dst'] < df['hawker_dst'].median()
    df['is_near_mall'] = df['shoppingmall_dst'] < df['shoppingmall_dst'].median()

    df['crowded_cost'] = df['is_crowded'] * df['population_2020']
    df['col_cost'] = df['is_real_value'] * df['real_value']
    df['school_cost'] = df['is_near_school'] * df['school_dst']
    df['supermarket_cost'] = df['is_near_supermarket'] * df['supermarket_dst']
    df['mrt_cost'] = df['is_near_mrt'] * df['MRT_dst']
    df['park_cost'] = df['is_near_park'] * df['parks_dst']
    df['hawker_cost'] = df['is_near_mall'] * df['hawker_dst']
    df['mall_cost'] = df['is_near_hawker'] * df['shoppingmall_dst']
    
    return df


def compute_distance(df: pd.DataFrame):
    grouped = df.groupby("trj_id")
    num_in_group = grouped.cumcount()
    df['rawlat_lag'] = grouped['rawlat'].shift()
    df['rawlng_lag'] = grouped['rawlng'].shift()
    
    df_dist = df.loc[num_in_group>0, ['trj_id', 'rawlat', 'rawlng', 'rawlat_lag', 'rawlng_lag']]
    df_dist['dist'] = df_dist.parallel_apply(lambda x: geopy.distance.geodesic(
        (x[3], x[4]), (x[1], x[2])).km, axis=1)
    
    df = df.drop(columns=['rawlat_lag', 'rawlng_lag'], errors='ignore')
    
    return df_dist.groupby('trj_id').aggregate({'dist': 'sum'})


def generate_trip_info(df: pd.DataFrame):

    trip_info = df.groupby('trj_id').agg(
        {'trj_id': 'first', 'driving_mode': 'first', 'osname': 'first'})

    grouped = df.groupby("trj_id")
    trip_info['start_time'] = grouped['pingtimestamp'].aggregate(min)
    trip_info['end_time'] = grouped['pingtimestamp'].aggregate(max)
    trip_info['eta'] = trip_info['end_time'] - trip_info['start_time']
    trip_info['eta'] = trip_info['eta'].dt.seconds.div(60).astype(int) + trip_info['eta'].dt.days.multiply(1440).astype(int)
    trip_info['start_lat'] = grouped['rawlat'].aggregate('first')
    trip_info['end_lat'] = grouped['rawlat'].aggregate('last')
    trip_info['start_lng'] = grouped['rawlng'].aggregate('first')
    trip_info['end_lng'] = grouped['rawlng'].aggregate('last')
    trip_info['time_of_day'] = trip_info['start_time'].dt.hour
    trip_info['day_of_week'] = trip_info['start_time'].dt.weekday
    trip_info['avg_speed'] = grouped['speed'].aggregate('mean')
    trip_info['median_speed'] = grouped['speed'].aggregate('median')

    trip_info['distance'] = compute_distance(df)
    logging.info(f'Computed distanes')

    trip_info['crowded_cost_raw'] = grouped['population_2020'].aggregate('sum') # has 162 na
    trip_info['col_cost_raw'] = grouped['real_value'].aggregate('sum') # has 240725 na
    trip_info['school_cost_raw'] = grouped['school_dst'].aggregate('sum')
    trip_info['supermarket_cost_raw'] = grouped['supermarket_dst'].aggregate('sum')
    trip_info['mrt_cost_raw'] = grouped['MRT_dst'].aggregate('sum')
    trip_info['parks_cost_raw'] = grouped['parks_dst'].aggregate('sum')
    trip_info['hawker_cost_raw'] = grouped['hawker_dst'].aggregate('sum')
    trip_info['mall_cost_raw'] = grouped['shoppingmall_dst'].aggregate('sum')
    
    features = df.groupby('trj_id').agg(
        {'is_crowded': 'mean', 
        'is_real_value': 'mean', 
        'is_near_school': 'mean',
        'is_near_supermarket': 'mean',
        'is_near_mrt': 'mean',
        'is_near_park': 'mean',
        'is_near_hawker': 'mean',
        'is_near_mall': 'mean',
        'crowded_cost': 'sum',
        'col_cost': 'sum',
        'school_cost': 'sum',
        'supermarket_cost': 'sum',
        'mrt_cost': 'sum',
        'park_cost': 'sum',
        'hawker_cost': 'sum',
        'mall_cost': 'sum',
        })

    trip_info = pd.concat((trip_info, features), axis=1)
    return trip_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--df_folder', type=str)
    parser.add_argument('--out_folder', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    df_paths = [i for i in glob.glob(
        os.path.join(args.df_folder, "*.parquet"), recursive=True)]
    
    for i, df_path in enumerate(df_paths):
        filename = os.path.basename(df_path).split('_')[0]
        
        ping_df = read_gps_ping_data(df_path)
        logging.info(f'Read {df_path}')

        plot_feature_dist(ping_df, f'{args.out_folder}/{filename}')
        logging.info(f'Output distribution plots to {args.out_folder}/{filename}')

        ping_df = get_intermediate_feature(ping_df)
        logging.info(f'Generated intermediate features')

        trip_info = generate_trip_info(ping_df)
        logging.info(f'Generated trip info')
        
        trip_info.to_parquet(f'{args.out_folder}/trip_info_{filename}.parquet')
        logging.info(f'Written file {args.out_folder}/trip_info_{filename}.parquet, {i+1}/{len(df_paths)}')
