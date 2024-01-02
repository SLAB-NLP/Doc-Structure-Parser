
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from tqdm import tqdm
from utility_scripts.add_clusters_doc_cover_rank import add_rank_to_df


def split_to_ranges(lst):
    ranges = []
    start = end = None
    start_idx = end_idx = None

    for idx, num in enumerate(lst):
        if start is None:
            start = num
            end = num
            start_idx = end_idx = idx
        elif num == end + 1:
            end = num
            end_idx = idx
        else:
            ranges.append((start_idx, end_idx))
            start = num
            end = num
            start_idx = end_idx = idx

    # Append the last range
    ranges.append((start_idx, end_idx))

    return ranges


def separate_communities_from_duplicates(dataframe):

    group_by_community = dataframe.groupby("community").groups
    for community in tqdm(group_by_community, desc="communities"):
        if community == -1:
            continue
        df_community = dataframe.loc[group_by_community[community]]
        indices_to_remove, indices_to_keep = filter_outliers(df_community)
        dataframe.loc[group_by_community[community][indices_to_remove], "community"] = -1
        filtered_df_community = dataframe.loc[group_by_community[community][indices_to_keep]]
        group_by_filename = filtered_df_community.groupby("filename").groups
        for filename in group_by_filename:
            community_file = filtered_df_community.loc[group_by_filename[filename]]
            community_file = community_file.sort_values(by=['title_index'])
            title_index = community_file['title_index']
            ranges_same_file = split_to_ranges(title_index.tolist())
            if len(ranges_same_file) == 1:
                continue
            remove_duplicates_with_low_similarity(dataframe, ranges_same_file, title_index)

    return dataframe


def filter_outliers(community_file):
    # filter out from cluster the 15% most distant nodes from the representative
    similarity_to_repr = community_file["similarity_to_repr"].values
    arg_sorted_by_sim = np.argsort(similarity_to_repr)
    to_remove, to_keep = arg_sorted_by_sim[:int(len(similarity_to_repr) * 0.15)], arg_sorted_by_sim[int(len(similarity_to_repr) * 0.15):]
    return to_remove, to_keep


def remove_duplicates_with_low_similarity(dataframe, ranges_same_file, title_index):
    max_sim = -np.inf
    best_range = None

    # find the best range
    for single_range in ranges_same_file:
        indices = title_index.iloc[single_range[0]:single_range[1] + 1].index
        current_max_sim = np.max(dataframe.loc[indices]["similarity_to_repr"].values)
        if current_max_sim > max_sim:
            max_sim = current_max_sim
            best_range = single_range

    # assign -1 to any other range that is not the best
    for range_to_remove in ranges_same_file:
        if range_to_remove == best_range:
            continue
        indices = title_index.iloc[range_to_remove[0]:range_to_remove[1] + 1].index
        dataframe.loc[indices, "community"] = -1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--df_path")
    args = parser.parse_args()

    meta_csv = pd.read_csv(args.df_path, index_col=False)
    processed_df = separate_communities_from_duplicates(meta_csv)
    out_path = args.df_path.replace('.csv', '_filtered.csv')
    processed_df.to_csv(out_path)

    add_rank_to_df(out_path)

