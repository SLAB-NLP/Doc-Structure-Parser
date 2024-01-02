from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm
import numpy as np
import re
from parser.run_louvain_algorithm import normalize_ws, get_weighted_similarities


numbering_regex = r'\d+\.?\s*'
letter_regex = r'^\(?\s*\w{1,3}\s*(\.|\))\s*'


def clear_representative(representative):
    if pd.isna(representative):
        return "NA"
    updated_title = re.sub(numbering_regex, '', representative)
    updated_title = re.sub('item', '', updated_title.lower())
    updated_title = re.sub('section', '', updated_title)
    updated_title = re.sub(r'\W+', ' ', updated_title).strip()
    return updated_title.lower().title()


def find_closest_vector_index(similarities, valid_indices):
    avg_similarity = np.mean(similarities, axis=1)
    closest_index_from_valid = np.argmax(avg_similarity[valid_indices])
    global_closest_index = valid_indices[closest_index_from_valid]
    similarity_to_center = similarities[:, global_closest_index]
    return global_closest_index, similarity_to_center


def add_representatives_to_csv(df, similarities):
    group_by_community = df.groupby("community").groups
    for comm in tqdm(group_by_community):
        only_community_df = df.loc[group_by_community[comm]]
        similarities_community = similarities[group_by_community[comm]][:, group_by_community[comm]]
        text_community = only_community_df["title_text"].values
        cleared = [clear_representative(text) for text in text_community]
        lengths = [len(clr) for clr in cleared]
        indices = np.where(lengths > np.full_like(lengths,fill_value=4))[0]
        if len(indices) > 0:
            closest_index, similarity_to_repr = find_closest_vector_index(similarities_community, indices)
            representative = cleared[closest_index]
        else:
            # if all titles are numbers, this is not a good cluster
            df.loc[group_by_community[comm], "community"] = -1
            representative = text_community[0]
            closest_index = similarity_to_repr = 0
        df.loc[group_by_community[comm], "representative"] = representative
        df.loc[group_by_community[comm], "original_repr"] = text_community[closest_index]
        df.loc[group_by_community[comm], "similarity_to_repr"] = similarity_to_repr

    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--csv_communities")
    parser.add_argument("--w_title", type=float, default=0)
    parser.add_argument("--w_text", type=float, default=0)
    parser.add_argument("--w_index", type=float, default=0)
    parser.add_argument("--similarity_dir", help="path to directory similarity matrices")

    args = parser.parse_args()
    args = normalize_ws(args)

    comm_df = pd.read_csv(args.csv_communities)
    comm_df['original_repr'] = np.full(shape=len(comm_df), fill_value=np.nan)
    comm_df['representative'] = np.full(shape=len(comm_df), fill_value=np.nan)
    comm_df['similarity_to_repr'] = np.full(shape=len(comm_df), fill_value=np.nan)
    weighted_similarities = get_weighted_similarities(args.similarity_dir, args.w_title, args.w_text, args.w_index)
    with_rep_pd = add_representatives_to_csv(comm_df, weighted_similarities)
    with_rep_pd.to_csv(args.csv_communities, index=False)

