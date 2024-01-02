
from argparse import ArgumentParser
import os
import numpy as np
import networkx as nx
import pandas as pd
from community import community_louvain
from collections import Counter

DEFAULT_PERCENTILE = 0.998


def get_weighted_similarities(data_dir, w_title, w_text, w_index):
    matrix = {}
    weights = {"title": w_title, "section": w_text, "index": w_index}
    for category in weights:
        sim_path = os.path.join(data_dir, f"{category}_sim.npy")
        matrix[category] = np.load(sim_path) * weights[category]

    weighted_matrix = (matrix["title"]*w_title) + (matrix["section"]*w_text) + (matrix["index"]*w_index)

    return weighted_matrix


def build_graph(similarities_weighted, percentile, nodes_info):
    num_neighbors = int((1 - percentile) * len(similarities_weighted))
    print(f"building graph with edges from {num_neighbors} neighbors with highest similarity")
    print(len(similarities_weighted), "nodes,", num_neighbors*len(similarities_weighted), "edges")
    graph = nx.Graph()
    skipped_non_consecutive_nodes = 0
    for node in range(len(similarities_weighted)):
        node_neighbors = similarities_weighted[node]
        max_neighbors = np.argsort(node_neighbors)[-num_neighbors:-1]
        node_fname = nodes_info.iloc[node]["filename"]
        node_title_index = nodes_info.iloc[node]["title_index"]
        for neighbor in max_neighbors:
            neighbor_fname = nodes_info.iloc[neighbor]["filename"]
            neighbor_title_index = nodes_info.iloc[neighbor]["title_index"]
            if node_fname == neighbor_fname and abs(node_title_index - neighbor_title_index) > 1:
                # do not link nodes from the same file that are not consecutive
                skipped_non_consecutive_nodes += 1
                continue
            graph.add_edge(node, neighbor, weight=node_neighbors[neighbor])
    print(f"truncated {skipped_non_consecutive_nodes} edges from the same file that are not consecutive, which is {100*skipped_non_consecutive_nodes/(num_neighbors*len(similarities_weighted)):.2f}% of edges")
    return graph


def analyze_communities(communities_dict):
    num_communities = len(communities_dict)
    mean = np.mean(list(communities_dict.values()))
    median = np.median(list(communities_dict.values()))
    print(f"{num_communities} communities", f"mean community size = {mean}",
          f"median community size = {median}", sep='\n\t')


def normalize_ws(arguments):
    sum_w = arguments.w_index + arguments.w_title + arguments.w_text
    if sum_w == 0:
        arguments.w_index = arguments.w_title = arguments.w_text = 0.33
    elif sum_w != 1:
        arguments.w_index /= sum_w
        arguments.w_title /= sum_w
        arguments.w_text /= sum_w
    return arguments


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--w_title", type=float, default=0)
    parser.add_argument("--w_text", type=float, default=0)
    parser.add_argument("--w_index", type=float, default=0)
    parser.add_argument("--similarity_dir", help="path to directory similarity matrices")
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)

    args = parser.parse_args()

    args = normalize_ws(args)
    dir_with_weights = os.path.join(args.similarity_dir, f"{args.w_title}title_{args.w_text}text_{args.w_index}index")

    meta_path_in = os.path.join(args.similarity_dir, 'meta.csv')
    meta_df = pd.read_csv(meta_path_in, index_col=False)

    print("\nCalculating weighted similarities with weights",
          args.w_title, args.w_text, args.w_index)
    print("percentile", args.percentile)
    weighted_similarities = get_weighted_similarities(args.similarity_dir, args.w_title,
                                                      args.w_text, args.w_index)

    print("\nBuilding graph")
    above_percentile_graph = build_graph(weighted_similarities, args.percentile, meta_df)

    print("\nRunning louvain algorithm for finding communities")
    communities = community_louvain.best_partition(above_percentile_graph)
    print("Communities done. dumping output")

    communities_lst = [communities[i] for i in range(len(meta_df))]
    meta_df["community"] = communities_lst
    meta_path_out = os.path.join(dir_with_weights, 'meta.csv')
    meta_df.to_csv(meta_path_out, index=False)

    comm_dict = Counter(communities_lst)
    analyze_communities(comm_dict)



