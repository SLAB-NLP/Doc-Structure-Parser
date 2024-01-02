import os

from argparse import ArgumentParser
import numpy as np
import pandas as pd


def divide_to_tens(df_for_single_cluster, full_df):
    shuffled_df = df_for_single_cluster.sample(frac=1)
    cluster_id = df_for_single_cluster["community"].iloc[0]
    cluster_rep = df_for_single_cluster["representative"].iloc[0]
    df_rows = []
    for i in range(0, len(df_for_single_cluster), 9):
        rows_for_single_annotation = shuffled_df.iloc[i:i+9]
        if len(rows_for_single_annotation) < 9 and i > 0:
            continue
        experiment_dict = {"cluster_id": cluster_id, "cluster_rep": cluster_rep}
        intruder_appear_index = np.random.randint(0, len(rows_for_single_annotation) + 1)
        intruder_row = sample_intruder(experiment_dict, full_df, i, intruder_appear_index)
        for j in range(len(rows_for_single_annotation) + 1):
            if j == intruder_appear_index:
                experiment_dict[f"index_{j}"] = experiment_dict[f"index_intruder_in_df"]
                experiment_dict[f"sentence_{j}_text"] = intruder_row["title_text"]
            else:
                index_in_df = j if j < intruder_appear_index else j - 1
                row = rows_for_single_annotation.iloc[index_in_df]
                experiment_dict[f"index_{j}"] = row.name
                experiment_dict[f"sentence_{j}_text"] = row["title_text"]
        df_rows.append(experiment_dict)
    return df_rows


def sample_intruder(experiment_dict, full_df, i, intruder_appear_index):
    random_intruder = full_df[full_df["rank"] != i].sample(n=1)
    intruder_row = random_intruder.iloc[0]
    experiment_dict[f"index_intruder_in_df"] = intruder_row.name
    experiment_dict[f"title_intruder"] = intruder_row["title_text"]
    experiment_dict["community_intruder"] = intruder_row["community"]
    experiment_dict["intruder_rank"] = intruder_row["rank"]
    experiment_dict["in_test_intruder_location"] = intruder_appear_index
    return intruder_row


def generate_test(full_df_path, override, out_path):
    df = pd.read_csv(full_df_path, index_col=False)
    # get clusters by rank from 1 to max ignore nan
    communities = df["community"].unique()
    full_df_rows = []
    for comm in communities:
        if comm == -1:
            continue
        df_i = df[df["community"] == comm]
        df_rows_for_single_cluster = divide_to_tens(df_i, df)
        full_df_rows.extend(df_rows_for_single_cluster)

    # sample 20 rows for qualification
    qual_len = 20
    df_experiments = pd.DataFrame(full_df_rows)
    indices_out = np.random.choice(len(full_df_rows), size=qual_len, replace=False)
    qualification_df = df_experiments.iloc[indices_out]
    full_experiment_df = df_experiments.iloc[df_experiments.index.difference(indices_out)]
    full_experiment_df = full_experiment_df.sample(frac=1)

    if not override:
        assert not os.path.exists(out_path.replace(".csv", "_experiment.csv")), f"file {out_path} already exists"
    full_experiment_df.to_csv(out_path.replace(".csv", "_experiment.csv"), index=False)
    divide_experiments_to_100s(full_experiment_df, out_path)
    path_to_qualification = out_path.replace(".csv", "_qualification.csv")
    qualification_df.to_csv(path_to_qualification, index=False)


def divide_experiments_to_100s(full_experiment_df, out_path):
    for i in range(len(full_experiment_df) // 100 + 1):
        truncated_df = full_experiment_df.iloc[i * 100:(i + 1) * 100]
        print(out_path.replace(".csv", f"_experiment_{i}.csv"))
        truncated_df.to_csv(out_path.replace(".csv", f"_experiment_{i}.csv"), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="path to csv file with all communities information (meta_filtered.csv)")
    parser.add_argument("--override", action="store_true", default=False,
                        help="override existing experiment files")
    parser.add_argument("--out", required=True,
                        help="path to output directory where all experiment files will be saved")
    args = parser.parse_args()
    generate_test(args.path, args.override, args.out)
