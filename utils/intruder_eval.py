import os

import pandas as pd
import numpy as np
from argparse import ArgumentParser

UPPER_CONF = 0.75
LOWER_CONF = 0.25
LOWER_ACCURACY = 0.4
NUM_OPTIONS = 10


def main(path):
    paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".csv")]
    score_by_id = {}
    for path_csv in paths:
        df = pd.read_csv(path_csv)
        for _, row in df.iterrows():
            update_scores(row, score_by_id)

    approved_workers = qualify_workers(score_by_id)

    calc_results_for_approved_workers(approved_workers, score_by_id)


def update_scores(row, score_by_id):
    intruder_index = int(row["Input.in_test_intruder_location"])
    correct = row[f"Answer.answers.{intruder_index}"]
    num_marked = sum(row[f"Answer.answers.{i}"] for i in range(NUM_OPTIONS))
    if row["WorkerId"] not in score_by_id:
        score_by_id[row["WorkerId"]] = []
    conf = 1 - ((num_marked - 1) / NUM_OPTIONS)
    score_by_id[row["WorkerId"]].append((conf, correct))


def calc_results_for_approved_workers(approved_workers, score_by_id):
    all_tuples = []
    for worker in approved_workers:
        all_tuples.extend(score_by_id[worker])

    total = len(all_tuples)
    correct_total = sum(conf_correct[1] for conf_correct in all_tuples)
    all_conf = [conf_correct[0] for conf_correct in all_tuples]

    results_df = pd.DataFrame(columns=["num_samples", "accuracy", "confidence_avg", "confidence_std"])
    results_df.loc["original"] = [total, round(correct_total / total, 3), round(np.mean(all_conf), 3), round(np.std(all_conf), 3)]

    # perform IQR outlier removal
    q1 = np.quantile(all_conf, 0.25)
    q3 = np.quantile(all_conf, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # filter out tuples with conf out of range
    filtered_tuples = [conf_correct for conf_correct in all_tuples if lower_bound <= conf_correct[0] <= upper_bound]
    total_filtered = len(filtered_tuples)
    correct_total_filtered = sum(conf_correct[1] for conf_correct in filtered_tuples)
    all_conf_filtered = [conf_correct[0] for conf_correct in filtered_tuples]
    results_df.loc["IQR"] = [total_filtered, round(correct_total_filtered / total_filtered, 3), round(np.mean(all_conf_filtered), 3), round(np.std(all_conf_filtered), 3)]

    print(results_df.to_markdown())
    print("Performing IQR outlier removal")
    print(f"q1: {q1}, q3:{q3}, IQR=({lower_bound},{upper_bound})")


def qualify_workers(score_by_id):
    approved_workers = set()
    for worker in score_by_id:
        total = len(score_by_id[worker])
        correct = sum(conf_correct[1] for conf_correct in score_by_id[worker])
        accuracy = correct / total
        avg_conf = np.mean([conf_correct[0] for conf_correct in score_by_id[worker]])
        print(f"{worker}: accuracy = {accuracy}, avg confidence = {avg_conf}, total = {len(score_by_id[worker])}")

        if (avg_conf >= 0.75 and accuracy <= 0.4) or (avg_conf <= 0.25):
            print("worker rejected", accuracy, avg_conf, len(score_by_id[worker]))
        else:
            approved_workers.add(worker)
    return approved_workers


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path",
                        help="path to the results directory. Expecting the files to follow mechanical turk format.")
    args = parser.parse_args()
    main(args.path)
