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
        score_by_id[row["WorkerId"]] = {"conf": [], "correct": 0, "total": 0}
    if correct:
        score_by_id[row["WorkerId"]]["correct"] += 1
    score_by_id[row["WorkerId"]]["conf"].append(1 - ((num_marked - 1) / NUM_OPTIONS))
    score_by_id[row["WorkerId"]]["total"] += 1


def calc_results_for_approved_workers(approved_workers, score_by_id):
    total = 0
    correct_total = 0
    all_conf = []
    for worker in approved_workers:
        total += score_by_id[worker]['total']
        correct_total += score_by_id[worker]['correct']
        all_conf.extend(score_by_id[worker]['conf'])
    print(f"approved {len(approved_workers)} workers out of {len(score_by_id)}")
    print("num samples:", total)
    print("accumulated accuracy:", round(correct_total / total, 3))
    print("accumulated avg conf:", round(np.mean(all_conf), 3))


def qualify_workers(score_by_id):
    approved_workers = set()
    for worker in score_by_id:
        accuracy = score_by_id[worker]['correct'] / score_by_id[worker]['total']
        avg_conf = np.mean(score_by_id[worker]['conf'])
        print(f"{worker}: accuracy = {accuracy}, avg confidence = {avg_conf}, total = {score_by_id[worker]['total']}")

        if (avg_conf >= UPPER_CONF and accuracy <= LOWER_ACCURACY) or (avg_conf <= LOWER_CONF):
            print("worker rejected", accuracy, avg_conf, score_by_id[worker]['total'])
        else:
            approved_workers.add(worker)
    return approved_workers


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path",
                        help="path to the results directory. Expecting the files to follow mechanical turk format.")
    args = parser.parse_args()
    main(args.path)
