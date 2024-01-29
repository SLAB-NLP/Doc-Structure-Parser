import json
import os.path
from argparse import ArgumentParser
import pandas as pd
import numpy as np

pd.set_option("display.precision", 3)

COLUMNS_PER_CLASS_RESULTS = ["gold title", "class size in segments", "exact segment match accuracy",
                             "class size in sections",
                             "exact section match precision", "exact section match recall", "exact section match f1",
                             "partial section match precision", "partial section match recall", "partial section match f1"]


def eval_df(predictions, gold_labels, title_mapping, out, quiet=False):

    predictions.loc[predictions["community"] == -1, "representative"] = "NA"
    predictions = predictions.merge(gold_labels, on=["filename", "title_text", "section_text", "title_index", "normalized_index", "original_title_line"], how="left")
    predictions = predictions.dropna(subset=["gold_cluster"])

    predictions["predicted_mapping"] = predictions.apply(lambda x: title_mapping.get(x["representative"], "NA"), axis=1)
    predictions["prediction_correct"] = predictions.apply(lambda x: x["gold_title"] == x["predicted_mapping"], axis=1)
    missing_gold_class = set(predictions["gold_title"].unique()).difference(set(title_mapping.values()))

    df_per_gold_title = pd.DataFrame(columns=COLUMNS_PER_CLASS_RESULTS)
    predictions_no_missing_class = predictions[~predictions['gold_title'].isin(missing_gold_class)]
    predictions_no_missing_pred = predictions[predictions["predicted_mapping"] != "NA"]

    accuracy_per_gold_title = predictions_no_missing_class.groupby("gold_title")["prediction_correct"].mean()

    intersection_recall_df = predictions_no_missing_class.groupby(["gold_title", "filename"])["prediction_correct"].sum() > 0
    intersection_recall_df = intersection_recall_df.groupby("gold_title").mean()
    exact_section_match_recall = predictions_no_missing_class.groupby(["gold_title", "filename"])["prediction_correct"].sum() >= predictions_no_missing_class.groupby(["gold_title", "filename"])["prediction_correct"].size()
    exact_section_match_recall = exact_section_match_recall.groupby("gold_title").mean()

    intersection_precision_df = predictions_no_missing_pred.groupby(["predicted_mapping", "filename"])["prediction_correct"].sum() > 0
    intersection_precision_df = intersection_precision_df.groupby("predicted_mapping").mean()
    exact_section_match_precision = predictions_no_missing_pred.groupby(["predicted_mapping", "filename"])["prediction_correct"].sum() >= predictions_no_missing_pred.groupby(["predicted_mapping", "filename"])["prediction_correct"].size()
    exact_section_match_precision = exact_section_match_precision.groupby("predicted_mapping").mean()
    if len(intersection_precision_df) == 1:
        exact_section_match_precision, intersection_precision_df = handle_only_single_class_predicted(
            exact_section_match_precision, intersection_precision_df, exact_section_match_recall.index)

    f1_score_exact = 2 * (exact_section_match_precision * exact_section_match_recall) / (exact_section_match_precision + exact_section_match_recall)
    f1_score_partial = 2 * (intersection_precision_df * intersection_recall_df) / (intersection_precision_df + intersection_recall_df)

    # set results in table
    df_per_gold_title["partial section match recall"] = intersection_recall_df[accuracy_per_gold_title.index].values
    df_per_gold_title["exact section match recall"] = exact_section_match_recall[accuracy_per_gold_title.index].values
    df_per_gold_title["partial section match precision"] = intersection_precision_df[accuracy_per_gold_title.index].values
    df_per_gold_title["partial section match f1"] = f1_score_partial[accuracy_per_gold_title.index].values
    df_per_gold_title["exact section match f1"] = f1_score_exact[accuracy_per_gold_title.index].values
    df_per_gold_title["exact section match precision"] = exact_section_match_precision[accuracy_per_gold_title.index].values
    df_per_gold_title["gold title"] = accuracy_per_gold_title.index
    df_per_gold_title["exact segment match accuracy"] = accuracy_per_gold_title.values
    class_size = predictions_no_missing_class.groupby("gold_title").size()
    df_per_gold_title["class size in segments"] = class_size[accuracy_per_gold_title.index].values
    df_per_gold_title["class size in sections"] = predictions_no_missing_class.groupby("gold_title")["filename"].nunique()[accuracy_per_gold_title.index].values

    weighted_mean_exact = (f1_score_exact * class_size).sum() / class_size.sum()
    weighted_mean_partial = (f1_score_partial * class_size).sum() / class_size.sum()
    df_per_gold_title.to_csv(out, index=False, float_format='%.3f')

    columns_accumulated = ["category", "f1 macro (equal)", "f1 micro (weighted)"]
    df_accumulated = pd.DataFrame(columns=columns_accumulated)
    df_accumulated["category"] = ["exact match", "partial match"]
    df_accumulated["f1 macro (equal)"] = [f1_score_exact.mean(), f1_score_partial.mean()]
    df_accumulated["f1 micro (weighted)"] = [weighted_mean_exact, weighted_mean_partial]

    if not quiet:
        print()
        print(df_accumulated)
    df_accumulated.to_markdown(out.replace('.csv', '_accumulated.md'), index=False)

    return df_accumulated


def handle_only_single_class_predicted(exact_section_match_precision, intersection_precision_df, indices):
    intersection_precision = []
    exact_precision = []
    for gold in indices:
        if gold in intersection_precision_df:
            intersection_precision.append(intersection_precision_df[gold])
            exact_precision.append(exact_section_match_precision[gold])
        else:
            intersection_precision.append(1)
            exact_precision.append(1)
    intersection_precision_df = pd.Series(intersection_precision, index=indices)
    exact_section_match_precision = pd.Series(exact_precision, index=indices)
    return exact_section_match_precision, intersection_precision_df


def get_mapping(gold_df, path_to_mapping=""):
    if path_to_mapping != "":
        with open(path_to_mapping, "r") as f:
            toc_mapping = json.load(f)
    else:
        toc_mapping = {cls: cls for cls in gold_df["gold_title"].unique()}
    return toc_mapping


def main(pred_df, gold_df, toc_mapping, out_dir):

    # evaluate predictions against gold
    out_path = os.path.join(out_dir, "predictions_eval.csv")
    eval_df(pred_df, gold_df, toc_mapping, out_path)

    # most frequent class baseline
    reverse_mapping = {v: k for k, v in toc_mapping.items()}
    count_per_class = gold_df["gold_title"].value_counts()
    most_frequent_class = count_per_class.index[0]
    num_most_frequent_class = count_per_class[most_frequent_class]
    percentage_most_frequent_class = num_most_frequent_class / count_per_class.sum() * 100
    matching_representative_key = reverse_mapping[most_frequent_class]
    predict_most_frequent_class = pred_df.copy()
    predict_most_frequent_class["representative"] = matching_representative_key
    print("\n\nPREDICT MOST FREQUENT CLASS")
    print(f"that is, \"{most_frequent_class}\", with {num_most_frequent_class} instances ({percentage_most_frequent_class:.3f})")
    out_path = os.path.join(out_dir, "most_frequent_class_baseline.csv")
    eval_df(predict_most_frequent_class, gold_df, toc_mapping, out_path)

    # random baseline
    print("\n\nRANDOM PREDICTIONS")
    all_random_runs = []
    for i in range(100):
        random_predictions = pred_df.copy()
        predicting_labels = pd.Series(
            np.random.choice(gold_df["gold_title"].dropna().unique(), len(random_predictions)))
        prediction_reps = predicting_labels.apply(lambda x: reverse_mapping.get(x, "NA"))
        random_predictions["representative"] = prediction_reps.values
        out_dir_random = os.path.join(out_dir, "random_baseline")
        os.makedirs(out_dir_random, exist_ok=True)
        out_path = os.path.join(out_dir_random, f"eval_{i}.csv")
        df_accum = eval_df(random_predictions, gold_df, toc_mapping, out_path, quiet=True)
        all_random_runs.append(df_accum)
    # get the mean of all random runs
    all_random_runs = pd.concat(all_random_runs)
    all_random_runs = all_random_runs.groupby("category").mean()
    print("\n\nRANDOM PREDICTIONS MEAN")
    print(all_random_runs)
    all_random_runs.to_markdown(os.path.join(out_dir, "random_baseline_accum_mean.md"), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True,
                        help="path to predictions csv (meta_filtered.csv)")
    parser.add_argument("--gold", type=str, required=True,
                        help="path to gold csv. "
                             "expected to match the predictions csv format, with additional column of 'gold_title'")
    parser.add_argument("--toc_mapping", type=str, default="",
                        help="path to json file with mapping from representatives to gold titles. "
                             "If not provided, will assume that the gold titles match the representatives")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="path to output directory where all evaluation files will be saved")

    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions, index_col=False)
    gold = pd.read_csv(args.gold, index_col=False)
    mapping_toc = get_mapping(gold, path_to_mapping=args.toc_mapping)
    os.makedirs(args.out_dir, exist_ok=True)
    main(predictions, gold, mapping_toc, args.out_dir)
