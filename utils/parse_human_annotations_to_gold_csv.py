
from argparse import ArgumentParser
import pandas as pd
import numpy as np


def update_predictions_with_gold(gold_annotations, predictions_df):
    predictions_df["gold_title"] = np.nan
    for i, row in gold_annotations.iterrows():
        fname = row["filename"]
        fname_preds = predictions_df[predictions_df["filename"] == fname]
        for rep in gold_annotations.columns[3:]:
            if pd.isna(row[rep]):
                continue
            rep_range_tup = eval(row[rep])
            range_rep = np.arange(rep_range_tup[0], rep_range_tup[1] + 1) - 1
            indices = fname_preds[fname_preds["title_index"].isin(range_rep)].index
            predictions_df.loc[indices, "gold_title"] = rep
            predictions_df.loc[indices, "gold_cluster"] = np.max(predictions_df[predictions_df["representative"] == rep]["community"].unique())
        fname_preds = predictions_df[predictions_df["filename"] == fname]
        indices = fname_preds[pd.isna(fname_preds["gold_title"])].index
        predictions_df.loc[indices, "gold_title"] = "NA"
        predictions_df.loc[indices, "gold_cluster"] = -1
    predictions_df = predictions_df.dropna(subset=["gold_cluster"])
    predictions_df = predictions_df.drop(columns=["community", "original_repr", "representative", "similarity_to_repr", "rank"])
    return predictions_df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--annotations_from_streamlit", type=str, required=True)
    parser.add_argument("--gold_out_path", type=str, required=True)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions, index_col=False)
    gold = pd.read_csv(args.annotations_from_streamlit, index_col=False)
    predictions_with_gold = update_predictions_with_gold(gold, predictions)
    predictions_with_gold.to_csv(args.gold_out_path, index=False)
