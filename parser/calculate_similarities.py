
from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import shutil
import faiss


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_language_model(pre_trained_model_name):
    device = get_device()
    print("Available device:", device)
    model = SentenceTransformer(model_name_or_path=pre_trained_model_name, device=device)
    return model


def calculate_cosine_similarity(vectors):
    row_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / row_norms

    # Create an index for cosine similarity search
    index = faiss.IndexFlatIP(normalized_vectors.shape[1])  # IndexFlatIP for cosine similarity

    if torch.cuda.is_available():
        print("GPU available")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Add vectors to the index
    index.add(normalized_vectors)

    # Perform a cosine similarity search on all vectors
    similarities, indexes = index.search(normalized_vectors, k=2048)

    reordered_array = np.zeros((normalized_vectors.shape[0], normalized_vectors.shape[0]))
    for i in range(reordered_array.shape[0]):
        for index, j in enumerate(indexes[i]):
            reordered_array[i, j] = similarities[i, index]
    return reordered_array


def calculate_index_similarity(index_array):
    """
    similarity between (x,y) (fractions) is 1 - abs(x-y)
    :param index_array:
    :return:
    """
    # Calculate pairwise differences
    differences = np.abs(np.subtract.outer(index_array, index_array))

    # Calculate fraction similarities
    similarity_matrix = 1 - differences

    return similarity_matrix


def generate_similarities_matrix_embeddings(df, model, output_dir, batch_size):
    for category in ["title", "section"]:
        print("", f"{category.upper()}", "Embedding content", sep='\n')
        text = df[f"{category}_text"].values
        df.loc[df[pd.isna(df[f"{category}_text"])].index, "section_text"] = ""
        embedding = model.encode(text, show_progress_bar=True, batch_size=batch_size)
        print("Embedding done, now calculating similarities")
        sim_matrix = calculate_cosine_similarity(embedding)
        out_path = os.path.join(output_dir, f"{category}_sim.npy")
        np.save(out_path, sim_matrix)


def generate_index_similarities(df, output_dir):
    print("\nGenerating index similarity matrix")
    sim_matrix = calculate_index_similarity(df["normalized_index"].values)
    out_path = os.path.join(output_dir, f"index_sim.npy")
    np.save(out_path, sim_matrix)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", help="a pretrained model to load with SBert")
    parser.add_argument("--out_path",
                        help="path of output dir where meta df already exists")
    parser.add_argument("--batch_size", default=100, type=int,
                        help="batch size for encoding")

    args = parser.parse_args()
    df_path = os.path.join(args.out_path, 'meta.csv')
    print(f"\nLoading df from {df_path}")
    input_df = pd.read_csv(df_path, index_col=False)
    print("\nLoading language model", args.model)
    language_model = load_language_model(args.model)

    out_dir = os.path.join(args.out_path, args.model)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nData will be saved in {out_dir}")

    generate_index_similarities(input_df, out_dir)
    generate_similarities_matrix_embeddings(input_df, language_model, out_dir, args.batch_size)
    shutil.copy2(src=df_path, dst=os.path.join(out_dir, "meta.csv"))
