#!/bin/bash


# Function to display script usage
function usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --model_name MODEL_NAME       Specify the model name to load for sentence_transformers"
    echo "  -o, --out_path OUT_PATH           Specify the output path"
    echo "  -i, --in_path IN_PATH             Specify the input path of the data collection text files"
    echo "  --ds_name DS_NAME                 Specify the name of the dataset you parse"
    echo "  -h, --help                        Display this help message"
}


# Default parameter values
w_title="0.0"
w_text="0.0"
w_index="0.0"

# Parse command-line parameters
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--model_name)
            model_name="$2"
            shift 2
            ;;
        -o|--out_path)
            out_path="$2"
            shift 2
            ;;
        -i|--in_path)
            original_in_path="$2"
            shift 2
            ;;
        --w_title)
            w_title="$2"
            shift 2
            ;;
        --w_index)
            w_index="$2"
            shift 2
            ;;
        --w_text)
            w_text="$2"
            shift 2
            ;;
        --ds_name)
            ds_name="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --percentile)
            percentile="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$out_path" ] || [ -z "$original_in_path" ] || [ -z "$ds_name" ] || [ -z "$model_name" ]; then
    echo "Error: Output path, input path, model name and dataset name are required."
    usage
    exit 1
fi

similarity_dir="${out_path}${model_name}/"
out_path_with_w="${similarity_dir}${w_title}title_${w_text}text_${w_index}index/"
csv_communities="${out_path_with_w}meta.csv"
filtered_df="${similarity_dir}meta_filtered.csv"
logdir="${similarity_dir}logs/"

mkdir "${out_path}"
mkdir "${similarity_dir}"
mkdir "${out_path_with_w}"
mkdir "${logdir}"

export PYTHONPATH=./

# check if there exists a file in out_path called "meta.csv"
if [ -f "${out_path}meta.csv" ]; then
    echo "File ${out_path}meta.csv already exists. Skipping nodes info generation."
else
  cmd="python generate_nodes_info.py --text_dir_path ${original_in_path} --out_path ${out_path} --dataset_name ${ds_name} > ${logdir}01_nodes_info.log"
  echo "${cmd}"
  eval "${cmd}"
fi


# check if similarity metrics already calculated
if [ -f "${similarity_dir}title_sim.npy" ] && [ -f "${similarity_dir}section_sim.npy" ] && [ -f "${similarity_dir}index_sim.npy" ]; then
    echo "Similarity files already exist in ${similarity_dir}. Skipping calculate similarities."
else
  cmd="python calculate_similarities.py --model ${model_name} --out_path ${out_path} > ${logdir}02_similarities.log"
  # if --batch_size is specified, add it to the command
  if [ ! -z "$batch_size" ]; then
      cmd="${cmd} --batch_size ${batch_size}"
  fi
  echo "${cmd}"
  eval "${cmd}"
fi


cmd="python run_louvain_algorithm.py --similarity_dir ${similarity_dir} --w_index ${w_index} --w_title ${w_title} --w_text ${w_text} > ${logdir}03_louvain.log"
# if --percentile is specified, add it to the command
if [ ! -z "$percentile" ]; then
    cmd="${cmd} --percentile ${percentile}"
fi
echo "${cmd}"
eval "${cmd}"

cmd="python get_representatives.py --csv_communities ${csv_communities} --similarity_dir ${similarity_dir} --w_index ${w_index} --w_title ${w_title} --w_text ${w_text} > ${logdir}04_representative.log"
echo "${cmd}"
eval "${cmd}"

cmd="python filter_communities.py --df_path ${csv_communities} > ${logdir}05_comm_duplicates.log"
echo "${cmd}"
eval "${cmd}"
