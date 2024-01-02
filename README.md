# Doc-Structure-Parser

This is an unsupervised method to extract the conceptual table of contents of a data collection.

## Quickstart

```
git clone <github_url>
cd <NAME>
python3 -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
cd parser
./run_all.sh -m MODEL --ds_name DATASET_NAME \
    -i INPUT_DIR -o OUTPUT_DIR  \
    --w_title W_TITLE --w_text W_TEXT --w_index W_INDEX
```

Parameters:
- `MODEL`: name of the model used for encoding, loaded via sentence_transformers package. 
List of available models can be found [here](https://www.sbert.net/docs/pretrained_models.html).
- `DATASET_NAME`: name of the dataset to parse, used for headers detection in [`generate_nodes_info.py`](parser/generate_nodes_info.py). 
See **[#Apply New Dataset](#apply-new-dataset)** for further explanation on how to run this code on your own dataset.
- `INPUT_DIR`: path to the directory containing the dataset to parse. 
This directory is expected to contain plain text files.
- `OUTPUT_DIR`: path to the directory where the output will be saved. 
See **[#Output Files](#output-files)** for further explanation on the directory format and the files it contains.
this `OUTPUT_DIR` parameter must end with a seperator (i.e., '/').
- `W_TITLE`, `W_TEXT`, `W_INDEX`: weights used for the graph building. 
The edges in the graph are weighted sum of the three similarities measures, weighted according these given ws, based on prior information regarding the specific dataset. 
For example, if you think that in your dataset the titles are very similar but the order is not strict, set high W_TITLE and low W_INDEX. 

## Apply New Dataset

In order to run this code on your own data, the only thing you to provide is a method that detects header candidates.

This method accepts as an input a string which is a line from the text and a tokenizer, and it is expected to return a boolean value indicating whether this line is a header candidate or not.

This method should be implemented in the file [`generate_nodes_info.py`](parser/generate_nodes_info.py). It should be called from the function `is_title_candidate`. The following pair of lines should be added, similarly to the existing implementation of 10k and CUAD:
```
elif ds_name == 'DATASET_NAME':
    return is_title_candidate_DATASET_NAME(line, tokenizer)
```

Headers can be detected according to heuristics like number of tokens in the sentence, percentage of capitalized words, and any prior knowledge you may have regarding the dataset.

## Output Files

The output directory will look as follows:
```
├── meta.csv
├── <model_name>
│   ├── meta.csv
│   ├── title_sim.npy
│   ├── section_sim.npy
│   ├── index_sim.npy
│   ├── logs
│   │   ├── 01_nodes_info.log
│   │   ├── 02_similarities.log
│   │   ├── 03_louvain.log
│   │   ├── 04_representative.log
│   │   ├── 05_comm_duplicates.log
│   ├── <w>title_<w>text_<w>index
│   │   ├── meta.csv
│   │   ├── meta_filtered.csv
```

The final output is the file `meta_filtered.csv` in the `<model_name>/<w>title_<w>text_<w>index` directory.
Each line in this csv represents a section in a document from the dataset, along with information regarding its mapping to the Conceptual Table of Contents of the data collection.

`<model_name>` is the name of the model used for encoding and `<w>title_<w>text_<w>index` are the different weights set when building the graph for louvain algorithm.

## Visualize Results

## Generate Data for Evaluation