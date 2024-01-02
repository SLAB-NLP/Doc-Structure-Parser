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
./run_all.sh
```

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

`<model_name>` is the name of the model used for encoding and `<w>title_<w>text_<w>index` are the different weights set when building the graph for louvain algorithm.