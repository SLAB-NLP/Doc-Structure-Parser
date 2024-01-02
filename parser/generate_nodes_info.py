
from argparse import ArgumentParser
import pandas as pd
import re
import os
from tqdm import tqdm
import spacy


ONLY_NUMBERS_REGEX = '^(((\d{1,2} *)\.* *)){1,4}?'
NUM_TOKENS_FOR_TITLE = 10
DF_COLUMNS = ["filename", "title_index", "normalized_index", "original_title_line",
              "title_text", "section_text"]
ROMAN_NUMERALS_REGEX_START = r'^[IXVixv]{1,4}'
ROMAN_NUMERALS_REGEX_END = r'[IXVixv]{1,4}$'


def unite_numbering_lines(lines):
    only_numbering_lines = []
    results = []
    for i, line in enumerate(lines):
        if is_numbering_line(line):
            only_numbering_lines.append(i)
            continue

    if len(only_numbering_lines) == 0:
        return lines

    if only_numbering_lines[-1] == len(lines)-1:  # last line is a digit line
        only_numbering_lines = only_numbering_lines[:-1]  # remove it

    j = 0
    while j < len(lines):
        to_add = [lines[j]]
        while j in only_numbering_lines:
            j += 1
            to_add.append(lines[j])
        results.append(" ".join(to_add))
        j += 1
    lines = results
    return lines


def is_numbering_line(line):
    line_no_not_w = re.sub(r'\W', '', line)
    line_no_numbers = re.sub(r'\d', '', line_no_not_w)
    line_no_romans = re.sub(ROMAN_NUMERALS_REGEX_START, '', line_no_numbers)
    line_no_romans = re.sub(ROMAN_NUMERALS_REGEX_END, '', line_no_romans)
    clear_line = line_no_romans
    return len(clear_line) <= 2


def load_sent_tokenizer():
    """ loads the sentence tokenization model """
    nlp = spacy.load("en_core_web_lg")
    nlp.disable_pipes(["tagger", "ner", "lemmatizer", "attribute_ruler"])
    return nlp


def is_title_candidate(line, ds_name, sentence_tokenizer):
    """ to add new pre-processing function, change `DATASET` to your dataset name, and
    implement a boolean function similarly to CUAD and 10k """
    if ds_name == 'CUAD':
        return is_title_candidate_cuad(line, sentence_tokenizer)
    elif ds_name == '10k':
        return is_title_candidate_10k(line, sentence_tokenizer)
    # elif DATASET == <name>:  # template for a new dataset
    #     return <user implementation>(line)
    else:
        raise NotImplementedError


def is_title_candidate_10k(line, sentence_tokenizer):
    if line.lower().startswith("item ") or line.lower().startswith("part "):
        return True
    if len(sentence_tokenizer(line)) > NUM_TOKENS_FOR_TITLE:
        return False
    if re.fullmatch(r'none ?\.?\W*', line.lower()) or re.fullmatch(r'omitted ?\.?\W*', line.lower()):
        return False
    words = re.findall(r'\b\w*[A-Z]\w*\b', line)
    total_words = len(re.findall(r'\b[a-zA-Z]+\b', line))
    if total_words == 0:
        return False
    percentage_of_capital = len(words) / total_words
    return percentage_of_capital > 0.6


def is_title_candidate_cuad(line, sentence_tokenizer):
    return len(sentence_tokenizer(line)) <= NUM_TOKENS_FOR_TITLE


def detect_titles(lines, ds_name, sentence_tokenizer):
    unite = []
    title_lines = []
    index = 0
    while index < len(lines):
        if not is_title_candidate(lines[index], ds_name, sentence_tokenizer):
            unite.append(lines[index])
            index += 1
            continue
        # current line is title
        current_line = lines[index]
        r = 1
        while index+r < len(lines):
            next_line = lines[index+r]
            if is_title_candidate(next_line, ds_name, sentence_tokenizer):
                # title followed by a title.
                current_line = f"{current_line}\n\n{next_line}"
                r += 1
            else:
                title_lines.append((len(unite), current_line))
                unite.append(current_line)
                break
        if index+r == len(lines):
            title_lines.append((len(unite), current_line))
            unite.append(current_line)
        index += r

    return unite, title_lines


def process_single_file(txt_path, sentence_tokenizer, filename, ds_name):
    file_df = pd.DataFrame(columns=DF_COLUMNS)
    with open(txt_path, 'rt', encoding='utf8') as f:
        data = f.read(tokenization_model.max_length)

    raw_lines = get_data_lines(data, sentence_tokenizer)
    lines, title_lines = detect_titles(raw_lines, ds_name, sentence_tokenizer)
    if len(title_lines) == 0:
        print("no titles detected", filename)
        return None
    if title_lines[-1][0] < len(lines)-1:  # last line is not a header
        title_lines.append((len(lines), "EOF"))
    num_titles = len(title_lines)-1  # added EOF as last title
    all_sections_texts = []
    for index, (j, title) in enumerate(title_lines[:-1]):
        first_text_line_index = j+1
        next_title_index_in_original = title_lines[index+1][0]
        current_title_text_under = lines[first_text_line_index:next_title_index_in_original]
        section_text = ' '.join(current_title_text_under)
        all_sections_texts.append(section_text)
        normalized_index = index / num_titles
        title_dict = {"filename": filename,
                      "title_index": index, "normalized_index": normalized_index,
                      "original_title_line": j,
                      "title_text": title, "section_text": section_text}
        file_df.loc[len(file_df)] = title_dict

    return file_df


def get_data_lines(data, sentence_tokenizer):
    docs = sentence_tokenizer.pipe([data])
    lines = [[line.text.strip() for line in doc.sents] for doc in docs]
    lines = lines[0]
    lines = [line for line in lines if line != '']
    lines = unite_numbering_lines(lines)
    lines = split_none(lines)
    return lines


def split_none(lines):
    n_lines = []
    for line in lines:
        none_full_match = re.fullmatch(r'\s*((None)|(Omitted)|(Not Applicable))\.?\s*', line, flags=re.IGNORECASE)
        if none_full_match is not None:
            n_lines.append(line)
            continue
        none_in_start = re.search('^((None)|(Omitted)|(Not Applicable))\.\s*', line, flags=re.IGNORECASE)
        if none_in_start is not None:
            n_lines.append(none_in_start.group())
            n_lines.append(line[none_in_start.span()[1]:])
            continue
        none_in_end = re.search('((None)|(Omitted)|(Not Applicable))\.?\s*$', line,  flags=re.IGNORECASE)
        if none_in_end is not None:
            n_lines.append(line[:none_in_end.span()[0]])
            n_lines.append(none_in_end.group())
            continue
        n_lines.append(line)
    return n_lines


def process_corpus(sentence_tokenizer, data_path, ds_name):
    full_df = pd.DataFrame(columns=DF_COLUMNS)
    full_df = full_df.astype({'filename': 'str', 'title_index': 'int',
                              'normalized_index': 'float', 'original_title_line': 'int',
                              'title_text': 'str', 'section_text': 'str'})
    files = os.listdir(data_path)
    for file in tqdm(files):
        file_txt_path = os.path.join(data_path, file)
        file_df = process_single_file(file_txt_path, sentence_tokenizer, file, ds_name)
        if file_df is not None:
            full_df = pd.concat([full_df, file_df], ignore_index=True)

    return full_df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--text_dir_path",
                        help="path to the directory with all text files to process")
    parser.add_argument("--out_path", help="path of output dir")
    parser.add_argument("--dataset_name", help="the name of the dataset, used as a flag "
                                               "for the title detection function")

    args = parser.parse_args()
    out_dir = args.out_path
    print(f"\ndata will be saved in {out_dir}\n")
    print(f"generating meta data csv")
    os.makedirs(out_dir, exist_ok=True)
    tokenization_model = load_sent_tokenizer()
    input_df = process_corpus(tokenization_model, args.text_dir_path, args.dataset_name)
    df_path = os.path.join(out_dir, "meta.csv")
    input_df.to_csv(df_path, encoding="utf8", index=False)
