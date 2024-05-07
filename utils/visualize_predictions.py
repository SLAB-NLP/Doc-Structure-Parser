
import re
import streamlit as st
import pandas as pd
import random
import sys

# wide mode
st.set_page_config(layout="wide")

if "i" not in st.session_state:
    st.session_state.i = 0
    st.session_state.df = None


def generate_random_colors(length):
    colors = []
    for _ in range(length):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        colors.append(f"rgb({red}, {green}, {blue})")
    return colors


def generate_colors_map():
    df = st.session_state['df']
    representatives = []
    for rank in range(1, st.session_state['num_clusters'] + 1):
        rep = df[df["rank"] == rank]["representative"].values[0]
        representatives.append(rep)
    colors = generate_random_colors(len(representatives))
    color_map = {}
    for i, rep in enumerate(representatives):
        color_map[rep] = colors[i]
    st.session_state['color_map'] = color_map


def generate_sidebar_linking(color_map, line_numbers):
    for representative in color_map:
        id_rep = get_id_rep(representative)
        if representative in line_numbers:
            st.sidebar.markdown(
                f"<a style='border: 3px solid {color_map[representative]}; padding: 5px; font-size: 16px; color: black;' href='#{id_rep}'>{representative}</a>",
                unsafe_allow_html=True)
        else:
            st.sidebar.markdown(
                f"<a style='border: 3px solid {color_map[representative]}; padding: 5px; font-size: 16px; color: gray;'>{representative}</a>",
                unsafe_allow_html=True)


@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)


def load_example():
    csv_file = "appendix/10k_example_output.csv"
    df = load_csv(csv_file)
    st.session_state["df"] = df


def load_example_shoa_05_title():
    csv_file = "appendix/shoa_05title.csv"
    df = load_csv(csv_file)
    st.session_state["df"] = df


def load_example_shoa_08_title():
    csv_file = "appendix/shoa_08title.csv"
    df = load_csv(csv_file)
    st.session_state["df"] = df


def main():
    # Page title and description
    st.title("Conceptual ToC Viewer")


    # File selection
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        st.write("Select a CSV file generated in the clustering pipeline to see how the "
                 "conceptual ToC is applied in each doc.")
        csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
        st.write("or, press here to load an example file:")
        st.button("Load example (10k)", on_click=load_example)
        st.button("Load example (shoa 05-title)", on_click=load_example_shoa_05_title)
        st.button("Load example (shoa 08-title)", on_click=load_example_shoa_08_title)
    if csv_file is not None:
        df = load_csv(csv_file)
        st.session_state["df"] = df

    if st.session_state.df is not None:
        # Load CSV data
        df = st.session_state["df"]

        st.number_input("num clusters to display", min_value=1,
                        max_value=int(df["rank"].max()),
                        value=int(df["rank"].max()),
                        key="num_clusters",
                        on_change=generate_colors_map)

        # Set a variable once after a new CSV file is loaded
        if 'color_map' not in st.session_state:
            generate_colors_map()
        color_map = st.session_state['color_map']

        st.sidebar.markdown("<h3 style='font-size: 24px;'>ToC (color mapping)</h3>",
                            unsafe_allow_html=True)

        group_by_filename = df.groupby("filename").groups

        # Display file selection dropdown
        selected_file = st.selectbox("Select a file", group_by_filename.keys())

        # Filter dataframe based on selected file
        filtered_df = df.loc[group_by_filename[selected_file]]
        filtered_df = filtered_df.sort_values(by=['title_index']).reset_index()

        if not filtered_df.empty:
            display_single_file(color_map, filtered_df)
        else:
            st.write("Selected file not found in the CSV.")


def display_single_file(color_map, filtered_df):

    st.header("Text Content:")

    all_paragraphs, labels_start_end = get_paragraphs(filtered_df)

    line_numbers = {}
    is_open = (False, None)
    for i, paragraph in enumerate(all_paragraphs):
        prev_label, current_label = labels_start_end[i]
        if prev_label is not None:
            line_numbers[prev_label] = (line_numbers[prev_label], i)
            color = color_map[prev_label]
            st.markdown(f"""<hr style="height:10px;border:none;color:{color};background-color:{color};" /> """, unsafe_allow_html=True)
            is_open = (False, None)
        if current_label is not None:
            id_rep = get_id_rep(current_label)
            st.markdown(f"<h3 id='{id_rep}'>{current_label}</h3>", unsafe_allow_html=True)
            color = color_map[current_label]
            st.markdown(f"""<hr style="height:10px;border:none;color:{color};background-color:{color};" /> """,
                        unsafe_allow_html=True)
            is_open = (True, current_label)
            line_numbers[current_label] = i+1

        st.markdown(paragraph)

    if is_open[0]:
        label = is_open[1]
        color = color_map[label]
        st.markdown(
            f"""<hr style="height:10px;border:none;color:{color};background-color:{color};" /> """,
            unsafe_allow_html=True)
        line_numbers[label] = (line_numbers[label], len(all_paragraphs))

    generate_sidebar_linking(color_map, line_numbers)


def get_id_rep(representative):
    representative = re.sub(r' +', '-', representative)
    return representative.lower()


def get_paragraphs(filtered_df):
    all_paragraphs = []
    all_labels = []
    labels_start_end = []
    for i, row in filtered_df.iterrows():
        paragraph = f"{row['title_text']}\n\n{row['section_text']}\n\n"
        if i + 1 < len(filtered_df):
            if row['section_text'] == filtered_df.loc[i + 1]['title_text']:
                paragraph += f"{filtered_df.loc[i + 1]['section_text']}\n\n"
        label = row["representative"] if row["rank"] <= st.session_state['num_clusters'] else None
        prev = current = None
        if i == 0 and label is not None:
            current = label
        if i > 0 and all_labels[-1] != label:
            if i > 0 and all_labels[-1] is not None:
                prev = all_labels[-1]
            if label is not None:
                current = label
        labels_start_end.append((prev, current))
        all_paragraphs.append(paragraph)
        all_labels.append(label)

    return all_paragraphs, labels_start_end


if __name__ == '__main__':

    main()
