import os
import pandas as pd


def load_cv_corpus(data_dir, metadata_file="clips_durations.tsv"):
    """
    Load and display metadata for the Common Voice dataset.

    Parameters:
        data_dir (str): Path to the dataset directory.
        metadata_file (str): Name of the metadata file (default is "validated.tsv").

    Returns:
        pd.DataFrame: Loaded metadata as a DataFrame.
    """
    # Path to the metadata file
    metadata_path = os.path.join(data_dir, metadata_file)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file '{metadata_file}' not found in directory '{data_dir}'.")

    # Load metadata file into a DataFrame
    df = pd.read_csv(metadata_path, sep="\t")

    # Display the first few rows of the DataFrame
    print("Loaded Metadata:")
    print(df.head())

    return df


if __name__ == "__main__":
    # Replace this with the actual path to your dataset
    print(ls scratch-shared/partial-asr/datasets/)
    dataset_directory = "scratch-shared/partial-asr/datasets/cv-corpus-19.0-2024-09-13"

    try:
        # Load the dataset metadata
        metadata = load_cv_corpus(dataset_directory)

        # Show additional details about the dataset
        print("\nDataset Information:")
        print(metadata.info())
    except Exception as e:
        print(f"Error: {e}")
