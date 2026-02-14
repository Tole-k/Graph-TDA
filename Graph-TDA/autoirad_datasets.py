import os
import pandas as pd
from tqdm import tqdm


def list_dataset_files(datasets_dir: str) -> list[str]:
    """Lists all datasets in directory.

    Args:
        datasets_dir (str): Directory with datasets.

    Returns:
        list[tuple[str, str]]: List of ..., ... pairs.
    """

    def is_dataset_file(path: str) -> bool:
        return path.endswith(".csv") or path.endswith(".dat")

    paths = []
    directory = os.path.join(datasets_dir)
    for root, _, files in os.walk(directory):
        for file in files:
            if not is_dataset_file(file):
                continue
            paths.append(os.path.join(root, file))

    paths.sort()
    return paths


def read_data_file(path: str) -> pd.DataFrame:
    """Reads given dataset.

    Args:
        path (str): Path to the file.

    Raises:
        ValueError: If file format is not supported

    Returns:
        pd.DataFrame: dataset
    """
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf8") as f:
            dataset = pd.read_csv(f)
    elif path.endswith(".dat"):
        with open(path, "r", encoding="utf8") as f:
            dataset = pd.read_csv(f, delimiter="\t").drop("Unnamed: 0", axis=1)
    else:
        raise ValueError(f"File format not supported for file: {path}")
    return dataset


def generate_training_parameters(
    datasets_dir: str = "AutoIRAD-datasets",
    scores_file: str = os.path.join("Graph-TDA", "family_rma.csv"),
):
    scores_df = pd.read_csv(scores_file)
    scores_df.set_index("dataset", inplace=True)
    datasets = []
    paths = list_dataset_files(datasets_dir)
    scores = []

    for path in tqdm(paths, total=len(paths)):
        dataset_name = os.path.splitext(os.path.basename(path))[0].replace("_R", "")
        dataset = read_data_file(path)
        X = dataset.drop(dataset.columns[-1], axis=1).values
        if dataset_name in scores_df.index:
            scores.append(scores_df.loc[dataset_name].values.tolist())
        else:
            print(f"Warning: No scores found for dataset {dataset_name}. Skipping.")
            continue
        datasets.append(X)
    print(f"Loaded {len(datasets)} datasets with scores.")
    return datasets, scores
