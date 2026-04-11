import json

import pandas as pd

from conftest import load_module


make_circles = load_module(
    "tests._make_circles",
    "qbiocode/data_generation/make_circles.py",
)
make_class = load_module(
    "tests._make_class",
    "qbiocode/data_generation/make_class.py",
)
make_spheres = load_module(
    "tests._make_spheres",
    "qbiocode/data_generation/make_spheres.py",
)


def test_generate_circles_datasets_writes_expected_files(tmp_path):
    make_circles.generate_circles_datasets(
        n_samples=[12],
        noise=[0.15],
        save_path=str(tmp_path),
        random_state=7,
    )

    dataset_path = tmp_path / "circles_data-1.csv"
    config_path = tmp_path / "dataset_config.json"

    assert dataset_path.exists()
    assert config_path.exists()

    dataset = pd.read_csv(dataset_path)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    assert list(dataset.columns) == ["0", "1", "class"]
    assert len(dataset) == 12
    assert dataset["class"].isin([0, 1]).all()
    assert list(config.values()) == [{"n_samples": 12, "noise": 0.15}]


def test_generate_classification_datasets_only_writes_valid_configurations(tmp_path):
    make_class.generate_classification_datasets(
        n_samples=[10],
        n_features=[4, 3],
        n_informative=[2],
        n_redundant=[2],
        n_classes=[2],
        n_clusters_per_class=[1],
        weights=[[0.5, 0.5]],
        save_path=str(tmp_path),
        random_state=11,
    )

    csv_files = sorted(tmp_path.glob("class_data-*.csv"))
    config_path = tmp_path / "dataset_config.json"

    assert [path.name for path in csv_files] == ["class_data-1.csv"]

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    assert list(config.values()) == [
        {
            "n_samples": 10,
            "n_features": 4,
            "n_informative": 2,
            "n_redundant": 2,
            "n_classes": 2,
            "n_clusters_per_class": 1,
            "weights": [0.5, 0.5],
        }
    ]


def test_generate_points_in_nd_sphere_respects_radius_threshold():
    points = make_spheres.generate_points_in_nd_sphere(
        n_s=25,
        dim=4,
        radius=3,
        thresh=0.6,
    )

    norms = (points ** 2).sum(axis=1) ** 0.5

    assert points.shape == (25, 4)
    assert (norms <= 3).all()
    assert (norms >= 1.8).all()
