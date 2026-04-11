import pytest

from conftest import ensure_package, load_module


def load_generator_module():
    ensure_package("qbiocode", "qbiocode")
    ensure_package("qbiocode.data_generation", "qbiocode/data_generation")

    for module_name in [
        "make_circles",
        "make_moons",
        "make_class",
        "make_s_curve",
        "make_spheres",
        "make_spirals",
        "make_swiss_roll",
    ]:
        load_module(
            f"qbiocode.data_generation.{module_name}",
            f"qbiocode/data_generation/{module_name}.py",
        )

    return load_module(
        "qbiocode.data_generation.generator",
        "qbiocode/data_generation/generator.py",
    )


@pytest.mark.parametrize(
    ("dataset_type", "module_attr", "function_name", "expected_kwargs"),
    [
        (
            "circles",
            "circles",
            "generate_circles_datasets",
            {"n_samples": [9], "noise": [0.2], "save_path": "out", "random_state": 5},
        ),
        (
            "classes",
            "make_class",
            "generate_classification_datasets",
            {
                "n_samples": [9],
                "n_features": [6],
                "n_informative": [2],
                "n_redundant": [1],
                "n_classes": [2],
                "n_clusters_per_class": [1],
                "weights": [[0.5, 0.5]],
                "save_path": "out",
                "random_state": 5,
            },
        ),
        (
            "spheres",
            "spheres",
            "generate_spheres_datasets",
            {"n_s": [9], "dim": [6], "radius": [4], "save_path": "out", "random_state": 5},
        ),
        (
            "swiss_roll",
            "swiss_roll",
            "generate_swiss_roll_datasets",
            {
                "n_samples": [9],
                "noise": [0.2],
                "hole": [True],
                "save_path": "out",
                "random_state": 5,
            },
        ),
    ],
)
def test_generate_data_dispatches_to_expected_backend(
    monkeypatch,
    dataset_type,
    module_attr,
    function_name,
    expected_kwargs,
):
    generator = load_generator_module()
    captured = {}

    def fake_backend(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(getattr(generator, module_attr), function_name, fake_backend)

    generator.generate_data(
        type_of_data=dataset_type,
        save_path="out",
        n_samples=[9],
        noise=[0.2],
        hole=[True],
        n_classes=[2],
        dim=[6],
        rad=[4],
        n_features=[6],
        n_informative=[2],
        n_redundant=[1],
        n_clusters_per_class=[1],
        weights=[[0.5, 0.5]],
        random_state=5,
    )

    assert captured == expected_kwargs


def test_generate_data_rejects_unknown_dataset_type():
    generator = load_generator_module()

    with pytest.raises(ValueError, match="Invalid type_of_data"):
        generator.generate_data(type_of_data="unknown", save_path="out")
