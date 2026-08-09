import pandas as pd

from twinrag.detection import SensorModel, SensorSpec


def _dataset(values, parameter="pressure", asset="101"):
    return pd.DataFrame(
        {
            "timestamp_s": [hour * 3600 for hour in range(len(values))],
            "asset_id": asset,
            "asset_type": "node",
            "parameter": parameter,
            "value": values,
            "scenario": "test",
        }
    )


def test_noise_is_reproducible_for_a_seed():
    dataset = _dataset([50.0] * 10)

    first = SensorModel(seed=7).observe(dataset)
    second = SensorModel(seed=7).observe(dataset)

    pd.testing.assert_series_equal(first["value"], second["value"])


def test_different_seeds_give_different_readings():
    dataset = _dataset([50.0] * 10)

    first = SensorModel(seed=1).observe(dataset)
    second = SensorModel(seed=2).observe(dataset)

    assert not first["value"].equals(second["value"])


def test_row_order_does_not_change_a_reading():
    """
    Noise must attach to a reading's identity, not to its row position,
    or a reshuffled input would silently produce different data.
    """

    dataset = _dataset([50.0] * 10)

    straight = SensorModel(seed=3).observe(dataset)

    shuffled = SensorModel(seed=3).observe(
        dataset.sample(frac=1.0, random_state=0)
    )

    left = straight.set_index("timestamp_s")["value"].sort_index()
    right = shuffled.set_index("timestamp_s")["value"].sort_index()

    pd.testing.assert_series_equal(left, right)


def test_truth_is_preserved_alongside_the_noisy_reading():
    dataset = _dataset([50.0, 51.0, 52.0])

    observed = SensorModel(seed=0).observe(dataset)

    assert observed["truth"].tolist() == [50.0, 51.0, 52.0]
    assert observed["value"].tolist() != [50.0, 51.0, 52.0]


def test_disabling_noise_passes_truth_through_but_keeps_the_scale():
    """
    The noise-free ablation still needs a normaliser, otherwise its
    scores are not comparable with the noisy run's.
    """

    dataset = _dataset([50.0] * 5)

    observed = SensorModel(seed=0, noise_enabled=False).observe(dataset)

    assert observed["value"].tolist() == [50.0] * 5
    assert (observed["noise_std"] > 0).all()


def test_relative_noise_scales_with_magnitude():
    dataset = _dataset([1.0, 100.0], parameter="flowrate")

    model = SensorModel(
        specs={"flowrate": SensorSpec(absolute=0.0, relative=0.1)},
        seed=0,
    )

    observed = model.observe(dataset).sort_values("timestamp_s")

    assert observed["noise_std"].tolist() == [0.1, 10.0]


def test_sensor_coverage_drops_uninstrumented_assets():
    dataset = pd.concat(
        [_dataset([50.0] * 3, asset="101"), _dataset([40.0] * 3, asset="105")],
        ignore_index=True,
    )

    observed = SensorModel(sensor_assets=["101"], seed=0).observe(dataset)

    assert set(observed["asset_id"]) == {"101"}


def test_numeric_asset_ids_become_strings():
    """Net3 IDs are numeric text; read_csv turns them into ints."""

    dataset = _dataset([50.0] * 3)
    dataset["asset_id"] = 101

    observed = SensorModel(seed=0).observe(dataset)

    assert observed["asset_id"].tolist() == ["101"] * 3
