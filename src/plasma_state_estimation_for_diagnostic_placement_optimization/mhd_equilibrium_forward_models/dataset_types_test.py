"""
Simple tests to make sure that in the logic handling the dataset we don't introduce
data leakage which would contaminate the results.
"""

import numpy as np
import pydantic
import pytest
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_types,
    testing_utils,
)


def test_a_partial_or_overlapping_split_is_rejected() -> None:
    """Two partitions sharing an id would leak the test set into training."""
    basis = testing_utils.tiny_basis()
    fields = {
        "name": "tiny",
        "generated": "2026-08-24",
        "plasma_configuration_id": "DRnvsp6pzgmPZWwXdF6GDCG",
        "recommended_max_fnorm_vol": 0.01,
        "n_samples": 3,
        "basis": basis,
    }

    with pytest.raises(pydantic.ValidationError, match="together or not at all"):
        dataset_types.DatasetMetadata(**fields, train_ids=["a"])
    with pytest.raises(pydantic.ValidationError, match="partitions overlap"):
        dataset_types.DatasetMetadata(
            **fields, train_ids=["a", "b"], validation_ids=["b"], test_ids=["c"]
        )


def test_a_split_sharing_a_row_index_is_rejected() -> None:
    """A row may belong to at most one partition."""
    with pytest.raises(pydantic.ValidationError, match="share at least one row index"):
        dataset_types.DataSplit(
            train=np.array([0, 1, 2]), validation=np.array([2]), test=np.array([3])
        )
