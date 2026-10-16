import pytest

from stride import RouterConfig


def test_unknown_config_rejected():
    with pytest.raises(ValueError, match='Unknown router keys'):
        RouterConfig.from_dict({'budjet': 32})


@pytest.mark.parametrize(
    'values',
    [
        {'budget': 0},
        {'visionzip_context_tokens': 0},
        {'vispruner_important_ratio': 1.1},
        {'stride_concept_contrast_min': -1.0},
        {'otprune_gamma': 0.0},
        {'stride_semantic_anchor_fraction': -0.1},
        {'stride_semantic_gain_min': -0.1},
        {'stride_residual_ridge': 0.0},
        {'method': 'cove'},
    ],
)
def test_invalid_values_rejected(values):
    with pytest.raises(ValueError):
        RouterConfig(**values)


@pytest.mark.parametrize(
    'values',
    [
        {'stride_semantic_space': 'unknown'},
        {'stride_output_order': 'score'},
        {'stride_position_mode': 'absolute-ish'},
    ],
)
def test_invalid_interface_diagnostic_modes_are_rejected(values):
    with pytest.raises(ValueError):
        RouterConfig(**values)
