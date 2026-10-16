#!/usr/bin/env python
"""Structural STRIDE ablations for the exact-reference study."""

from __future__ import annotations


STRIDE_ABLATIONS: dict[str, dict[str, object]] = {
    'random_pruning': {'method': 'random'},
    'no_semantics': {
        'stride_use_semantics': False,
    },
    'no_intent_routing': {
        'stride_use_intent_routing': False,
    },
    'no_diversity_expert': {
        'stride_use_diversity_expert': False,
    },
    'no_residual_space': {
        'stride_use_residual_space': False,
    },
    'no_projected_geometry': {
        'stride_use_projected_geometry': False,
    },
    'no_modality_calibration': {
        'stride_use_modality_calibration': False,
    },
    # Effective-path ablations.  These target the two always-active halves of
    # the hybrid selector and the two ways routed patches are presented to the
    # language model.  Unlike the conditional semantic/intent switches above,
    # they exercise the main path for broad benchmark classes.
    'salience_only': {
        'vispruner_important_ratio': 1.0,
        'stride_use_diversity_expert': False,
    },
    'diversity_only': {
        'vispruner_important_ratio': 0.0,
        'stride_use_diversity_expert': True,
    },
    'selection_order': {
        'stride_output_order': 'selection',
    },
    'dense_positions': {
        'stride_position_mode': 'dense',
    },
}


if __name__ == '__main__':
    names = ', '.join(STRIDE_ABLATIONS)
    raise SystemExit(
        'Use scripts/run_full_ablations.py --config '
        'configs/experiments/ablations.yaml. Available variants: ' + names
    )
