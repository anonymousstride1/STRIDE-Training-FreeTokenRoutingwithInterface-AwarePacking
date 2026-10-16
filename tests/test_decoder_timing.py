import pytest
import torch

from stride.models.timing import DecoderTimer


def test_decoder_timer_separates_first_forward_and_removes_hooks_on_failure():
    model = torch.nn.Identity()
    syncs = []
    timer = DecoderTimer(model, lambda: syncs.append(True))
    with pytest.raises(RuntimeError):
        with timer:
            model(torch.ones(1, 10, 4))
            model(torch.ones(1, 1, 4))
            raise RuntimeError('generation failed')
    assert len(timer.durations) == 2
    assert timer.prefill == timer.durations[0]
    assert timer.decode == timer.durations[1]
    assert len(syncs) == 4
    assert not model._forward_hooks and not model._forward_pre_hooks
