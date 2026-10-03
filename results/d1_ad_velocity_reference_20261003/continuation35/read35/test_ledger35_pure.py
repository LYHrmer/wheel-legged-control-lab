"""Zero-authority model calls remain forbidden even in a failed episode."""
from ledger_read35 import model_count_valid35


def test_failed_episode_cannot_hide_attempted_training_call():
    for key in ('backward', 'learn', 'train', 'save', 'forward', 'evaluate_actions', 'predict_values'):
        assert model_count_valid35({}, key, 0, False)
        assert not model_count_valid35({key: {'attempted': 1, 'returned': 0}}, key, 0, False)
        assert not model_count_valid35({key: {'attempted': 1, 'returned': 1}}, key, 0, False)


def test_authorized_prediction_may_have_one_failed_pending_attempt():
    counts = {'predict': {'attempted': 202, 'returned': 201}}
    assert model_count_valid35(counts, 'predict', 201, False)
    assert not model_count_valid35(counts, 'predict', 201, True)
    counts['predict']['attempted'] = 203
    assert not model_count_valid35(counts, 'predict', 201, False)
