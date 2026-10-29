from continual.drift import AccuracyWindowDetector, DistributionShiftDetector


def test_accuracy_window_fires_on_sustained_drop():
    det = AccuracyWindowDetector(window=10, reference_accuracy=0.9, min_drop=0.1, min_samples=10)
    assert not det.update([True] * 10).fired
    signal = det.update([False] * 10)
    assert signal.fired
    assert signal.detail["window_accuracy"] == 0.0


def test_accuracy_window_quiet_before_min_samples():
    det = AccuracyWindowDetector(window=100, min_samples=50)
    assert not det.update([False] * 5).fired


def test_accuracy_window_reset_rebaselines():
    det = AccuracyWindowDetector(window=10, reference_accuracy=0.9, min_samples=10)
    det.update([False] * 10)
    det.reset_reference(0.5)
    assert det.reference_accuracy == 0.5
    assert det.window_accuracy == 0.0


def test_distribution_screen_flags_shifted_means():
    det = DistributionShiftDetector(z_threshold=1.0)
    ref = [[0.0, 0.0]] * 50
    det.fit(ref)
    signal = det.screen([[5.0, 5.0]] * 20)
    assert signal.fired
    assert signal.detail["max_z"] > 1.0


def test_distribution_screen_quiet_on_same_distribution():
    det = DistributionShiftDetector(z_threshold=1.0)
    ref = [[float(i % 3), 0.0] for i in range(60)]
    det.fit(ref)
    signal = det.screen([[float(i % 3), 0.0] for i in range(20)])
    assert not signal.fired
