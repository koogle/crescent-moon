import time

from crescent_moon.deadline import cancel_at_deadline


def test_cancels_outstanding_calls_at_the_deadline():
    cancelled = []
    result = cancel_at_deadline(
        time.time() + 0.05, lambda: ["fc-a", "fc-b"], poll_seconds=0.01, cancel=cancelled.append
    )
    assert (result, cancelled) == (["fc-a", "fc-b"], ["fc-a", "fc-b"])


def test_finished_work_ends_the_wait_without_cancelling():
    cancelled = []
    result = cancel_at_deadline(
        time.time() + 60,
        lambda: ["fc-a"],
        done=lambda: True,
        poll_seconds=0.01,
        cancel=cancelled.append,
    )
    assert (result, cancelled) == (None, [])
