"""The launch deadline's independent supervisor: cancel calls still outstanding at it.

`modal_app.py` fixes one deadline per launch, bounds the GPU call's timeout by it and
spawns `cancel_at_deadline` on a separate CPU container, so shutdown does not depend on
the launching machine staying connected.
"""

import time


def calls_finished(call_ids):
    """True once every call has returned or failed (none still queued or running)."""
    import modal

    for call_id in call_ids:
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return False
        except Exception:  # noqa: BLE001, S110 - a failed or cancelled call has finished
            pass
    return True


def cancel_at_deadline(deadline_unix, calls, done=lambda: False, poll_seconds=300, cancel=None):
    """Wait for the deadline, then cancel outstanding calls.

    `calls()` returns the call IDs to cancel; `done()` ends the wait early once the work
    has finished. Returns the cancelled IDs, or None when the work finished first.
    """
    if cancel is None:
        import modal

        def cancel(call_id):
            modal.FunctionCall.from_id(call_id).cancel()

    while (remaining := deadline_unix - time.time()) > 0:
        time.sleep(min(remaining, poll_seconds))
        if done():
            return None
    cancelled = list(calls())
    for call_id in cancelled:
        cancel(call_id)
    return cancelled
