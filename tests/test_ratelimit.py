"""
Tests for the async rate limiter.
Run with: pytest tests/test_ratelimit.py -v
"""
import time

from diffprompt.ratelimit import AsyncRateLimiter


async def test_disabled_is_noop():
    """rate <= 0 disables limiting; many acquires should be instant."""
    rl = AsyncRateLimiter(rate=0)
    start = time.monotonic()
    for _ in range(200):
        await rl.acquire()
    assert time.monotonic() - start < 0.1


async def test_initial_burst_is_immediate():
    """The first `rate` acquires come from the full bucket with no wait."""
    rl = AsyncRateLimiter(rate=5, per=10.0)
    start = time.monotonic()
    for _ in range(5):
        await rl.acquire()
    assert time.monotonic() - start < 0.2


async def test_paces_after_bucket_drains():
    """
    After draining the bucket, further acquires are spaced by per/rate.
    5 per 1s -> the 6th and 7th cost ~0.2s each. Lower bound kept generous
    to avoid timing flakiness.
    """
    rl = AsyncRateLimiter(rate=5, per=1.0)
    start = time.monotonic()
    for _ in range(7):
        await rl.acquire()
    assert time.monotonic() - start >= 0.3
