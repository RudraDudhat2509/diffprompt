"""
Async token-bucket rate limiter.

Bounds the *rate* of operations (requests per period), which is what trips
free-tier API limits — distinct from a Semaphore, which only bounds how many
run concurrently. Used to pace Groq calls so we don't burst past its RPM cap.
"""
from __future__ import annotations
import asyncio
import time


class AsyncRateLimiter:
    """
    Allow at most `rate` acquisitions per `per` seconds, smoothed over time.

    Tokens refill continuously: after draining the bucket, each further
    acquire waits just long enough for one token to regenerate, so calls get
    spaced out rather than fired in a burst then stalled.

    A rate of 0 (or less) disables limiting entirely — acquire() is a no-op.
    """

    def __init__(self, rate: int, per: float = 60.0):
        self.rate = rate
        self.per = per
        self._tokens = float(rate)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self.rate <= 0:
            return  # limiting disabled

        while True:
            async with self._lock:
                now = time.monotonic()
                refill = (now - self._updated) * (self.rate / self.per)
                self._tokens = min(float(self.rate), self._tokens + refill)
                self._updated = now

                if self._tokens >= 1:
                    self._tokens -= 1
                    return

                # Not enough — compute the wait, then release the lock while
                # sleeping so other coroutines can recheck independently.
                wait = (1 - self._tokens) * (self.per / self.rate)

            await asyncio.sleep(wait)
