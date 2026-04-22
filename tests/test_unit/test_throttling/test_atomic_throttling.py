import asyncio
import json
import threading
import time
from http import HTTPStatus
from typing import Final
from unittest.mock import MagicMock

import pytest
from django.http import HttpResponse
from freezegun.api import FrozenDateTimeFactory
from inline_snapshot import snapshot
from typing_extensions import override

from dmr import Controller, ResponseSpec, validate
from dmr.plugins.pydantic import PydanticFastSerializer
from dmr.test import DMRAsyncRequestFactory, DMRRequestFactory
from dmr.throttling import AsyncThrottle, Rate, SyncThrottle, ThrottlingReport
from dmr.throttling.algorithms import LeakyBucket
from dmr.throttling.backends import DjangoCache
from dmr.throttling.headers import XRateLimit

_xrate = XRateLimit()

_LIMIT: Final = 3


def test_incr_with_expiry_starts_at_one(freezer: FrozenDateTimeFactory) -> None:
    """First call creates a new window and returns count=1."""
    backend = DjangoCache()
    now = int(time.time())

    count, window_expiry = backend.incr_with_expiry(MagicMock(), MagicMock(), 'k', 60)

    assert count == 1
    assert window_expiry == now + 60


def test_incr_with_expiry_increments_within_window(freezer: FrozenDateTimeFactory) -> None:
    """Successive calls within the same window produce 1, 2, 3, …"""
    backend = DjangoCache()

    counts = [
        backend.incr_with_expiry(MagicMock(), MagicMock(), 'k', 60)[0]
        for _ in range(4)
    ]

    assert counts == [1, 2, 3, 4]


def test_incr_with_expiry_window_expiry_is_stable(freezer: FrozenDateTimeFactory) -> None:
    """All calls within the same window return the same expiry timestamp."""
    backend = DjangoCache()
    now = int(time.time())

    exps = [
        backend.incr_with_expiry(MagicMock(), MagicMock(), 'k', 30)[1]
        for _ in range(3)
    ]

    assert exps == [now + 30, now + 30, now + 30]


def test_incr_with_expiry_resets_after_ttl(freezer: FrozenDateTimeFactory) -> None:
    """After the TTL elapses, the next call starts a fresh window at count=1."""
    backend = DjangoCache()

    for _ in range(5):
        backend.incr_with_expiry(MagicMock(), MagicMock(), 'k', 60)

    freezer.tick(delta=61)

    count, _ = backend.incr_with_expiry(MagicMock(), MagicMock(), 'k', 60)
    assert count == 1


class _SyncController(Controller[PydanticFastSerializer]):
    throttling = [SyncThrottle(_LIMIT, Rate.second)]

    def get(self) -> str:
        return 'ok'


class _AsyncController(Controller[PydanticFastSerializer]):
    throttling = [AsyncThrottle(_LIMIT, Rate.second)]

    async def get(self) -> str:
        return 'ok'


def test_allows_exactly_max_requests_sync(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Exactly max_requests are served; the very next one is rate-limited."""
    for n in range(_LIMIT):
        assert (
            _SyncController.as_view()(dmr_rf.get('/')).status_code == HTTPStatus.OK
        ), f'request {n + 1} of {_LIMIT} was unexpectedly blocked'

    response = _SyncController.as_view()(dmr_rf.get('/'))
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    assert response.headers['X-RateLimit-Limit'] == str(_LIMIT)
    assert response.headers['X-RateLimit-Remaining'] == '0'
    assert json.loads(response.content) == snapshot({
        'detail': [{'msg': 'Too many requests', 'type': 'ratelimit'}],
    })


@pytest.mark.asyncio
async def test_allows_exactly_max_requests_async(
    dmr_async_rf: DMRAsyncRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Exactly max_requests are served; the very next one is rate-limited (async)."""
    for n in range(_LIMIT):
        response = await dmr_async_rf.wrap(
            _AsyncController.as_view()(dmr_async_rf.get('/')),
        )
        assert response.status_code == HTTPStatus.OK, (
            f'request {n + 1} of {_LIMIT} was unexpectedly blocked'
        )

    response = await dmr_async_rf.wrap(
        _AsyncController.as_view()(dmr_async_rf.get('/')),
    )
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    assert response.headers['X-RateLimit-Limit'] == str(_LIMIT)
    assert response.headers['X-RateLimit-Remaining'] == '0'
    assert json.loads(response.content) == snapshot({
        'detail': [{'msg': 'Too many requests', 'type': 'ratelimit'}],
    })


def test_window_resets_after_ttl_sync(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """After the rate window expires, the counter resets and requests are allowed again."""
    for _ in range(_LIMIT):
        assert _SyncController.as_view()(dmr_rf.get('/')).status_code == HTTPStatus.OK

    assert _SyncController.as_view()(dmr_rf.get('/')).status_code == HTTPStatus.TOO_MANY_REQUESTS

    freezer.tick(delta=int(Rate.second))

    assert _SyncController.as_view()(dmr_rf.get('/')).status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_window_resets_after_ttl_async(
    dmr_async_rf: DMRAsyncRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """After the rate window expires, the counter resets and requests are allowed again (async)."""
    for _ in range(_LIMIT):
        response = await dmr_async_rf.wrap(
            _AsyncController.as_view()(dmr_async_rf.get('/')),
        )
        assert response.status_code == HTTPStatus.OK

    response = await dmr_async_rf.wrap(
        _AsyncController.as_view()(dmr_async_rf.get('/')),
    )
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    freezer.tick(delta=int(Rate.second))

    response = await dmr_async_rf.wrap(
        _AsyncController.as_view()(dmr_async_rf.get('/')),
    )
    assert response.status_code == HTTPStatus.OK


class _NonAtomicBackend(DjangoCache):
    """DjangoCache variant that opts out of atomic increment, forcing the non-atomic path."""

    @override
    def incr_with_expiry(self, endpoint, controller, cache_key, ttl_seconds):  # type: ignore[override]
        return None

    @override
    async def aincr_with_expiry(self, endpoint, controller, cache_key, ttl_seconds):  # type: ignore[override]
        return None


class _NonAtomicSyncController(Controller[PydanticFastSerializer]):
    throttling = [SyncThrottle(_LIMIT, Rate.second, backend=_NonAtomicBackend())]

    def get(self) -> str:
        return 'ok'


class _NonAtomicAsyncController(Controller[PydanticFastSerializer]):
    throttling = [AsyncThrottle(_LIMIT, Rate.second, backend=_NonAtomicBackend())]

    async def get(self) -> str:
        return 'ok'


def test_non_atomic_fallback_enforces_limit_sync(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Rate limiting is correct when the backend opts out of atomic increment (sync)."""
    for n in range(_LIMIT):
        assert (
            _NonAtomicSyncController.as_view()(dmr_rf.get('/')).status_code == HTTPStatus.OK
        ), f'request {n + 1} was unexpectedly blocked'

    assert (
        _NonAtomicSyncController.as_view()(dmr_rf.get('/')).status_code
        == HTTPStatus.TOO_MANY_REQUESTS
    )

    freezer.tick(delta=int(Rate.second))

    assert _NonAtomicSyncController.as_view()(dmr_rf.get('/')).status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_non_atomic_fallback_enforces_limit_async(
    dmr_async_rf: DMRAsyncRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Rate limiting is correct when the backend opts out of atomic increment (async)."""
    for n in range(_LIMIT):
        response = await dmr_async_rf.wrap(
            _NonAtomicAsyncController.as_view()(dmr_async_rf.get('/')),
        )
        assert response.status_code == HTTPStatus.OK, f'request {n + 1} was unexpectedly blocked'

    response = await dmr_async_rf.wrap(
        _NonAtomicAsyncController.as_view()(dmr_async_rf.get('/')),
    )
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    freezer.tick(delta=int(Rate.second))

    response = await dmr_async_rf.wrap(
        _NonAtomicAsyncController.as_view()(dmr_async_rf.get('/')),
    )
    assert response.status_code == HTTPStatus.OK


class _SyncReportController(Controller[PydanticFastSerializer]):
    @validate(
        ResponseSpec(str, status_code=HTTPStatus.OK, headers=_xrate.provide_headers_specs()),
        throttling=[SyncThrottle(_LIMIT, Rate.second, response_headers=[_xrate])],
    )
    def get(self) -> HttpResponse:
        return self.to_response('ok', headers=ThrottlingReport(self).report())


def test_throttling_report_decrements_remaining_sync(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """ThrottlingReport reflects decreasing remaining count as atomic increments accumulate."""
    for expected_remaining in range(_LIMIT - 1, -1, -1):
        response = _SyncReportController.as_view()(dmr_rf.get('/'))
        assert response.status_code == HTTPStatus.OK
        assert response.headers.get('X-RateLimit-Remaining') == str(expected_remaining)
        assert response.headers.get('X-RateLimit-Limit') == str(_LIMIT)


class _AsyncReportController(Controller[PydanticFastSerializer]):
    @validate(
        ResponseSpec(str, status_code=HTTPStatus.OK, headers=_xrate.provide_headers_specs()),
        throttling=[AsyncThrottle(_LIMIT, Rate.second, response_headers=[_xrate])],
    )
    async def get(self) -> HttpResponse:
        return self.to_response('ok', headers=await ThrottlingReport(self).areport())


@pytest.mark.asyncio
async def test_throttling_report_decrements_remaining_async(
    dmr_async_rf: DMRAsyncRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """ThrottlingReport reflects decreasing remaining count as async atomic increments accumulate."""
    for expected_remaining in range(_LIMIT - 1, -1, -1):
        response = await dmr_async_rf.wrap(
            _AsyncReportController.as_view()(dmr_async_rf.get('/')),
        )
        assert response.status_code == HTTPStatus.OK
        assert response.headers.get('X-RateLimit-Remaining') == str(expected_remaining)
        assert response.headers.get('X-RateLimit-Limit') == str(_LIMIT)


class _AtomicGuardBackend(DjangoCache):
    """Backend that raises if incr_with_expiry is called, confirming LeakyBucket never calls it."""

    @override
    def incr_with_expiry(self, endpoint, controller, cache_key, ttl_seconds):  # type: ignore[override]
        raise AssertionError('LeakyBucket must not call incr_with_expiry')

    @override
    async def aincr_with_expiry(self, endpoint, controller, cache_key, ttl_seconds):  # type: ignore[override]
        raise AssertionError('LeakyBucket must not call incr_with_expiry')


class _LeakyBucketController(Controller[PydanticFastSerializer]):
    throttling = [
        SyncThrottle(
            _LIMIT,
            Rate.second,
            algorithm=LeakyBucket(),
            backend=_AtomicGuardBackend(),
        ),
    ]

    def get(self) -> str:
        return 'ok'


def test_leaky_bucket_does_not_use_atomic_path(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
) -> None:
    """LeakyBucket uses the non-atomic get/set path and never calls incr_with_expiry."""
    for _ in range(_LIMIT):
        assert _LeakyBucketController.as_view()(dmr_rf.get('/')).status_code == HTTPStatus.OK

    assert (
        _LeakyBucketController.as_view()(dmr_rf.get('/')).status_code
        == HTTPStatus.TOO_MANY_REQUESTS
    )

    freezer.tick(delta=int(Rate.second))

    assert _LeakyBucketController.as_view()(dmr_rf.get('/')).status_code == HTTPStatus.OK


_N_CONCURRENT: Final = 20


def test_incr_with_expiry_is_safe_under_concurrent_threads() -> None:
    """Concurrent threads each receive a unique, gapless count.

    LocMemCache is single-process only, but this test pins the intra-process
    thread-safety guarantee of the add+incr pattern.  In a real multi-worker
    deployment Redis / Memcached provide the same guarantee cross-process via
    server-side atomic operations.
    """
    backend = DjangoCache()
    results: list[int] = []
    lock = threading.Lock()

    def _worker() -> None:
        count, _ = backend.incr_with_expiry(
            MagicMock(), MagicMock(), 'thread_concurrent::c', 60,
        )
        with lock:
            results.append(count)

    threads = [threading.Thread(target=_worker) for _ in range(_N_CONCURRENT)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == list(range(1, _N_CONCURRENT + 1))


@pytest.mark.asyncio
async def test_incr_with_expiry_is_safe_under_concurrent_coroutines() -> None:
    """Concurrent coroutines each receive a unique, gapless count.

    asyncio.gather schedules coroutines on a single thread, so there is no
    true interleaving while aincr_with_expiry delegates to the synchronous
    incr_with_expiry.  This test documents the expected outcome and will
    remain valid if aincr_with_expiry is later made truly async.
    """
    backend = DjangoCache()

    async def _worker() -> int:
        count, _ = await backend.aincr_with_expiry(
            MagicMock(), MagicMock(), 'async_concurrent::c', 60,
        )
        return count

    results = list(await asyncio.gather(*[_worker() for _ in range(_N_CONCURRENT)]))
    assert sorted(results) == list(range(1, _N_CONCURRENT + 1))
