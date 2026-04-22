import abc
import dataclasses
import time
from typing import TYPE_CHECKING

from django.core.cache import DEFAULT_CACHE_ALIAS, BaseCache, caches
from typing_extensions import TypedDict, override

from dmr.settings import default_parser, default_renderer

if TYPE_CHECKING:
    from dmr.controller import Controller
    from dmr.endpoint import Endpoint
    from dmr.serializer import BaseSerializer


class CachedRateLimit(TypedDict):
    """Representation of a cached object's metadata."""

    # We usually store `int(time.time())` result here:
    time: int
    # We overly complicate the storage a bit, because this design
    # allows future potential algorithms to store requests as lists,
    # if it is needed.
    history: list[int]


class BaseThrottleBackend:
    """
    Base class for all throttling backends.

    It must provide sync and async API for sync and async throttling classes.
    """

    __slots__ = ()

    @abc.abstractmethod
    def get(
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
    ) -> CachedRateLimit | None:
        """Sync get the cached rate limit state."""
        raise NotImplementedError

    @abc.abstractmethod
    async def aget(
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
    ) -> CachedRateLimit | None:
        """Async get the cached rate limit state."""
        raise NotImplementedError

    @abc.abstractmethod
    def set(
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
        cache_object: CachedRateLimit,
        *,
        ttl_seconds: int,
    ) -> None:
        """Sync set the cached rate limit state."""
        raise NotImplementedError

    @abc.abstractmethod
    async def aset(
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
        cache_object: CachedRateLimit,
        *,
        ttl_seconds: int,
    ) -> None:
        """Async set the cached rate limit state."""
        raise NotImplementedError

    def incr_with_expiry(  # noqa: WPS324
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
        ttl_seconds: int,
    ) -> tuple[int, int] | None:
        """
        Atomically increment the request counter for *cache_key*.

        Creates the counter (and its window expiry) with *ttl_seconds* TTL
        if it does not exist yet.

        Returns ``(new_count, window_expiry_unix_timestamp)`` on success,
        or ``None`` when this backend does not support atomic increment -
        in that case the caller falls back to the non-atomic
        read-modify-write path.

        .. warning::

            Backends that return ``None`` are only safe within a single
            worker process.  For multi-worker deployments (Gunicorn,
            uvicorn with ``--workers N``) use a backend whose
            ``incr_with_expiry`` returns a real value, such as the
            default :class:`DjangoCache` backed by Redis or Memcached.
        """
        return None  # noqa: WPS324

    async def aincr_with_expiry(  # noqa: WPS324
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
        ttl_seconds: int,
    ) -> tuple[int, int] | None:
        """Async version of :meth:`incr_with_expiry`."""
        return None  # noqa: WPS324


@dataclasses.dataclass(slots=True, frozen=True)
class DjangoCache(BaseThrottleBackend):  # noqa: WPS214
    """
    Uses Django cache framework for storing the rate limiting state.

    .. seealso::

        https://docs.djangoproject.com/en/stable/topics/cache/

    """

    cache_name: str = DEFAULT_CACHE_ALIAS
    _cache: BaseCache = dataclasses.field(init=False)

    def __post_init__(
        self,
    ) -> None:
        """Initialize the cache backend."""
        object.__setattr__(self, '_cache', caches[self.cache_name])

    @override
    def get(
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
    ) -> CachedRateLimit | None:
        """Sync get the cached rate limit state.

        Two storage formats co-exist in the same cache namespace:

        * **Atomic path** (written by :meth:`incr_with_expiry`):
          ``{cache_key}::c`` (``int`` counter) +
          ``{cache_key}::t`` (window expiry).
        * **Non-atomic / legacy path** (written by :meth:`set`):
          ``{cache_key}`` (serialised :class:`CachedRateLimit`).

        The atomic format is detected by the presence of an integer at
        ``{cache_key}::c``.  A non-integer value (including ``None``) falls
        through to the legacy deserialisation path.  In the returned
        :class:`CachedRateLimit`, ``history[0]`` holds the raw counter value;
        :class:`~dmr.throttling.algorithms.SimpleRate` reads ``history[0]``
        as the request count in both cases.
        """
        count = self._cache.get(f'{cache_key}::c')
        if isinstance(count, int):
            window_expiry = self._cache.get(f'{cache_key}::t') or 0
            return CachedRateLimit(history=[count], time=window_expiry)
        stored_cache = self._cache.get(cache_key)
        return self._load_cache(controller, stored_cache)

    @override
    async def aget(
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
    ) -> CachedRateLimit | None:
        """Async get the cached rate limit state.

        See :meth:`get` for the dual-format detection logic.
        """
        count = await self._cache.aget(f'{cache_key}::c')
        if isinstance(count, int):
            window_expiry = await self._cache.aget(f'{cache_key}::t') or 0
            return CachedRateLimit(history=[count], time=window_expiry)
        stored_cache = await self._cache.aget(cache_key)
        return self._load_cache(controller, stored_cache)

    @override
    def set(
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
        cache_object: CachedRateLimit,
        *,
        ttl_seconds: int,
    ) -> None:
        """Sync set the cached rate limit state."""
        self._cache.set(
            cache_key,
            self._dump_cache(controller, cache_object),
            timeout=ttl_seconds,
        )

    @override
    async def aset(
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
        cache_object: CachedRateLimit,
        *,
        ttl_seconds: int,
    ) -> None:
        """Async set the cached rate limit state."""
        await self._cache.aset(
            cache_key,
            self._dump_cache(controller, cache_object),
            timeout=ttl_seconds,
        )

    @override
    def incr_with_expiry(
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
        ttl_seconds: int,
    ) -> tuple[int, int]:
        """
        Atomically increment using Django's ``cache.add`` + ``cache.incr``.

        For Redis and Memcached backends each operation is atomic at the cache
        server level, making the counter safe across multiple worker processes.
        For ``LocMemCache`` (typically used in tests) the guarantee only holds
        within a single process.

        **Race condition between** ``add`` **and** ``incr``

        ``cache.add`` and ``cache.incr`` are separate round-trips.  Between the
        two calls another worker may execute its own ``incr`` - this is safe and
        intentional: each caller receives a distinct, monotonically increasing
        value.

        The window-expiry key (``{cache_key}::t``) is set in a separate
        ``cache.add`` call, so it may not yet exist when a concurrent ``incr``
        returns.  The ``or window_expiry`` fallback at the end of this method
        handles that by computing ``now + ttl_seconds`` locally.  Different
        workers may derive values up to 1 second apart for the same window;
        this is acceptable because the expiry timestamp is used only in
        response headers, never for enforcement.
        """
        now = int(time.time())
        window_expiry = now + ttl_seconds
        count_key = f'{cache_key}::c'
        time_key = f'{cache_key}::t'
        try:
            count = self._cache.incr(count_key)
        except ValueError:
            if self._cache.add(count_key, 0, ttl_seconds):
                self._cache.add(time_key, window_expiry, ttl_seconds)
            try:  # noqa: WPS505
                count = self._cache.incr(count_key)
            except ValueError:
                self._cache.set(count_key, 1, ttl_seconds)
                self._cache.set(time_key, window_expiry, ttl_seconds)
                return 1, window_expiry
        return count, self._cache.get(time_key) or window_expiry

    @override
    async def aincr_with_expiry(
        self,
        endpoint: 'Endpoint',
        controller: 'Controller[BaseSerializer]',
        cache_key: str,
        ttl_seconds: int,
    ) -> tuple[int, int]:
        """Async version of :meth:`incr_with_expiry`.

        Delegates to the synchronous :meth:`incr_with_expiry` so that
        ``time.time()`` is evaluated in the caller's context - important for
        correct behaviour under ``freezegun`` in tests.

        .. warning::

            This method calls the synchronous :meth:`incr_with_expiry`
            directly, which performs blocking cache I/O (``cache.add`` +
            ``cache.incr`` round-trips).  Under ASGI this blocks the event
            loop for the duration of those round-trips.  Django's cache
            framework does not currently expose native async variants of
            ``add`` / ``incr``; a future revision should delegate to async
            operations once that support lands (maybe).
        """
        return self.incr_with_expiry(
            endpoint,
            controller,
            cache_key,
            ttl_seconds,
        )

    def _load_cache(
        self,
        controller: 'Controller[BaseSerializer]',
        stored_cache: bytes | None,
    ) -> CachedRateLimit | None:
        if stored_cache is None:
            return None

        return controller.serializer.deserialize(  # type: ignore[no-any-return]
            stored_cache,
            parser=default_parser,
            request=controller.request,
            model=CachedRateLimit,
        )

    def _dump_cache(
        self,
        controller: 'Controller[BaseSerializer]',
        cache_object: CachedRateLimit,
    ) -> bytes:
        return controller.serializer.serialize(
            cache_object,
            renderer=default_renderer,
        )
