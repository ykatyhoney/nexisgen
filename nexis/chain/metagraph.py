"""Metagraph helpers for validator miner discovery."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import inspect
from typing import Any, AsyncIterator
import bittensor as bt


async def _resolve_maybe_awaitable(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _build_subtensor(network: str) -> Any:

    factory = getattr(bt, "AsyncSubtensor", None)
    if factory is None:
        raise RuntimeError(
            "bittensor.AsyncSubtensor is unavailable; upgrade bittensor to a version that supports async subtensor."
        )
    try:
        return factory(network=network)
    except TypeError:
        return factory(network)


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    close_coro = getattr(coro, "close", None)
    if callable(close_coro):
        close_coro()
    raise RuntimeError("Synchronous chain helpers cannot run inside an active event loop")


async def _close_subtensor(subtensor: Any) -> None:
    if subtensor is None:
        return
    close = getattr(subtensor, "close", None)
    if callable(close):
        await _resolve_maybe_awaitable(close())
        return
    substrate = getattr(subtensor, "substrate", None)
    close = getattr(substrate, "close", None)
    if callable(close):
        await _resolve_maybe_awaitable(close())


@asynccontextmanager
async def _open_subtensor(network: str) -> AsyncIterator[Any]:
    subtensor = _build_subtensor(network)
    enter = getattr(subtensor, "__aenter__", None)
    if callable(enter):
        async with subtensor:
            yield subtensor
        return
    try:
        yield subtensor
    finally:
        await _close_subtensor(subtensor)


async def fetch_hotkeys_from_metagraph_async(
    *,
    netuid: int,
    network: str,
    subtensor: Any | None = None,
) -> list[str]:
    """Load subnet metagraph and return discovered hotkeys."""
    if subtensor is not None:
        metagraph = await _resolve_maybe_awaitable(subtensor.metagraph(netuid))
        hotkeys = [hk for hk in list(metagraph.hotkeys) if isinstance(hk, str) and hk]
        return hotkeys
    async with _open_subtensor(network) as owned_subtensor:
        metagraph = await _resolve_maybe_awaitable(owned_subtensor.metagraph(netuid))
        hotkeys = [hk for hk in list(metagraph.hotkeys) if isinstance(hk, str) and hk]
        return hotkeys


def fetch_hotkeys_from_metagraph(
    *,
    netuid: int,
    network: str,
) -> list[str]:
    return _run_async(
        fetch_hotkeys_from_metagraph_async(
            netuid=netuid,
            network=network,
        )
    )


async def fetch_current_block_async(*, network: str, subtensor: Any | None = None) -> int:
    """Fetch current chain block from subtensor."""
    if subtensor is not None:
        return await fetch_current_block_from_subtensor(subtensor=subtensor)
    async with _open_subtensor(network) as subtensor:
        return await fetch_current_block_from_subtensor(subtensor=subtensor)


async def fetch_current_block_from_subtensor(*, subtensor: Any) -> int:
    """Fetch current chain block from an existing subtensor connection."""
    get_current_block = getattr(subtensor, "get_current_block", None)
    if callable(get_current_block):
        block = await _resolve_maybe_awaitable(get_current_block())
    else:
        block = await _resolve_maybe_awaitable(getattr(subtensor, "block"))
    return int(block)


def fetch_current_block(*, network: str) -> int:
    return _run_async(fetch_current_block_async(network=network))

