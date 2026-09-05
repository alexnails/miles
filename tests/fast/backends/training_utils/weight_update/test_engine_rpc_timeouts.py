import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from miles.backends.sglang_utils.sglang_api_client import (
    WEIGHT_SYNC_RPC_TIMEOUT_SECONDS,
    SGLangApiClient,
)
from miles.utils.distributed_utils import GLOO_GROUP_TIMEOUT

_MODULE = "miles.backends.sglang_utils.sglang_api_client"


def _post_kwargs(call) -> dict:
    post = AsyncMock()
    post.return_value = MagicMock()
    post.return_value.json.return_value = {"success": True}
    with patch(f"{_MODULE}.GeneralHttpClientProvider") as provider:
        provider.client.return_value.post = post
        asyncio.run(call(SGLangApiClient(server_url="http://engine:30000")))
    return post.await_args.kwargs


def test_pull_weights_bounds_the_read_timeout():
    """A wedged receiver on the engine host must not block the caller forever."""
    kwargs = _post_kwargs(
        lambda client: client.pull_weights(target_version=7, local_checkpoint_dir="/local", source_dir="/shared")
    )
    assert kwargs["timeout"] == WEIGHT_SYNC_RPC_TIMEOUT_SECONDS


def test_update_weights_from_disk_bounds_the_read_timeout():
    kwargs = _post_kwargs(lambda client: client.update_weights_from_disk(model_path="/local"))
    assert kwargs["timeout"] == WEIGHT_SYNC_RPC_TIMEOUT_SECONDS


def test_pause_and_continue_generation_bound_the_read_timeout():
    for call in (lambda client: client.pause_generation(), lambda client: client.continue_generation()):
        assert _post_kwargs(call)["timeout"] == WEIGHT_SYNC_RPC_TIMEOUT_SECONDS


def test_engine_rpc_timeout_is_shorter_than_the_gloo_barrier():
    """Rank 0 drives the engine RPCs while every other rank waits at a gloo barrier. If the
    barrier expires first, the ranks die on a timeout that says nothing about the engine that
    actually stalled."""
    assert WEIGHT_SYNC_RPC_TIMEOUT_SECONDS < GLOO_GROUP_TIMEOUT.total_seconds()
