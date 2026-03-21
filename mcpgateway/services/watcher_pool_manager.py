# -*- coding: utf-8 -*-
"""Watcher Pool Manager for proactive tool list refresh via MCP notifications.

This module implements a lazy pool manager that maintains one SSE connection per
active upstream MCP server, listening for notifications/tools/list_changed events
and triggering tool list updates in the gateway registry.

Copyright 2026
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti
"""

# flake8: noqa: DAR101, DAR201, DAR401

# Future
from __future__ import annotations

# Standard
import asyncio
from dataclasses import dataclass, field
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set

# Third-Party
import mcp.types as mcp_types

if TYPE_CHECKING:
    # First-Party
    from mcpgateway.services.mcp_session_pool import MCPSessionPool

logger = logging.getLogger(__name__)


@dataclass
class ServerWatcherState:
    """State for a single server's watcher connection.

    Attributes:
        server_url: The upstream MCP server URL.
        gateway_id: The gateway ID for this watcher.
        ref_count: Number of active sessions using this server.
        last_activity: Timestamp of last session activity.
        watcher_task: The asyncio.Task running the watcher.
        created_at: When this state was created.
    """

    server_url: str
    gateway_id: str
    ref_count: int = 0
    last_activity: float = field(default_factory=time.time)
    watcher_task: Optional[asyncio.Task] = None
    created_at: float = field(default_factory=time.time)

    @property
    def is_idle(self, idle_timeout: float) -> bool:
        """Check if this server is idle.

        Args:
            idle_timeout: Idle timeout in seconds.

        Returns:
            True if ref_count == 0 and idle for > idle_timeout seconds.
        """
        if self.ref_count > 0:
            return False
        return (time.time() - self.last_activity) > idle_timeout

    @property
    def watcher_is_alive(self) -> bool:
        """Check if watcher task is running.

        Returns:
            True if task exists and not done.
        """
        return self.watcher_task is not None and not self.watcher_task.done()


class WatcherPoolManager:
    """Manages lazy SSE watchers per active upstream server.

    One watcher connection per (server_url, gateway_id) tuple. Watchers start
    on first session acquisition and stop when idle timeout expires.

    Attributes:
        idle_timeout_seconds: Seconds before closing idle watcher.
        idle_check_interval_seconds: How often to check idle condition.
        max_reconnect_attempts: Max reconnect retries before giving up.
        max_backoff_seconds: Max wait between reconnect attempts.
    """

    def __init__(
        self,
        idle_timeout_seconds: float = 300.0,
        idle_check_interval_seconds: float = 30.0,
        max_reconnect_attempts: int = 10,
        max_backoff_seconds: float = 60.0,
        tools_refresh_retries: int = 3,
        tools_refresh_backoff: Optional[list[float]] = None,
        sse_connect_timeout: float = 30.0,
        session_pool: Optional[MCPSessionPool] = None,
        gateway_service: Optional[Any] = None,
    ) -> None:
        """Initialize the watcher pool manager.

        Args:
            idle_timeout_seconds: Close watcher after this idle time.
            idle_check_interval_seconds: Idle check frequency.
            max_reconnect_attempts: Max reconnect retries.
            max_backoff_seconds: Max backoff wait.
            tools_refresh_retries: Retries for tools/list fetch.
            tools_refresh_backoff: Backoff delays for retries.
            sse_connect_timeout: Timeout for SSE connection.
            session_pool: MCPSessionPool instance for acquiring sessions.
            gateway_service: GatewayService instance for refresh operations.
        """
        self.idle_timeout_seconds = idle_timeout_seconds
        self.idle_check_interval_seconds = idle_check_interval_seconds
        self.max_reconnect_attempts = max_reconnect_attempts
        self.max_backoff_seconds = max_backoff_seconds
        self.tools_refresh_retries = tools_refresh_retries
        self.tools_refresh_backoff = tools_refresh_backoff or [1.0, 2.0, 4.0]
        self.sse_connect_timeout = sse_connect_timeout
        self._session_pool = session_pool
        self._gateway_service = gateway_service

        # State: key is (server_url, gateway_id)
        self._watchers: Dict[tuple[str, str], ServerWatcherState] = {}
        self._lock = asyncio.Lock()

        # Lifecycle
        self._initialized = False
        self._shutdown_event = asyncio.Event()

        logger.info(
            "WatcherPoolManager initialized: idle_timeout=%ss, idle_check=%ss",
            idle_timeout_seconds,
            idle_check_interval_seconds,
        )

    async def initialize(self) -> None:
        """Initialize the watcher pool manager (async startup hook).

        Sets up shutdown event and flags manager as ready.
        """
        if self._initialized:
            return

        self._shutdown_event.clear()
        self._initialized = True
        logger.info("WatcherPoolManager initialized and ready")

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown manager and cancel all watcher tasks.

        Args:
            timeout: Grace period to wait for tasks to complete.
        """
        if not self._initialized:
            return

        logger.info("WatcherPoolManager shutting down...")
        self._shutdown_event.set()

        async with self._lock:
            # Cancel all watcher tasks
            tasks_to_cancel = [
                state.watcher_task
                for state in self._watchers.values()
                if state.watcher_task is not None
            ]

            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete
            if tasks_to_cancel:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for watcher tasks to complete")

            self._watchers.clear()

        logger.info("WatcherPoolManager shutdown complete")

    async def on_session_acquired(self, server_url: str, gateway_id: Optional[str]) -> None:
        """Called when a session is acquired for a server.

        Increments ref count and starts watcher if needed.

        Args:
            server_url: The upstream server URL.
            gateway_id: The gateway ID (may be None, converted to "").
        """
        if not self._initialized:
            return

        gw_id = gateway_id or ""
        key = (server_url, gw_id)

        async with self._lock:
            # Get or create state
            if key not in self._watchers:
                self._watchers[key] = ServerWatcherState(
                    server_url=server_url,
                    gateway_id=gw_id,
                )

            state = self._watchers[key]
            state.ref_count += 1
            state.last_activity = time.time()

            logger.debug(f"Session acquired for {server_url} (gw:{gw_id[:8] if gw_id else 'none'}): ref_count={state.ref_count}")

            # Start watcher if not already running
            if not state.watcher_is_alive:
                logger.info(f"Starting watcher for {server_url} (gw:{gw_id[:8] if gw_id else 'none'})")
                state.watcher_task = asyncio.create_task(self._run_watcher(server_url, gw_id))

    async def on_session_released(self, server_url: str, gateway_id: Optional[str]) -> None:
        """Called when a session is released.

        Decrements ref count but does NOT stop watcher (idle timeout decides).

        Args:
            server_url: The upstream server URL.
            gateway_id: The gateway ID (may be None, converted to "").
        """
        if not self._initialized:
            return

        gw_id = gateway_id or ""
        key = (server_url, gw_id)

        async with self._lock:
            if key in self._watchers:
                state = self._watchers[key]
                state.ref_count = max(0, state.ref_count - 1)
                state.last_activity = time.time()

                logger.debug(f"Session released for {server_url}: ref_count={state.ref_count}")

    async def on_session_expired(self, server_url: str, gateway_id: Optional[str]) -> None:
        """Called when a session expires.

        Decrements ref count (same as released - let idle timeout decide).

        Args:
            server_url: The upstream server URL.
            gateway_id: The gateway ID (may be None, converted to "").
        """
        if not self._initialized:
            return

        gw_id = gateway_id or ""
        key = (server_url, gw_id)

        async with self._lock:
            if key in self._watchers:
                state = self._watchers[key]
                state.ref_count = max(0, state.ref_count - 1)
                state.last_activity = time.time()

                logger.debug(f"Session expired for {server_url}: ref_count={state.ref_count}")

    async def _run_watcher(self, server_url: str, gateway_id: str) -> None:
        """Long-lived watcher task for a server.

        Runs two concurrent coroutines:
        1. sse_listener - receives notifications and refreshes tools
        2. idle_monitor - checks for idle timeout and cancels listener if met

        Args:
            server_url: The upstream server URL.
            gateway_id: The gateway ID.
        """
        logger.info(f"Watcher task started for {server_url} (gw:{gateway_id[:8] if gateway_id else 'none'})")

        try:
            # Run SSE listener and idle monitor concurrently
            await asyncio.gather(
                self._sse_listener(server_url, gateway_id),
                self._idle_monitor(server_url, gateway_id),
            )
        except asyncio.CancelledError:
            logger.info(f"Watcher task cancelled for {server_url}")
            raise
        except Exception as e:
            logger.exception(f"Error in watcher task for {server_url}: {e}")
        finally:
            # Clean up state when task exits
            async with self._lock:
                key = (server_url, gateway_id)
                if key in self._watchers:
                    del self._watchers[key]
                    logger.info(f"Watcher state cleaned up for {server_url}")

    async def _sse_listener(self, server_url: str, gateway_id: str) -> None:
        """Listen for SSE notifications from upstream server.

        Establishes SSE connection, detects notifications/tools/list_changed,
        and triggers tool list refresh when changes detected.

        Args:
            server_url: The upstream server URL.
            gateway_id: The gateway ID.
        """
        logger.info(f"SSE listener starting for {server_url}")

        if not self._session_pool:
            logger.warning("SSE listener: session pool not available, exiting")
            return

        reconnect_attempt = 0

        while True:
            try:
                # Check if shutdown requested
                if self._shutdown_event.is_set():
                    logger.info(f"SSE listener shutting down for {server_url}")
                    return

                # Acquire a session for SSE transport
                pooled = await self._session_pool.acquire(
                    url=server_url,
                    user_identity="system:watcher",
                    gateway_id=gateway_id,
                    transport_type=None,  # Will use default or SSE
                )

                try:
                    # Listen for notifications on this session
                    await self._listen_for_notifications(server_url, gateway_id, pooled.session)
                    reconnect_attempt = 0  # Reset on success

                except asyncio.CancelledError:
                    logger.info(f"SSE listener cancelled for {server_url}")
                    raise
                except Exception as e:
                    logger.warning(f"SSE listener error for {server_url}: {e}")
                    # Continue to reconnect logic below

                finally:
                    # Release session
                    try:
                        await self._session_pool.release(pooled)
                    except Exception as e:
                        logger.debug(f"Error releasing watcher session: {e}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"Error in SSE listener for {server_url}: {e}")

            # Reconnection logic
            if self._shutdown_event.is_set():
                return

            # Check if server still has active sessions
            async with self._lock:
                key = (server_url, gateway_id)
                if key not in self._watchers or self._watchers[key].ref_count == 0:
                    logger.info(f"SSE listener exiting (no active sessions) for {server_url}")
                    return

            # Try to reconnect with exponential backoff
            if reconnect_attempt < self.max_reconnect_attempts:
                wait_seconds = min(
                    self.max_backoff_seconds,
                    2 ** reconnect_attempt  # 1s, 2s, 4s, 8s, ..., up to max_backoff
                )
                reconnect_attempt += 1

                logger.info(
                    f"Reconnecting to {server_url} (attempt {reconnect_attempt}/"
                    f"{self.max_reconnect_attempts}), waiting {wait_seconds}s"
                )
                await asyncio.sleep(wait_seconds)
            else:
                logger.error(f"Max reconnect attempts exceeded for {server_url}, exiting watcher")
                return

    async def _listen_for_notifications(self, server_url: str, gateway_id: str, session: Any) -> None:
        """Listen for notifications on MCP session.

        Args:
            server_url: The upstream server URL.
            gateway_id: The gateway ID.
            session: The MCP ClientSession to listen on.
        """
        # Note: This would need to iterate over session notifications
        # The actual implementation depends on MCP SDK's notification API
        # For now, this is a placeholder that shows the structure

        logger.debug(f"Listening for notifications from {server_url}")

        # TODO: Implement based on MCP SDK's notification handling
        # Pseudo-code:
        # async for notification in session.notifications():
        #     await self._handle_notification(server_url, gateway_id, notification)

        # Placeholder: wait forever (will be cancelled by idle monitor)
        await asyncio.sleep(float('inf'))

    async def _handle_notification(
        self, server_url: str, gateway_id: str, notification: Any
    ) -> None:
        """Handle a notification from the MCP server.

        Args:
            server_url: The upstream server URL.
            gateway_id: The gateway ID.
            notification: The notification object.
        """
        # Check if it's a tools/list_changed notification
        if not self._is_tools_changed_notification(notification):
            logger.debug(f"Ignoring non-tools notification from {server_url}")
            return

        logger.info(f"Received tools list change from {server_url}")

        if not self._gateway_service:
            logger.warning("Gateway service not available, cannot refresh tools")
            return

        # Refresh tools for this gateway
        await self._refresh_tools_for_server(server_url, gateway_id)

    def _is_tools_changed_notification(self, notification: Any) -> bool:
        """Check if notification is tools/list_changed.

        Args:
            notification: The notification object.

        Returns:
            True if it's a tools list changed notification.
        """
        # Handle mcp_types.ServerNotification
        if isinstance(notification, mcp_types.ServerNotification):
            notification_root = notification.root
        else:
            notification_root = notification

        # Check class name for tools list changed
        root_class = type(notification_root).__name__

        return "ToolListChangedNotification" in root_class or "ToolsListChangedNotification" in root_class

    async def _refresh_tools_for_server(self, server_url: str, gateway_id: str) -> None:
        """Refresh tools list for a server.

        Calls GatewayService._refresh_gateway_tools_resources_prompts() to fetch
        and update the tool registry. Only refreshes tools (not resources/prompts).

        Args:
            server_url: The upstream server URL.
            gateway_id: The gateway ID.
        """
        if not self._gateway_service:
            logger.warning("Cannot refresh tools: gateway_service not available")
            return

        try:
            logger.debug(f"Refreshing tools for {server_url} via GatewayService")

            # Use the existing gateway service method designed for tool refresh
            result = await self._gateway_service._refresh_gateway_tools_resources_prompts(  # pyright: ignore
                gateway_id=gateway_id,
                created_via="watcher_pool",  # Audit trail
                include_resources=False,  # Only refresh tools
                include_prompts=False,
            )

            if result.get("success"):
                logger.info(
                    f"Tools refreshed for {server_url}: "
                    f"added={result.get('tools_added', 0)}, "
                    f"removed={result.get('tools_removed', 0)}, "
                    f"updated={result.get('tools_updated', 0)}"
                )
            else:
                logger.warning(f"Tools refresh failed for {server_url}: {result.get('error')}")

        except Exception as e:
            logger.exception(f"Error refreshing tools for {server_url}: {e}")

    async def _idle_monitor(self, server_url: str, gateway_id: str) -> None:
        """Monitor idle timeout and signal cancellation if met.

        Runs independently of SSE listener, waking every idle_check_interval
        to check if watcher should be shut down.

        Args:
            server_url: The upstream server URL.
            gateway_id: The gateway ID.
        """
        logger.debug(f"Idle monitor starting for {server_url}")
        key = (server_url, gateway_id)

        try:
            while True:
                await asyncio.sleep(self.idle_check_interval_seconds)

                # Check idle condition
                async with self._lock:
                    if key not in self._watchers:
                        break  # State was cleaned up, exit

                    state = self._watchers[key]

                    # Check if idle
                    if state.ref_count == 0 and (time.time() - state.last_activity) > self.idle_timeout_seconds:
                        logger.info(f"Idle timeout reached for {server_url}: closing watcher")
                        if state.watcher_task and not state.watcher_task.done():
                            state.watcher_task.cancel()
                        break

        except asyncio.CancelledError:
            logger.debug(f"Idle monitor cancelled for {server_url}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Return watcher pool statistics.

        Returns:
            Dict with active watchers, total ref count, etc.
        """
        # Note: This is called synchronously, not async, so we can't acquire lock
        # Use this for read-only stats
        return {
            "active_watchers": len(self._watchers),
            "watchers": [
                {
                    "server_url": state.server_url,
                    "gateway_id": state.gateway_id,
                    "ref_count": state.ref_count,
                    "is_alive": state.watcher_is_alive,
                    "idle_seconds": time.time() - state.last_activity,
                }
                for state in self._watchers.values()
            ],
        }


# Module-level singleton instance
_watcher_pool_manager: Optional[WatcherPoolManager] = None


def get_watcher_pool_manager() -> WatcherPoolManager:
    """Get or create the global WatcherPoolManager instance.

    Returns:
        The global WatcherPoolManager instance.

    Raises:
        RuntimeError: If not initialized.
    """
    global _watcher_pool_manager
    if _watcher_pool_manager is None:
        raise RuntimeError("WatcherPoolManager not initialized")
    return _watcher_pool_manager


def init_watcher_pool_manager(
    idle_timeout_seconds: float = 300.0,
    idle_check_interval_seconds: float = 30.0,
    max_reconnect_attempts: int = 10,
    max_backoff_seconds: float = 60.0,
    tools_refresh_retries: int = 3,
    tools_refresh_backoff: Optional[list[float]] = None,
    sse_connect_timeout: float = 30.0,
) -> WatcherPoolManager:
    """Initialize the global WatcherPoolManager singleton.

    Args:
        idle_timeout_seconds: Idle timeout in seconds.
        idle_check_interval_seconds: Idle check frequency.
        max_reconnect_attempts: Max reconnect retries.
        max_backoff_seconds: Max backoff wait.
        tools_refresh_retries: Retries for tools/list fetch.
        tools_refresh_backoff: Backoff delays.
        sse_connect_timeout: SSE connection timeout.

    Returns:
        The initialized WatcherPoolManager instance.
    """
    global _watcher_pool_manager
    if _watcher_pool_manager is not None:
        raise RuntimeError("WatcherPoolManager already initialized")

    _watcher_pool_manager = WatcherPoolManager(
        idle_timeout_seconds=idle_timeout_seconds,
        idle_check_interval_seconds=idle_check_interval_seconds,
        max_reconnect_attempts=max_reconnect_attempts,
        max_backoff_seconds=max_backoff_seconds,
        tools_refresh_retries=tools_refresh_retries,
        tools_refresh_backoff=tools_refresh_backoff,
        sse_connect_timeout=sse_connect_timeout,
    )
    return _watcher_pool_manager
