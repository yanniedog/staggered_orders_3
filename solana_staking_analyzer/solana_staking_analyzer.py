#!/usr/bin/env python3
"""Solana staking performance analyzer.

Reads a wallet configuration, discovers native stake accounts, collects
historical rewards, and evaluates validator performance relative to expected
network yield and validator commission. Outputs a console summary as well as
JSON/CSV artifacts for downstream analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import httpx
import pandas as pd
from solders.pubkey import Pubkey


LAMPORTS_PER_SOL = 1_000_000_000
STAKE_PROGRAM_ID = "Stake11111111111111111111111111111111111111"
STAKE_INSTRUCTION_KEYWORDS = {
    "initialize",
    "delegate",
    "authorize",
    "authorizewithseed",
    "deactivate",
    "withdraw",
    "split",
    "merge",
}
MAX_RPC_RETRIES = 8
MAX_ENDPOINT_FAILURES = 3
RETRY_BACKOFF_SECONDS = 1.0
MAX_RETRY_BACKOFF_SECONDS = 12.0
RETRY_BACKOFF_JITTER = 0.25
RATE_LIMIT_STATUS_CODES = {429}
CONFIG_FILENAME = "config.cfg"
ENDPOINT_ROTATION_STATUS_CODES = {429, 503}
SOLSCAN_API_BASE = "https://pro-api.solscan.io/v2.0"
SOLSCAN_DEFAULT_PAGE_SIZE = 100
SOLSCAN_MAX_RETRIES = 6
SOLSCAN_RATE_LIMIT_STATUS_CODES = {429}
SOLSCAN_RETRY_BACKOFF_SECONDS = 0.75
SOLSCAN_MAX_RETRY_BACKOFF_SECONDS = 8.0
SOLSCAN_RETRY_BACKOFF_JITTER = 0.2


def try_extract_stake_accounts_from_instruction(instruction: Dict[str, Any]) -> List[str]:
    if not isinstance(instruction, dict):
        return []

    candidates: Set[str] = set()
    name = (
        instruction.get("name")
        or instruction.get("instruction_name")
        or instruction.get("type")
        or ""
    )
    name_lower = str(name).lower()
    if not any(keyword in name_lower for keyword in STAKE_INSTRUCTION_KEYWORDS):
        # Continue examining accounts even if the instruction label is unfamiliar.
        pass

    params = instruction.get("params")
    if isinstance(params, dict):
        for key in ("stakeAccount", "stake_account", "stake", "account", "source", "destination"):
            value = params.get(key)
            if isinstance(value, str) and len(value) >= 32:
                candidates.add(value)

    for key in ("stakeAccount", "stake_account", "stake", "account"):
        value = instruction.get(key)
        if isinstance(value, str) and len(value) >= 32:
            candidates.add(value)

    accounts_field = instruction.get("accounts") or instruction.get("account_keys") or []
    if isinstance(accounts_field, list):
        for entry in accounts_field:
            if isinstance(entry, dict):
                role = str(entry.get("name") or entry.get("role") or "").lower()
                pubkey = entry.get("pubkey") or entry.get("address") or entry.get("pubKey")
                if isinstance(pubkey, str) and len(pubkey) >= 32:
                    if not role or "stake" in role or role in {"stakeaccount", "stake_account"}:
                        candidates.add(pubkey)
            elif isinstance(entry, str) and len(entry) >= 32:
                candidates.add(entry)

    return list(candidates)


DEFAULT_PRIMARY_RPC_ENDPOINTS: Sequence[str] = (
    "https://api.mainnet-beta.solana.com",
    "https://ssc-dao.genesysgo.net",
)
DEFAULT_SECONDARY_RPC_ENDPOINTS: Sequence[str] = (
    "https://solana-api.projectserum.com",
    "https://rpc.ankr.com/solana",
)
DEFAULT_TERTIARY_RPC_ENDPOINTS: Sequence[str] = (
    "https://solana.public-rpc.com",
)


class ConfigurationError(RuntimeError):
    """Raised when configuration is invalid."""


class RPCError(RuntimeError):
    """Raised when the Solana RPC returns an error response."""


def lamports_to_sol(value: Optional[int]) -> float:
    return 0.0 if value in (None, 0) else value / LAMPORTS_PER_SOL


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def summarize_payload(payload: Dict[str, Any], limit: int = 800) -> str:
    try:
        serialized = json.dumps(payload, default=str)
    except TypeError:
        serialized = str(payload)
    if len(serialized) > limit:
        return serialized[: limit - 3] + "..."
    return serialized


def summarize_result(result: Any, limit: int = 400) -> str:
    try:
        serialized = json.dumps(result, default=str)
    except TypeError:
        serialized = str(result)
    if len(serialized) > limit:
        return serialized[: limit - 3] + "..."
    return serialized


def normalize_epoch(value: Any) -> Optional[int]:
    epoch = safe_int(value)
    if epoch is None:
        return None
    if epoch in {2**64 - 1, 2**32 - 1}:
        return None
    return epoch


@dataclass
class RewardRecord:
    epoch: int
    amount_lamports: int
    post_balance_lamports: int
    effective_slot: Optional[int]
    commission: Optional[int]
    block_time: Optional[int] = None

    @property
    def amount_sol(self) -> float:
        return lamports_to_sol(self.amount_lamports)


@dataclass
class StakeAccount:
    pubkey: str
    delegated_lamports: int
    validator_vote: Optional[str]
    activation_epoch: Optional[int]
    deactivation_epoch: Optional[int]
    authorized_staker: Optional[str]
    authorized_withdrawer: Optional[str]
    rent_exempt_reserve_lamports: Optional[int]
    stake_type: Optional[str]
    account_balance_lamports: Optional[int]
    lockup: Dict[str, Any] = field(default_factory=dict)
    rewards: List[RewardRecord] = field(default_factory=list)
    first_reward_slot: Optional[int] = None
    last_reward_slot: Optional[int] = None
    first_reward_time: Optional[datetime] = None
    last_reward_time: Optional[datetime] = None
    total_rewards_lamports: int = 0
    duration_days: Optional[float] = None
    realized_yield_pct: Optional[float] = None
    annualized_yield_pct: Optional[float] = None
    expected_yield_pct: Optional[float] = None
    validator_commission: Optional[int] = None
    validator_uptime_ratio: Optional[float] = None
    validator_delinquent: Optional[bool] = None
    delegation_history: List["DelegationEvent"] = field(default_factory=list)
    historical_validators: List[str] = field(default_factory=list)
    first_seen_unix: Optional[int] = None
    last_seen_unix: Optional[int] = None
    history_signature_count: int = 0

    @property
    def delegated_sol(self) -> float:
        return lamports_to_sol(self.delegated_lamports)

    @property
    def total_rewards_sol(self) -> float:
        return lamports_to_sol(self.total_rewards_lamports)


class SolanaRPCClient:
    def __init__(
        self,
        endpoint_groups: Sequence[Sequence[str]],
        concurrency_limit: int,
        timeout: float = 35.0,
    ) -> None:
        if not endpoint_groups:
            raise ConfigurationError("At least one RPC endpoint must be provided")

        normalized_groups: List[List[str]] = []
        for group in endpoint_groups:
            normalized_group: List[str] = []
            for endpoint in group:
                value = endpoint.strip() if isinstance(endpoint, str) else str(endpoint).strip()
                if not value:
                    continue
                if value not in normalized_group:
                    normalized_group.append(value)
            if normalized_group:
                normalized_groups.append(normalized_group)

        if not normalized_groups:
            raise ConfigurationError("No usable RPC endpoints after normalization")

        self._endpoint_groups = normalized_groups
        self._group_index = 0
        self.endpoints = self._endpoint_groups[self._group_index]
        self._current_index = 0
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        self._request_id = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self._unhealthy: Dict[str, float] = {}
        self._endpoint_failures: Dict[str, int] = {}

    async def __aenter__(self) -> "SolanaRPCClient":
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def request(self, method: str, params: Optional[List[Any]] = None) -> Any:
        if self._client is None:
            raise RuntimeError("RPC client not initialized; use async context manager")

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or [],
        }
        payload_summary = summarize_payload(payload)

        attempt = 0
        while True:
            attempt += 1
            async with self._semaphore:
                delay: Optional[float] = None
                try:
                    endpoint = self._ensure_active_endpoint()
                    self.logger.info(
                        "RPC Request -> method=%s attempt=%d endpoint=%s payload=%s",
                        method,
                        attempt,
                        endpoint,
                        payload_summary,
                    )
                    response = await self._client.post(endpoint, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    self.logger.info(
                        "RPC Response <- method=%s attempt=%d status=%s endpoint=%s body=%s",
                        method,
                        attempt,
                        response.status_code,
                        endpoint,
                        summarize_result(data),
                    )
                except httpx.HTTPStatusError as exc:
                    endpoint = self._ensure_active_endpoint()
                    status_code = exc.response.status_code if exc.response is not None else None
                    delay = self._compute_retry_delay(attempt, status_code)
                    if status_code in RATE_LIMIT_STATUS_CODES:
                        self.logger.info(
                            "Rate limit on %s attempt=%d via %s; backing off %.2fs",
                            method,
                            attempt,
                            endpoint,
                            delay,
                        )
                    else:
                        self.logger.warning("HTTP error on %s attempt %d via %s: %s", method, attempt, endpoint, exc)
                    if status_code in ENDPOINT_ROTATION_STATUS_CODES:
                        self._mark_endpoint_unhealthy(endpoint, 5.0)
                        rotated = self._rotate_endpoint()
                        if rotated:
                            self.logger.warning("Switching RPC endpoint to %s after status %s", rotated, status_code)
                    else:
                        self._mark_endpoint_unhealthy(endpoint, 15.0)
                        self._record_endpoint_failure(endpoint)
                    if attempt >= MAX_RPC_RETRIES:
                        raise RPCError(f"HTTP error on method {method}: {exc}") from exc
                except httpx.RequestError as exc:
                    endpoint = self._ensure_active_endpoint()
                    self.logger.warning("Request error on %s attempt %d via %s: %s", method, attempt, endpoint, exc)
                    self._mark_endpoint_unhealthy(endpoint, 30.0)
                    fatal = isinstance(getattr(exc, "__cause__", None), socket.gaierror)
                    if not fatal:
                        message = str(exc).lower()
                        fatal = "getaddrinfo failed" in message or "dns" in message
                    self._record_endpoint_failure(endpoint, fatal=fatal)
                    rotated = self._rotate_endpoint()
                    if rotated:
                        self.logger.warning("Switching RPC endpoint to %s after request error", rotated)
                    if attempt >= MAX_RPC_RETRIES:
                        raise RPCError(f"Request error on method {method}: {exc}") from exc
                    delay = self._compute_retry_delay(attempt, None)
                else:
                    if "error" in data:
                        message = data["error"].get("message", "Unknown RPC error")
                        self.logger.warning("RPC error on %s attempt %d: %s", method, attempt, message)
                        lower_message = message.lower()
                        if "cleaned up" in lower_message or "does not exist on node" in lower_message:
                            endpoint = self._ensure_active_endpoint()
                            self._mark_endpoint_unhealthy(endpoint, 60.0)
                            self._record_endpoint_failure(endpoint)
                            rotated = self._rotate_endpoint()
                            if rotated:
                                self.logger.warning(
                                    "Switching RPC endpoint to %s after history gap on %s",
                                    rotated,
                                    endpoint,
                                )
                        if attempt >= MAX_RPC_RETRIES:
                            raise RPCError(f"RPC error on method {method}: {message}")
                        delay = self._compute_retry_delay(attempt, None)
                    else:
                        if endpoint in self._unhealthy:
                            self._unhealthy.pop(endpoint, None)
                        self._cleanup_unhealthy()
                        return data.get("result")

            if attempt >= MAX_RPC_RETRIES:
                break
            if delay is None:
                delay = self._compute_retry_delay(attempt, None)
            await asyncio.sleep(delay)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _ensure_active_endpoint(self) -> str:
        if not self.endpoints:
            if not self._promote_group():
                raise RPCError("All RPC endpoints exhausted")
        if self._current_index >= len(self.endpoints):
            self._current_index = 0
        if not self.endpoints:
            raise RPCError("All RPC endpoints exhausted")
        return self.endpoints[self._current_index]

    def _rotate_endpoint(self) -> Optional[str]:
        if not self.endpoints:
            promoted = self._promote_group()
            return self.endpoints[self._current_index] if promoted and self.endpoints else None
        self._cleanup_unhealthy()
        starting_index = self._current_index
        for _ in range(len(self.endpoints)):
            self._current_index = (self._current_index + 1) % len(self.endpoints)
            endpoint = self.endpoints[self._current_index]
            expiry = self._unhealthy.get(endpoint)
            if expiry is not None and time.monotonic() < expiry:
                continue
            return endpoint
        self._current_index = starting_index
        promoted = self._promote_group()
        return self.endpoints[self._current_index] if promoted and self.endpoints else None

    def _promote_group(self) -> bool:
        if self._group_index + 1 >= len(self._endpoint_groups):
            self.logger.error("No additional RPC endpoint groups available")
            return False
        self._group_index += 1
        self.endpoints = self._endpoint_groups[self._group_index]
        self._current_index = 0
        self.logger.warning(
            "Promoting to fallback RPC endpoint group %d (%d endpoints)",
            self._group_index + 1,
            len(self.endpoints),
        )
        return bool(self.endpoints)

    def _mark_endpoint_unhealthy(self, endpoint: str, duration: float) -> None:
        self._unhealthy[endpoint] = time.monotonic() + max(duration, 1.0)

    def _cleanup_unhealthy(self) -> None:
        now = time.monotonic()
        expired = [endpoint for endpoint, expiry in self._unhealthy.items() if expiry <= now]
        for endpoint in expired:
            self._unhealthy.pop(endpoint, None)

    def _record_endpoint_failure(self, endpoint: Optional[str], fatal: bool = False) -> None:
        if not endpoint:
            return
        count = self._endpoint_failures.get(endpoint, 0) + 1
        self._endpoint_failures[endpoint] = count
        if fatal or count >= MAX_ENDPOINT_FAILURES:
            if endpoint in self.endpoints:
                try:
                    index = self.endpoints.index(endpoint)
                except ValueError:
                    self._endpoint_failures.pop(endpoint, None)
                    return
                self.logger.error(
                    "Removing RPC endpoint %s after %d consecutive failures", endpoint, count
                )
                self.endpoints.pop(index)
                self._unhealthy.pop(endpoint, None)
                self._endpoint_failures.pop(endpoint, None)
                if not self.endpoints:
                    promoted = self._promote_group()
                    if not promoted:
                        self.logger.error("All RPC endpoint groups exhausted after removing %s", endpoint)
                elif self._current_index >= len(self.endpoints):
                    self._current_index = 0
            else:
                self._endpoint_failures.pop(endpoint, None)

    def _compute_retry_delay(self, attempt: int, status_code: Optional[int]) -> float:
        base = RETRY_BACKOFF_SECONDS
        if status_code in RATE_LIMIT_STATUS_CODES:
            backoff = base * (2 ** (attempt - 1))
        else:
            backoff = base * attempt
        delay = min(backoff, MAX_RETRY_BACKOFF_SECONDS)
        jitter = random.uniform(0.0, RETRY_BACKOFF_JITTER)
        return delay + jitter


class SolscanClient:
    def __init__(
        self,
        api_key: Optional[str],
        concurrency_limit: int,
        base_url: str = SOLSCAN_API_BASE,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._headers = {"accept": "application/json"}
        if api_key:
            self._headers["token"] = api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(max(1, concurrency_limit))
        self._timeout = timeout
        self._base_url = base_url.rstrip("/") or SOLSCAN_API_BASE
        self.logger = logging.getLogger(self.__class__.__name__)

    async def __aenter__(self) -> "SolscanClient":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Solscan client not initialized; use async context manager")
        normalized_path = path if path.startswith("/") else f"/{path}"
        attempt = 0
        while True:
            attempt += 1
            async with self._semaphore:
                try:
                    response = await self._client.get(normalized_path, params=params, headers=self._headers)
                    if response.status_code in SOLSCAN_RATE_LIMIT_STATUS_CODES:
                        raise httpx.HTTPStatusError("rate limited", request=response.request, response=response)
                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload, dict):
                        return payload
                    return {"data": payload}
                except httpx.HTTPStatusError as exc:
                    delay = self._compute_retry_delay(attempt)
                    status_code = exc.response.status_code if exc.response else None
                    self.logger.warning(
                        "Solscan HTTP error on %s attempt=%d status=%s params=%s", normalized_path, attempt, status_code, params
                    )
                    if attempt >= SOLSCAN_MAX_RETRIES:
                        raise
                    await asyncio.sleep(delay)
                except httpx.RequestError as exc:
                    delay = self._compute_retry_delay(attempt)
                    self.logger.warning(
                        "Solscan request error on %s attempt=%d params=%s error=%s",
                        normalized_path,
                        attempt,
                        params,
                        exc,
                    )
                    if attempt >= SOLSCAN_MAX_RETRIES:
                        raise
                    await asyncio.sleep(delay)

    async def paginate(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data_key: str = "data",
        page_size: int = SOLSCAN_DEFAULT_PAGE_SIZE,
    ) -> List[Any]:
        results: List[Any] = []
        page = 1
        params = dict(params or {})
        while True:
            params.update({"page": page, "page_size": page_size})
            payload = await self.get(path, params=params)
            data = payload.get(data_key) if isinstance(payload, dict) else None
            if not data:
                break
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
            page += 1
        return results

    def _compute_retry_delay(self, attempt: int) -> float:
        backoff = SOLSCAN_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
        delay = min(backoff, SOLSCAN_MAX_RETRY_BACKOFF_SECONDS)
        jitter = random.uniform(0.0, SOLSCAN_RETRY_BACKOFF_JITTER)
        return delay + jitter


def load_config(config_path: Path, wallet_config_path: Optional[Path] = None) -> Dict[str, Any]:
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        try:
            config = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON in configuration file: {exc}") from exc

    wallet_config_path = wallet_config_path or (config_path.parent / "wallet_config.json")
    if not wallet_config_path.exists():
        raise ConfigurationError(
            f"Wallet configuration file not found at {wallet_config_path}. "
            "Create it (optionally by copying wallet_config.example.cfg) and set 'wallet_address'."
        )
    with wallet_config_path.open("r", encoding="utf-8") as wallet_handle:
        try:
            wallet_config = json.load(wallet_handle)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON in wallet configuration file: {exc}") from exc

    wallet_address = wallet_config.get("wallet_address")
    if not wallet_address:
        raise ConfigurationError("Wallet configuration must include 'wallet_address'.")
    try:
        Pubkey.from_string(wallet_address)
    except Exception as exc:  # pylint: disable=broad-except
        raise ConfigurationError(f"Invalid wallet address: {wallet_address}") from exc
    config["wallet_address"] = wallet_address

    data_source = str(config.get("data_source", "solscan")).strip().lower()
    if data_source not in {"solscan", "rpc"}:
        raise ConfigurationError("'data_source' must be either 'solscan' or 'rpc'.")
    config["data_source"] = data_source

    solscan_api_key = str(config.get("solscan_api_key") or os.getenv("SOLSCAN_API_KEY") or "").strip() or None
    if solscan_api_key:
        config["solscan_api_key"] = solscan_api_key
    solscan_base_url = str(config.get("solscan_base_url") or SOLSCAN_API_BASE).strip().rstrip("/")
    config["solscan_base_url"] = solscan_base_url or SOLSCAN_API_BASE
    config["solscan_page_size"] = safe_int(config.get("solscan_page_size")) or SOLSCAN_DEFAULT_PAGE_SIZE

    rpc_endpoint = config.get("rpc_endpoint")
    rpc_endpoints = config.get("rpc_endpoints")
    fallback_endpoints = config.get("rpc_fallback_endpoints")

    if data_source == "rpc":
        def _normalize_group(candidates: Optional[Sequence[Any]]) -> List[str]:
            group: List[str] = []
            if not candidates:
                return group
            for endpoint in candidates:
                if isinstance(endpoint, str):
                    value = endpoint.strip()
                else:
                    value = str(endpoint).strip()
                if not value or value in seen_endpoints:
                    continue
                group.append(value)
                seen_endpoints.add(value)
            return group

        seen_endpoints: set[str] = set()
        endpoint_groups: List[List[str]] = []

        resolved_user: Sequence[Any]
        if rpc_endpoints and isinstance(rpc_endpoints, list):
            resolved_user = rpc_endpoints
        elif isinstance(rpc_endpoint, str) and rpc_endpoint:
            resolved_user = [rpc_endpoint]
        else:
            resolved_user = []

        user_group = _normalize_group(resolved_user)
        if user_group:
            endpoint_groups.append(user_group)

        fallback_group = _normalize_group(fallback_endpoints if isinstance(fallback_endpoints, list) else [])
        if fallback_group:
            endpoint_groups.append(fallback_group)

        primary_group = _normalize_group(DEFAULT_PRIMARY_RPC_ENDPOINTS)
        if primary_group:
            endpoint_groups.append(primary_group)

        secondary_group = _normalize_group(DEFAULT_SECONDARY_RPC_ENDPOINTS)
        if secondary_group:
            endpoint_groups.append(secondary_group)

        tertiary_group = _normalize_group(DEFAULT_TERTIARY_RPC_ENDPOINTS)
        if tertiary_group:
            endpoint_groups.append(tertiary_group)

        endpoint_groups = [group for group in endpoint_groups if group]
        if not endpoint_groups:
            raise ConfigurationError("No valid RPC endpoints configured or available as fallbacks.")

        flattened_endpoints = [endpoint for group in endpoint_groups for endpoint in group]
        config["rpc_endpoint_groups"] = endpoint_groups
        config["rpc_endpoints"] = flattened_endpoints
        config["rpc_endpoint"] = flattened_endpoints[0]
    else:
        config["rpc_endpoint_groups"] = []
        config["rpc_endpoints"] = []
        config["rpc_endpoint"] = None

    concurrency_limit = safe_int(config.get("concurrency_limit")) or 4
    config["concurrency_limit"] = max(1, concurrency_limit)

    config.setdefault("output_format", ["console", "json", "csv"])
    config.setdefault("json_output_file", "validator_analysis.json")
    config.setdefault("csv_output_file", "validator_analysis.csv")
    config.setdefault("stake_accounts_csv_output_file", "stake_account_analysis.csv")
    config.setdefault("max_transaction_signatures", 2000)
    transaction_signature_limit = safe_int(config.get("max_transaction_signatures"))
    if "max_stake_signatures" not in config:
        config["max_stake_signatures"] = transaction_signature_limit or 5000
    if "max_wallet_signatures" not in config:
        wallet_default = (transaction_signature_limit or 2000) * 2
        config["max_wallet_signatures"] = max(wallet_default, 5000)
    config.setdefault("transaction_backoff_seconds", 0.35)
    config.setdefault("log_level", "INFO")

    return config


async def fetch_stake_accounts(client: SolanaRPCClient, wallet_address: str, logger: logging.Logger) -> Dict[str, StakeAccount]:
    logger.info("Discovering stake accounts for wallet %s", wallet_address)
    accounts: Dict[str, StakeAccount] = {}

    async def _query_with_offset(offset: int) -> List[Dict[str, Any]]:
        params = [
            STAKE_PROGRAM_ID,
            {
                "encoding": "jsonParsed",
                "filters": [
                    {"memcmp": {"offset": offset, "bytes": wallet_address}},
                ],
            },
        ]
        result = await client.request("getProgramAccounts", params)
        return result or []

    fetched_accounts: List[Dict[str, Any]] = []
    for offset in (12, 44):  # staker and withdrawer offsets
        subset = await _query_with_offset(offset)
        fetched_accounts.extend(subset)
        logger.debug("Found %d stake accounts at offset %d", len(subset), offset)

    if not fetched_accounts:
        logger.warning("No stake accounts found for wallet %s", wallet_address)
        return accounts

    for entry in fetched_accounts:
        pubkey = entry.get("pubkey")
        if not pubkey:
            continue

        account_info = entry.get("account", {})
        parsed = (account_info.get("data") or {}).get("parsed")
        if not parsed or parsed.get("type") not in {"delegated", "initialized", "stake"}:
            continue

        info = parsed.get("info", {})
        stake = info.get("stake", {})
        delegation = stake.get("delegation") or {}
        validator_vote = (
            delegation.get("voteAccount")
            or delegation.get("voter")
            or delegation.get("votePubkey")
        )
        delegated_lamports = safe_int(delegation.get("stake")) or 0
        activation_epoch = normalize_epoch(delegation.get("activationEpoch"))
        deactivation_epoch = normalize_epoch(delegation.get("deactivationEpoch"))

        meta = info.get("meta", {})
        authorized = meta.get("authorized", {})
        rent_reserve = safe_int(meta.get("rentExemptReserve"))

        stake_account = accounts.get(pubkey)
        if stake_account is None:
            stake_account = StakeAccount(
                pubkey=pubkey,
                delegated_lamports=delegated_lamports,
                validator_vote=validator_vote,
                activation_epoch=activation_epoch,
                deactivation_epoch=deactivation_epoch,
                authorized_staker=authorized.get("staker"),
                authorized_withdrawer=authorized.get("withdrawer"),
                rent_exempt_reserve_lamports=rent_reserve,
                stake_type=parsed.get("type"),
                account_balance_lamports=safe_int(account_info.get("lamports")),
                lockup=meta.get("lockup", {}),
            )
            accounts[pubkey] = stake_account
        else:
            # Update dynamic fields in case newer data is present.
            stake_account.delegated_lamports = max(stake_account.delegated_lamports, delegated_lamports)
            stake_account.validator_vote = stake_account.validator_vote or validator_vote
            stake_account.activation_epoch = stake_account.activation_epoch or activation_epoch
            stake_account.deactivation_epoch = stake_account.deactivation_epoch or deactivation_epoch

    logger.info("Discovered %d unique stake accounts", len(accounts))
    return accounts


@dataclass
class DelegationEvent:
    signature: str
    slot: Optional[int]
    block_time: Optional[int]
    vote_account: str
    instruction_type: Optional[str]
    stake_account: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "slot": self.slot,
            "block_time": self.block_time,
            "block_time_iso": datetime.fromtimestamp(self.block_time, tz=timezone.utc).isoformat() if self.block_time else None,
            "vote_account": self.vote_account,
            "instruction_type": self.instruction_type,
            "stake_account": self.stake_account,
        }


def extract_delegate_events(
    tx: Dict[str, Any],
    default_stake_account: Optional[str],
    signature: str,
) -> List[DelegationEvent]:
    events: List[DelegationEvent] = []
    block_time = safe_int(tx.get("blockTime"))
    slot = safe_int(tx.get("slot"))

    transaction = tx.get("transaction") or {}
    message = transaction.get("message") if isinstance(transaction, dict) else {}
    instructions: List[Dict[str, Any]] = []
    if isinstance(message, dict):
        instructions.extend(message.get("instructions") or [])

    meta = tx.get("meta") or {}
    for inner in meta.get("innerInstructions") or []:
        instructions.extend(inner.get("instructions") or [])

    for instruction in instructions:
        parsed = instruction.get("parsed") or {}
        if not parsed:
            continue
        program = instruction.get("program") or instruction.get("programId") or instruction.get("program_id")
        if program not in {"stake", STAKE_PROGRAM_ID}:
            continue
        info = parsed.get("info") or {}
        instruction_type = parsed.get("type") or instruction.get("type")
        if instruction_type and "delegate" not in str(instruction_type).lower():
            continue
        vote_account = (
            info.get("voteAccount")
            or info.get("vote_account")
            or info.get("votePubkey")
            or info.get("vote_pubkey")
        )
        if not vote_account:
            continue
        stake_account = (
            info.get("stakeAccount")
            or info.get("stake_account")
            or info.get("stakePubkey")
            or info.get("stake_pubkey")
            or default_stake_account
        )
        if not stake_account:
            continue
        events.append(
            DelegationEvent(
                signature=signature,
                slot=slot,
                block_time=block_time,
                vote_account=vote_account,
                instruction_type=instruction_type,
                stake_account=stake_account,
            )
        )
    return events


def _extract_string(entry: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and len(value) >= 32:
            return value
    return None


def _update_history_meta(
    history: Dict[str, Dict[str, Any]],
    address: str,
    block_time: Optional[int],
    signature: str,
) -> None:
    meta = history.setdefault(address, {"first_seen": None, "last_seen": None, "seen_in": []})
    if block_time is not None:
        first = meta.get("first_seen")
        last = meta.get("last_seen")
        meta["first_seen"] = min(first, block_time) if first is not None else block_time
        meta["last_seen"] = max(last, block_time) if last is not None else block_time
    seen_in = meta.setdefault("seen_in", [])
    if signature not in seen_in:
        seen_in.append(signature)


async def solscan_fetch_current_stake_accounts(
    client: SolscanClient,
    wallet_address: str,
    page_size: int,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    params = {"address": wallet_address}
    records = await client.paginate("/account/stake", params=params, page_size=page_size)
    result: Dict[str, Dict[str, Any]] = {}
    for entry in records:
        if not isinstance(entry, dict):
            continue
        address = _extract_string(entry, ("address", "stake_address", "stakeAccount", "stake_account"))
        if address:
            result[address] = entry
    logger.info("Solscan returned %d current stake accounts for %s", len(result), wallet_address)
    return result


async def solscan_fetch_transactions_for_address(
    client: SolscanClient,
    address: str,
    limit: Optional[int],
    page_size: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    if limit is not None and limit <= 0:
        return []
    collected: List[Dict[str, Any]] = []
    page = 1
    while True:
        effective_page_size = page_size
        if limit is not None:
            remaining = limit - len(collected)
            if remaining <= 0:
                break
            effective_page_size = min(page_size, remaining)
        params = {"address": address, "page": page, "page_size": effective_page_size}
        payload = await client.get("/account/transactions", params=params)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not data:
            break
        if isinstance(data, list):
            collected.extend(data)
        else:
            collected.append(data)
        if limit is not None and len(collected) >= limit:
            break
        page += 1
    logger.info("Fetched %d Solscan transactions for address %s", len(collected), address)
    if limit is not None and len(collected) > limit:
        return collected[:limit]
    return collected


async def solscan_fetch_transaction_detail(
    client: SolscanClient,
    signature: str,
    cache: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    if signature in cache:
        return cache[signature]
    try:
        payload = await client.get("/transaction/detail", params={"tx": signature})
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to load Solscan transaction %s: %s", signature, exc)
        return None
    detail = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        cache[signature] = detail
    else:
        cache[signature] = None
    return cache[signature]


async def solscan_decode_account(
    client: SolscanClient,
    address: str,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    try:
        payload = await client.get("/account/data-decoded", params={"address": address})
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to decode Solscan account %s: %s", address, exc)
        return None
    decoded = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(decoded, dict):
        return None
    return decoded


def parse_stake_account_from_decoded(
    address: str,
    decoded: Optional[Dict[str, Any]],
    fallback_entry: Optional[Dict[str, Any]] = None,
) -> StakeAccount:
    lamports = None
    parsed: Optional[Dict[str, Any]] = None
    if decoded:
        account = decoded.get("account") or {}
        if isinstance(account, dict):
            lamports = safe_int(account.get("lamports"))
        data_section = decoded.get("data") or {}
        if isinstance(data_section, dict) and isinstance(data_section.get("parsed"), dict):
            parsed = data_section.get("parsed")
        elif isinstance(decoded.get("parsed"), dict):
            parsed = decoded.get("parsed")

    stake_type = parsed.get("type") if isinstance(parsed, dict) else None
    info = parsed.get("info") if isinstance(parsed, dict) else {}
    if not isinstance(info, dict):
        info = {}
    meta = info.get("meta") if isinstance(info.get("meta"), dict) else {}
    stake_data = info.get("stake") if isinstance(info.get("stake"), dict) else {}
    delegation = stake_data.get("delegation") if isinstance(stake_data.get("delegation"), dict) else {}

    validator_vote = _extract_string(
        delegation,
        ("voteAccount", "voter", "votePubkey", "votePubkeyAddress", "vote_pubkey"),
    )
    delegated_lamports = safe_int(delegation.get("stake")) or 0
    activation_epoch = normalize_epoch(
        delegation.get("activationEpoch")
        or delegation.get("activation_epoch")
    )
    deactivation_epoch = normalize_epoch(
        delegation.get("deactivationEpoch")
        or delegation.get("deactivation_epoch")
    )

    authorized_section = meta.get("authorized") if isinstance(meta.get("authorized"), dict) else {}
    rent_reserve = safe_int(meta.get("rentExemptReserve") or meta.get("rent_exempt_reserve"))
    lockup = meta.get("lockup") if isinstance(meta.get("lockup"), dict) else {}

    fallback_validator = None
    if not validator_vote and fallback_entry:
        fallback_validator = _extract_string(
            fallback_entry,
            ("vote_address", "validator_vote", "validator", "validatorVoteAccount", "voteAccount"),
        )

    account_balance_lamports = lamports if lamports is not None else safe_int(
        (fallback_entry or {}).get("balance")
    )

    stake_account = StakeAccount(
        pubkey=address,
        delegated_lamports=delegated_lamports,
        validator_vote=validator_vote or fallback_validator,
        activation_epoch=activation_epoch,
        deactivation_epoch=deactivation_epoch,
        authorized_staker=authorized_section.get("staker"),
        authorized_withdrawer=authorized_section.get("withdrawer"),
        rent_exempt_reserve_lamports=rent_reserve,
        stake_type=stake_type or (fallback_entry or {}).get("status"),
        account_balance_lamports=account_balance_lamports,
        lockup=lockup,
    )

    if stake_account.stake_type is None:
        if fallback_entry and fallback_entry.get("status"):
            stake_account.stake_type = fallback_entry.get("status")
        elif decoded is None:
            stake_account.stake_type = "historical_only"

    return stake_account


async def solscan_fetch_rewards_for_account(
    client: SolscanClient,
    address: str,
    page_size: int,
    logger: logging.Logger,
) -> List[RewardRecord]:
    records: List[RewardRecord] = []
    page = 1
    while True:
        params = {
            "address": address,
            "page": page,
            "page_size": page_size,
            "type": "stake",
            "category": "stake",
        }
        try:
            payload = await client.get("/account/reward", params=params)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load Solscan rewards for %s (page %d): %s", address, page, exc)
            break
        data = payload.get("data") if isinstance(payload, dict) else None
        if not data:
            break
        rows: List[Any]
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("data") or data.get("rewards") or []
        else:
            rows = []
        if not rows:
            break
        for row in rows:
            if not isinstance(row, dict):
                continue
            epoch = safe_int(row.get("epoch"))
            amount = safe_int(row.get("amount")) or 0
            post_balance = safe_int(row.get("post_balance") or row.get("postBalance")) or 0
            effective_slot = safe_int(row.get("slot") or row.get("effective_slot") or row.get("effectiveSlot"))
            commission = safe_int(row.get("commission"))
            record = RewardRecord(
                epoch=epoch or 0,
                amount_lamports=amount,
                post_balance_lamports=post_balance,
                effective_slot=effective_slot,
                commission=commission,
            )
            block_time = safe_int(row.get("block_time") or row.get("blockTime"))
            if block_time:
                record.block_time = block_time
            records.append(record)
        if len(rows) < page_size:
            break
        page += 1
    logger.info("Loaded %d Solscan reward records for %s", len(records), address)
    return records


async def solscan_populate_rewards(
    client: SolscanClient,
    stake_accounts: Dict[str, StakeAccount],
    page_size: int,
    average_slot_time_seconds: float,
    slots_in_epoch: Optional[int],
    logger: logging.Logger,
) -> None:
    for address, account in stake_accounts.items():
        rewards = await solscan_fetch_rewards_for_account(client, address, page_size, logger)
        if not rewards:
            continue
        account.rewards = rewards
        account.total_rewards_lamports = sum(record.amount_lamports for record in rewards)
        account.first_reward_slot = next((record.effective_slot for record in rewards if record.effective_slot is not None), None)
        account.last_reward_slot = next(
            (record.effective_slot for record in reversed(rewards) if record.effective_slot is not None),
            None,
        )

        cached_times = [record.block_time for record in rewards if record.block_time is not None]
        if cached_times:
            account.first_reward_time = datetime.fromtimestamp(min(cached_times), tz=timezone.utc)
            account.last_reward_time = datetime.fromtimestamp(max(cached_times), tz=timezone.utc)
        elif account.first_reward_slot is not None and account.last_reward_slot is not None and slots_in_epoch and slots_in_epoch > 0:
            duration = compute_duration_days_from_epochs(
                rewards[0].epoch,
                rewards[-1].epoch,
                slots_in_epoch,
                average_slot_time_seconds,
            )
            account.duration_days = duration

        if account.first_reward_time and account.last_reward_time:
            delta = account.last_reward_time - account.first_reward_time
            if delta.total_seconds() > 0:
                account.duration_days = max(delta.total_seconds() / 86_400, 0.0)

        if account.delegated_lamports > 0 and account.total_rewards_lamports > 0:
            account.realized_yield_pct = (account.total_rewards_lamports / account.delegated_lamports) * 100
            if account.duration_days and account.duration_days > 0:
                account.annualized_yield_pct = account.realized_yield_pct * (365 / account.duration_days)


async def solscan_build_validator_lookup(
    client: SolscanClient,
    validator_votes: Set[str],
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for vote_account in validator_votes:
        decoded = await solscan_decode_account(client, vote_account, logger)
        if not decoded:
            continue
        data_section = decoded.get("data") if isinstance(decoded.get("data"), dict) else {}
        parsed = data_section.get("parsed") if isinstance(data_section.get("parsed"), dict) else decoded.get("parsed")
        info = parsed.get("info") if isinstance(parsed, dict) else {}
        if not isinstance(info, dict):
            continue
        commission = safe_int(info.get("commission"))
        epoch_credits = info.get("epochCredits") or info.get("epoch_credits") or []
        if isinstance(epoch_credits, list) and epoch_credits and isinstance(epoch_credits[0], dict):
            # Convert list of dicts to list of lists if needed.
            converted: List[List[int]] = []
            for entry in epoch_credits:
                epoch = safe_int(entry.get("epoch")) or 0
                credits = safe_int(entry.get("credits")) or 0
                prev_credits = safe_int(entry.get("prevCredits") or entry.get("previousCredits")) or 0
                converted.append([epoch, credits, prev_credits])
            epoch_credits = converted
        activated_stake = safe_int(info.get("activatedStake") or info.get("activated_stake"))
        lookup[vote_account] = {
            "votePubkey": vote_account,
            "commission": commission,
            "epochCredits": epoch_credits if isinstance(epoch_credits, list) else [],
            "activatedStake": activated_stake,
            "delinquent": info.get("delinquent", False),
        }
    return lookup


async def solscan_fetch_network_context(
    client: SolscanClient,
    logger: logging.Logger,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    try:
        epoch_payload = await client.get("/cluster/epoch-info")
        epoch_data = epoch_payload.get("data") if isinstance(epoch_payload, dict) else epoch_payload
        if isinstance(epoch_data, dict):
            context["current_epoch"] = safe_int(epoch_data.get("epoch"))
            context["absolute_slot"] = safe_int(epoch_data.get("absoluteSlot") or epoch_data.get("absolute_slot"))
            context["slots_in_epoch"] = safe_int(epoch_data.get("slotsInEpoch") or epoch_data.get("slots_in_epoch"))
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to fetch Solscan epoch info: %s", exc)

    try:
        performance_payload = await client.get("/cluster/performance")
        performance_data = performance_payload.get("data") if isinstance(performance_payload, dict) else performance_payload
        if isinstance(performance_data, list) and performance_data:
            context["average_slot_time_seconds"] = compute_average_slot_time_seconds(performance_data)
        elif isinstance(performance_data, dict):
            context["average_slot_time_seconds"] = compute_average_slot_time_seconds([performance_data])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to fetch Solscan performance samples: %s", exc)

    try:
        inflation_payload = await client.get("/cluster/inflation-rate")
        inflation_data = inflation_payload.get("data") if isinstance(inflation_payload, dict) else inflation_payload
        if isinstance(inflation_data, dict):
            context["network_inflation"] = inflation_data
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to fetch Solscan inflation rate: %s", exc)

    return context


async def run_with_solscan(
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    wallet_address = config["wallet_address"]
    page_size = safe_int(config.get("solscan_page_size")) or SOLSCAN_DEFAULT_PAGE_SIZE
    max_wallet_signatures = safe_int(config.get("max_wallet_signatures"))
    max_stake_signatures = safe_int(config.get("max_stake_signatures"))
    api_key = config.get("solscan_api_key")
    base_url = config.get("solscan_base_url", SOLSCAN_API_BASE)
    concurrency_limit = max(1, safe_int(config.get("concurrency_limit")) or 4)

    stake_accounts: Dict[str, StakeAccount] = {}
    wallet_transactions: List[Dict[str, Any]] = []

    transaction_cache: Dict[str, Any] = {}
    history_meta: Dict[str, Dict[str, Any]] = {}
    address_events: Dict[str, List[DelegationEvent]] = {}
    recorded_signatures: Dict[str, Set[str]] = {}

    async with SolscanClient(api_key, concurrency_limit, base_url=base_url, timeout=35.0) as client:
        network_context = await solscan_fetch_network_context(client, logger)
        average_slot_time_seconds = network_context.get("average_slot_time_seconds")
        if average_slot_time_seconds is None:
            average_slot_time_seconds = 0.4
            network_context["average_slot_time_seconds"] = average_slot_time_seconds

        slots_in_epoch = network_context.get("slots_in_epoch")

        current_entries = await solscan_fetch_current_stake_accounts(client, wallet_address, page_size, logger)
        all_addresses: Set[str] = set(current_entries.keys())

        wallet_transactions = await solscan_fetch_transactions_for_address(
            client,
            wallet_address,
            max_wallet_signatures,
            page_size,
            logger,
        )

        for summary in reversed(wallet_transactions):
            signature = summary.get("tx_hash") or summary.get("signature") or summary.get("txHash")
            if not signature:
                continue
            detail = await solscan_fetch_transaction_detail(client, signature, transaction_cache, logger)
            if not detail:
                continue
            programs = detail.get("programs_involved") or detail.get("programsInvolved") or []
            if programs and STAKE_PROGRAM_ID not in programs:
                continue
            instructions = detail.get("parsed_instructions") or detail.get("parsedInstructions") or []
            if isinstance(instructions, list):
                for instr in instructions:
                    if isinstance(instr, dict):
                        for addr in try_extract_stake_accounts_from_instruction(instr):
                            if addr not in all_addresses:
                                all_addresses.add(addr)
            events = extract_delegate_events(detail, None, signature)
            for event in events:
                stake_addr = event.stake_account
                if not stake_addr:
                    continue
                seen = recorded_signatures.setdefault(stake_addr, set())
                if signature in seen:
                    continue
                address_events.setdefault(stake_addr, []).append(event)
                seen.add(signature)
                _update_history_meta(history_meta, stake_addr, event.block_time, signature)

        pending = list(all_addresses)
        processed_accounts: Set[str] = set()
        while pending:
            stake_addr = pending.pop()
            if stake_addr in processed_accounts:
                continue
            processed_accounts.add(stake_addr)
            account_transactions = await solscan_fetch_transactions_for_address(
                client,
                stake_addr,
                max_stake_signatures,
                page_size,
                logger,
            )
            for summary in reversed(account_transactions):
                signature = summary.get("tx_hash") or summary.get("signature") or summary.get("txHash")
                if not signature:
                    continue
                detail = await solscan_fetch_transaction_detail(client, signature, transaction_cache, logger)
                if not detail:
                    continue
                instructions = detail.get("parsed_instructions") or detail.get("parsedInstructions") or []
                if isinstance(instructions, list):
                    for instr in instructions:
                        if isinstance(instr, dict):
                            for addr in try_extract_stake_accounts_from_instruction(instr):
                                if addr not in all_addresses:
                                    all_addresses.add(addr)
                                    pending.append(addr)
                events = extract_delegate_events(detail, stake_addr, signature)
                if not events:
                    continue
                seen = recorded_signatures.setdefault(stake_addr, set())
                if signature in seen:
                    continue
                address_events.setdefault(stake_addr, []).extend(events)
                seen.add(signature)
                for event in events:
                    _update_history_meta(history_meta, stake_addr, event.block_time, signature)

        for address in sorted(all_addresses):
            decoded = await solscan_decode_account(client, address, logger)
            stake_account = parse_stake_account_from_decoded(address, decoded, current_entries.get(address))
            events = address_events.get(address, [])
            events.sort(key=lambda event: (event.block_time or 0, event.slot or 0, event.signature))
            if events:
                stake_account.delegation_history = events
                validators = [event.vote_account for event in events if event.vote_account]
                if validators:
                    stake_account.historical_validators = list(dict.fromkeys(validators))
                    if not stake_account.validator_vote:
                        stake_account.validator_vote = validators[-1]
            meta = history_meta.get(address)
            if meta:
                stake_account.first_seen_unix = meta.get("first_seen")
                stake_account.last_seen_unix = meta.get("last_seen")
                stake_account.history_signature_count = len(meta.get("seen_in") or [])
            stake_accounts[address] = stake_account

        await solscan_populate_rewards(
            client,
            stake_accounts,
            page_size,
            average_slot_time_seconds,
            slots_in_epoch,
            logger,
        )

        validator_votes = {account.validator_vote for account in stake_accounts.values() if account.validator_vote}
        validator_lookup = await solscan_build_validator_lookup(client, validator_votes, logger)
        inflation_rate = network_context.get("network_inflation") or {}

        validator_rows, stake_rows = analyze_validators(
            stake_accounts,
            validator_lookup,
            inflation_rate,
            slots_in_epoch or 0,
        )

    pipeline_context = {
        "solscan_base_url": base_url,
        "solscan_page_size": page_size,
        "solscan_wallet_transaction_count": len(wallet_transactions),
        "stake_account_count": len(stake_accounts),
        "validator_count": len(validator_rows),
        "solscan_history_tracked_accounts": len(history_meta),
        "solscan_historical_only_count": sum(
            1 for account in stake_accounts.values() if account.stake_type == "historical_only"
        ),
    }
    for key in ("current_epoch", "absolute_slot", "slots_in_epoch", "average_slot_time_seconds", "network_inflation"):
        if key in network_context:
            pipeline_context[key] = network_context[key]

    return validator_rows, stake_rows, pipeline_context


async def fetch_delegation_history(
    client: SolanaRPCClient,
    stake_accounts: Dict[str, StakeAccount],
    wallet_address: str,
    stake_max_signatures: Optional[int],
    wallet_max_signatures: Optional[int],
    transaction_cache: Dict[str, Any],
    transaction_backoff_seconds: float,
    logger: logging.Logger,
) -> None:
    def _resolve_limit(limit: Optional[int]) -> Optional[int]:
        if limit is None:
            return None
        if limit <= 0:
            return None
        return limit

    stake_limit = _resolve_limit(stake_max_signatures)
    wallet_limit = _resolve_limit(wallet_max_signatures)

    async def _fetch_signatures_for_address(address: str, limit_cap: Optional[int], description: str) -> List[Dict[str, Any]]:
        signatures: List[Dict[str, Any]] = []
        before: Optional[str] = None
        while limit_cap is None or len(signatures) < limit_cap:
            remaining = limit_cap - len(signatures) if limit_cap is not None else 1000
            limit = min(1000, remaining) if limit_cap is not None else 1000
            if limit <= 0:
                break
            params: List[Any] = [address, {"limit": limit}]
            if before:
                params[1]["before"] = before
            batch = await client.request("getSignaturesForAddress", params)
            if not batch:
                break
            signatures.extend(batch)
            if len(batch) < limit:
                break
            before = batch[-1].get("signature")

        if limit_cap is not None and len(signatures) >= limit_cap and before:
            logger.warning(
                "Signature fetch for %s capped at %d records; older history may exist. Increase limits to include more.",
                description,
                limit_cap,
            )

        return signatures

    async def _load_transaction(signature: str) -> Optional[Dict[str, Any]]:
        tx = transaction_cache.get(signature)
        if tx is None:
            tx = await client.request(
                "getTransaction",
                [
                    signature,
                    {
                        "encoding": "jsonParsed",
                        "maxSupportedTransactionVersion": 0,
                    },
                ],
            )
            if tx:
                transaction_cache[signature] = tx
            await asyncio.sleep(max(transaction_backoff_seconds, 0.0))
        return tx

    recorded_signatures: Dict[str, Set[str]] = {pubkey: set() for pubkey in stake_accounts}

    async def _ingest_account_history(
        pubkey: str,
        account: StakeAccount,
        limit_cap: Optional[int],
        reset_history: bool,
    ) -> None:
        signature_batch = await _fetch_signatures_for_address(pubkey, limit_cap, f"stake account {pubkey}")
        logger.info("Fetched %d signatures for stake account %s", len(signature_batch), pubkey)

        if reset_history:
            account.delegation_history.clear()
            recorded_signatures[pubkey] = set()

        seen_signatures = recorded_signatures.setdefault(pubkey, set())

        for sig_info in reversed(signature_batch):
            signature = sig_info.get("signature")
            if not signature or signature in seen_signatures:
                continue
            tx = await _load_transaction(signature)
            if not tx:
                continue
            events = extract_delegate_events(tx, pubkey, signature)
            if not events:
                continue
            account.delegation_history.extend(events)
            seen_signatures.add(signature)

    for pubkey, account in stake_accounts.items():
        logger.info("Retrieving delegation history for stake account %s", pubkey)
        await _ingest_account_history(pubkey, account, stake_limit, reset_history=True)

    initial_account_count = len(stake_accounts)
    logger.info("Scanning root wallet %s for historical delegation events", wallet_address)
    wallet_signatures = await _fetch_signatures_for_address(wallet_address, wallet_limit, f"wallet {wallet_address}")
    logger.info("Fetched %d signatures for wallet %s", len(wallet_signatures), wallet_address)

    newly_discovered_accounts: Set[str] = set()

    for sig_info in reversed(wallet_signatures):
        signature = sig_info.get("signature")
        if not signature:
            continue
        tx = await _load_transaction(signature)
        if not tx:
            continue
        events = extract_delegate_events(tx, None, signature)
        if not events:
            continue
        for event in events:
            stake_pubkey = event.stake_account
            if not stake_pubkey:
                continue
            account = stake_accounts.get(stake_pubkey)
            if account is None:
                account = StakeAccount(
                    pubkey=stake_pubkey,
                    delegated_lamports=0,
                    validator_vote=event.vote_account,
                    activation_epoch=None,
                    deactivation_epoch=None,
                    authorized_staker=None,
                    authorized_withdrawer=None,
                    rent_exempt_reserve_lamports=None,
                    stake_type="historical_only",
                    account_balance_lamports=None,
                    lockup={},
                )
                stake_accounts[stake_pubkey] = account
                recorded_signatures[stake_pubkey] = set()
                newly_discovered_accounts.add(stake_pubkey)
            seen_signatures = recorded_signatures.setdefault(stake_pubkey, set())
            if signature in seen_signatures:
                continue
            account.delegation_history.append(event)
            seen_signatures.add(signature)
            if not account.validator_vote:
                account.validator_vote = event.vote_account

    for pubkey in newly_discovered_accounts:
        account = stake_accounts[pubkey]
        logger.info("Retrieving delegation history for newly discovered stake account %s", pubkey)
        await _ingest_account_history(pubkey, account, stake_limit, reset_history=False)

    newly_discovered = len(stake_accounts) - initial_account_count
    if newly_discovered > 0:
        logger.info(
            "Discovered %d additional historical stake accounts via wallet transaction history",
            newly_discovered,
        )

    for account in stake_accounts.values():
        account.delegation_history.sort(
            key=lambda event: (
                event.block_time or 0,
                event.slot or 0,
                event.signature,
            )
        )
        if account.validator_vote and all(
            event.vote_account != account.validator_vote for event in account.delegation_history if event.vote_account
        ):
            account.delegation_history.append(
                DelegationEvent(
                    signature="CURRENT_STATE",
                    slot=None,
                    block_time=None,
                    vote_account=account.validator_vote,
                    instruction_type="current_delegation",
                    stake_account=account.pubkey,
                )
            )
        unique_validators = list(
            dict.fromkeys(
                event.vote_account for event in account.delegation_history if event.vote_account
            )
        )
        account.historical_validators = unique_validators


async def fetch_inflation_rewards(
    client: SolanaRPCClient,
    stake_accounts: Dict[str, StakeAccount],
    start_epoch: int,
    end_epoch: int,
    logger: logging.Logger,
) -> None:
    if start_epoch >= end_epoch:
        logger.warning("Activation epoch (%d) is not less than current epoch (%d); skipping reward fetch.", start_epoch, end_epoch)
        return

    pubkeys = list(stake_accounts.keys())
    logger.info("Fetching inflation rewards from epoch %d to %d (%d stake accounts)", start_epoch, end_epoch - 1, len(pubkeys))

    for epoch in range(start_epoch, end_epoch):
        rewards = await client.request("getInflationReward", [pubkeys, {"epoch": epoch}])
        if rewards is None:
            continue
        for pubkey, reward_entry in zip(pubkeys, rewards):
            if not reward_entry:
                continue
            record = RewardRecord(
                epoch=epoch,
                amount_lamports=safe_int(reward_entry.get("amount")) or 0,
                post_balance_lamports=safe_int(reward_entry.get("postBalance")) or 0,
                effective_slot=safe_int(reward_entry.get("effectiveSlot")),
                commission=safe_int(reward_entry.get("commission")),
            )
            account = stake_accounts[pubkey]
            account.rewards.append(record)
            account.total_rewards_lamports += record.amount_lamports

        if epoch % 10 == 0:
            logger.debug("Processed rewards through epoch %d", epoch)


async def get_block_time(
    client: SolanaRPCClient,
    slot: int,
    cache: Dict[int, Optional[int]],
) -> Optional[int]:
    if slot in cache:
        return cache[slot]
    result = await client.request("getBlockTime", [slot])
    cache[slot] = safe_int(result)
    return cache[slot]


def compute_duration_days_from_epochs(
    start_epoch: Optional[int],
    end_epoch: Optional[int],
    slots_per_epoch: int,
    average_slot_time_seconds: float,
) -> Optional[float]:
    if start_epoch is None or end_epoch is None or end_epoch < start_epoch:
        return None
    epoch_span = (end_epoch - start_epoch) + 1
    seconds = epoch_span * slots_per_epoch * average_slot_time_seconds
    return seconds / 86_400


async def enrich_stake_account_metrics(
    client: SolanaRPCClient,
    stake_accounts: Dict[str, StakeAccount],
    slots_per_epoch: int,
    average_slot_time_seconds: float,
    logger: logging.Logger,
) -> None:
    cache: Dict[int, Optional[int]] = {}

    for account in stake_accounts.values():
        if account.rewards:
            account.first_reward_slot = account.rewards[0].effective_slot
            account.last_reward_slot = account.rewards[-1].effective_slot

            if account.first_reward_slot is not None:
                first_time = await get_block_time(client, account.first_reward_slot, cache)
                if first_time is not None:
                    account.first_reward_time = datetime.fromtimestamp(first_time, tz=timezone.utc)

            if account.last_reward_slot is not None:
                last_time = await get_block_time(client, account.last_reward_slot, cache)
                if last_time is not None:
                    account.last_reward_time = datetime.fromtimestamp(last_time, tz=timezone.utc)

            if account.first_reward_time and account.last_reward_time:
                delta = account.last_reward_time - account.first_reward_time
                account.duration_days = max(delta.total_seconds() / 86_400, 0.0)
            else:
                account.duration_days = compute_duration_days_from_epochs(
                    account.rewards[0].epoch,
                    account.rewards[-1].epoch,
                    slots_per_epoch,
                    average_slot_time_seconds,
                )
        else:
            logger.debug("Stake account %s has no rewards yet", account.pubkey)

        if account.delegated_lamports > 0 and account.total_rewards_lamports > 0:
            account.realized_yield_pct = (account.total_rewards_lamports / account.delegated_lamports) * 100
            if account.duration_days and account.duration_days > 0:
                account.annualized_yield_pct = account.realized_yield_pct * (365 / account.duration_days)


def build_validator_lookup(vote_accounts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for category, delinquent_flag in (("current", False), ("delinquent", True)):
        for entry in vote_accounts.get(category, []) or []:
            entry_copy = dict(entry)
            entry_copy["delinquent"] = delinquent_flag
            lookup[entry["votePubkey"]] = entry_copy
    return lookup


def compute_validator_expected_yield(
    commission: Optional[int],
    network_validator_rate: float,
) -> Optional[float]:
    if commission is None:
        return network_validator_rate * 100
    return network_validator_rate * (1 - commission / 100) * 100


def compute_validator_uptime(entry: Dict[str, Any], slots_in_epoch: int) -> Optional[float]:
    credits = entry.get("epochCredits") or []
    if not credits:
        return None
    latest = credits[-1]
    if len(latest) < 3:
        return None
    epoch_credits = safe_int(latest[1])
    prev_credits = safe_int(latest[2])
    if epoch_credits is None or prev_credits is None:
        return None
    earned = max(epoch_credits - prev_credits, 0)
    if slots_in_epoch <= 0:
        return None
    uptime = earned / slots_in_epoch
    return min(max(uptime, 0.0), 1.0)


def analyze_validators(
    stake_accounts: Dict[str, StakeAccount],
    validator_lookup: Dict[str, Dict[str, Any]],
    inflation_rate: Dict[str, Any],
    slots_in_epoch: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    network_validator_rate = float(inflation_rate.get("validator", 0.0))

    validator_accounts: Dict[str, List[StakeAccount]] = {}
    historical_validator_events: Dict[str, List[DelegationEvent]] = {}
    for account in stake_accounts.values():
        for event in account.delegation_history:
            historical_validator_events.setdefault(event.vote_account, []).append(event)
        if not account.validator_vote:
            continue
        validator_accounts.setdefault(account.validator_vote, []).append(account)

    validator_rows: List[Dict[str, Any]] = []
    stake_rows: List[Dict[str, Any]] = []

    for account in stake_accounts.values():
        validator_info = validator_lookup.get(account.validator_vote or "")
        commission = safe_int((validator_info or {}).get("commission"))
        uptime = compute_validator_uptime(validator_info or {}, slots_in_epoch) if validator_info else None
        account.validator_commission = commission
        account.validator_uptime_ratio = uptime
        account.validator_delinquent = (validator_info or {}).get("delinquent")
        account.expected_yield_pct = compute_validator_expected_yield(commission, network_validator_rate)

        stake_rows.append(
            {
                "stake_account": account.pubkey,
                "validator_vote": account.validator_vote,
                "delegated_sol": account.delegated_sol,
                "total_rewards_sol": account.total_rewards_sol,
                "realized_yield_pct": account.realized_yield_pct,
                "annualized_yield_pct": account.annualized_yield_pct,
                "expected_yield_pct": account.expected_yield_pct,
                "activation_epoch": account.activation_epoch,
                "deactivation_epoch": account.deactivation_epoch,
                "duration_days": account.duration_days,
                "first_reward_time": account.first_reward_time.isoformat() if account.first_reward_time else None,
                "last_reward_time": account.last_reward_time.isoformat() if account.last_reward_time else None,
                "commission": commission,
                "uptime_ratio": uptime,
                "delinquent": account.validator_delinquent,
                "historical_validators": account.historical_validators,
                "delegation_history": [event.to_dict() for event in account.delegation_history],
                "delegation_event_count": len(account.delegation_history),
                "first_seen_unix": account.first_seen_unix,
                "last_seen_unix": account.last_seen_unix,
                "history_signature_count": account.history_signature_count,
            }
        )

    for validator_vote, accounts in validator_accounts.items():
        validator_info = validator_lookup.get(validator_vote, {})
        commission = safe_int(validator_info.get("commission"))
        uptime = compute_validator_uptime(validator_info, slots_in_epoch)
        activated_stake = lamports_to_sol(safe_int(validator_info.get("activatedStake")))
        delinquent = validator_info.get("delinquent", False)

        total_delegated = sum(acc.delegated_lamports for acc in accounts)
        total_rewards = sum(acc.total_rewards_lamports for acc in accounts)
        realized_pct = (total_rewards / total_delegated) * 100 if total_delegated else None

        annualized_values = [
            (acc.annualized_yield_pct, acc.delegated_lamports)
            for acc in accounts
            if acc.annualized_yield_pct is not None
        ]
        annualized_weighted = None
        if annualized_values:
            numerator = sum(value * weight for value, weight in annualized_values)
            denominator = sum(weight for _, weight in annualized_values)
            if denominator:
                annualized_weighted = numerator / denominator

        expected_pct = compute_validator_expected_yield(commission, network_validator_rate)

        status: str
        if annualized_weighted is None:
            status = "INSUFFICIENT_DATA"
        elif expected_pct is None:
            status = "UNKNOWN"
        elif annualized_weighted < expected_pct * 0.9:
            status = "UNDERPERFORMING"
        else:
            status = "HEALTHY"

        validator_rows.append(
            {
                "validator_vote": validator_vote,
                "delegated_sol": lamports_to_sol(total_delegated),
                "total_rewards_sol": lamports_to_sol(total_rewards),
                "realized_yield_pct": realized_pct,
                "annualized_yield_pct": annualized_weighted,
                "expected_yield_pct": expected_pct,
                "commission": commission,
                "uptime_ratio": uptime,
                "delinquent": delinquent,
                "wallet_stake_accounts": [acc.pubkey for acc in accounts],
                "activated_stake_sol": activated_stake,
                "status": status,
                "delegation_event_count": len(historical_validator_events.get(validator_vote, [])),
                "first_delegation_time": _get_first_event_iso(historical_validator_events.get(validator_vote)),
                "last_delegation_time": _get_last_event_iso(historical_validator_events.get(validator_vote)),
            }
        )

    historical_only_validators = set(historical_validator_events.keys()) - set(validator_accounts.keys())
    for validator_vote in historical_only_validators:
        events = historical_validator_events[validator_vote]
        validator_info = validator_lookup.get(validator_vote, {})
        commission = safe_int(validator_info.get("commission"))
        uptime = compute_validator_uptime(validator_info, slots_in_epoch) if validator_info else None
        activated_stake = lamports_to_sol(safe_int(validator_info.get("activatedStake"))) if validator_info else None
        validator_rows.append(
            {
                "validator_vote": validator_vote,
                "delegated_sol": 0.0,
                "total_rewards_sol": 0.0,
                "realized_yield_pct": None,
                "annualized_yield_pct": None,
                "expected_yield_pct": compute_validator_expected_yield(commission, network_validator_rate),
                "commission": commission,
                "uptime_ratio": uptime,
                "delinquent": validator_info.get("delinquent") if validator_info else None,
                "wallet_stake_accounts": list({event.stake_account for event in events if event.stake_account}),
                "activated_stake_sol": activated_stake,
                "status": "HISTORICAL_ONLY",
                "delegation_event_count": len(events),
                "first_delegation_time": _get_first_event_iso(events),
                "last_delegation_time": _get_last_event_iso(events),
            }
        )

    return validator_rows, stake_rows


def render_console_output(validator_rows: List[Dict[str, Any]], stake_rows: List[Dict[str, Any]], logger: logging.Logger) -> None:
    if not validator_rows:
        logger.warning("No validator data available to display")
        return

    validator_df = pd.DataFrame(validator_rows)
    display_columns = [
        "validator_vote",
        "delegated_sol",
        "total_rewards_sol",
        "annualized_yield_pct",
        "expected_yield_pct",
        "commission",
        "uptime_ratio",
        "status",
    ]
    validator_df = validator_df[display_columns].sort_values(by="annualized_yield_pct", ascending=False, na_position="last")
    logger.info("Validator Summary:\n%s", validator_df.to_string(index=False, justify="center", float_format=lambda x: f"{x:.4f}"))

    stake_df = pd.DataFrame(stake_rows)
    if stake_df.empty:
        logger.warning("No stake account data available to display")
        return
    stake_display_cols = [
        "stake_account",
        "validator_vote",
        "delegated_sol",
        "total_rewards_sol",
        "annualized_yield_pct",
        "expected_yield_pct",
        "duration_days",
    ]
    logger.info("Stake Account Detail:\n%s", stake_df[stake_display_cols].to_string(index=False, justify="center", float_format=lambda x: f"{x:.4f}"))


def write_json_output(
    output_path: Path,
    context: Dict[str, Any],
    validator_rows: List[Dict[str, Any]],
    stake_rows: List[Dict[str, Any]],
    logger: logging.Logger,
) -> None:
    payload = {
        "metadata": context,
        "validators": validator_rows,
        "stake_accounts": stake_rows,
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote JSON report to %s", output_path)


def write_csv_outputs(
    validator_path: Path,
    stake_path: Path,
    validator_rows: List[Dict[str, Any]],
    stake_rows: List[Dict[str, Any]],
    logger: logging.Logger,
) -> None:
    validator_df = pd.DataFrame(validator_rows)
    if "wallet_stake_accounts" in validator_df:
        validator_df["wallet_stake_accounts"] = validator_df["wallet_stake_accounts"].apply(lambda value: ";".join(value) if isinstance(value, list) else value)
    validator_df.to_csv(validator_path, index=False)

    stake_df = pd.DataFrame(stake_rows)
    if "historical_validators" in stake_df:
        stake_df["historical_validators"] = stake_df["historical_validators"].apply(lambda value: ";".join(value) if isinstance(value, list) else value)
    if "delegation_history" in stake_df:
        stake_df["delegation_history"] = stake_df["delegation_history"].apply(lambda value: json.dumps(value) if isinstance(value, list) else value)
    stake_df.to_csv(stake_path, index=False)
    logger.info("Wrote CSV report to %s", validator_path)
    logger.info("Wrote stake account CSV to %s", stake_path)


def compute_average_slot_time_seconds(performance_samples: Optional[List[Dict[str, Any]]]) -> float:
    if not performance_samples:
        return 0.4  # fallback ~400ms per slot
    sample = performance_samples[0]
    num_slots = safe_int(sample.get("numSlots")) or 0
    sample_period = safe_int(sample.get("samplePeriodSecs")) or 0
    if num_slots <= 0 or sample_period <= 0:
        return 0.4
    return sample_period / num_slots


def _get_first_event_iso(events: Optional[List[DelegationEvent]]) -> Optional[str]:
    if not events:
        return None
    ordered = sorted(events, key=lambda event: (event.block_time or 0, event.slot or 0))
    for event in ordered:
        if event.block_time:
            return datetime.fromtimestamp(event.block_time, tz=timezone.utc).isoformat()
    return None


def _get_last_event_iso(events: Optional[List[DelegationEvent]]) -> Optional[str]:
    if not events:
        return None
    ordered = sorted(events, key=lambda event: (event.block_time or 0, event.slot or 0), reverse=True)
    for event in ordered:
        if event.block_time:
            return datetime.fromtimestamp(event.block_time, tz=timezone.utc).isoformat()
    return None


async def run_with_rpc(
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    wallet_address = config["wallet_address"]
    transaction_backoff_seconds = float(config.get("transaction_backoff_seconds", 0.35) or 0.35)
    transaction_cache: Dict[str, Any] = {}

    async with SolanaRPCClient(config["rpc_endpoint_groups"], config["concurrency_limit"]) as client:
        epoch_info = await client.request("getEpochInfo")
        if not epoch_info:
            raise RPCError("Failed to retrieve epoch info")

        slots_in_epoch = safe_int(epoch_info.get("slotsInEpoch")) or 0
        current_epoch = safe_int(epoch_info.get("epoch")) or 0
        absolute_slot = safe_int(epoch_info.get("absoluteSlot"))

        epoch_schedule = await client.request("getEpochSchedule")
        if slots_in_epoch <= 0:
            slots_in_epoch = safe_int(epoch_schedule.get("slotsPerEpoch")) or 0

        performance_samples = await client.request("getRecentPerformanceSamples", [1])
        average_slot_time_seconds = compute_average_slot_time_seconds(performance_samples)

        inflation_rate = await client.request("getInflationRate")

        stake_accounts = await fetch_stake_accounts(client, wallet_address, logger)
        if stake_accounts:
            activation_epochs = [acc.activation_epoch for acc in stake_accounts.values() if acc.activation_epoch is not None]
            start_epoch = min(activation_epochs) if activation_epochs else current_epoch
            await fetch_inflation_rewards(client, stake_accounts, start_epoch, current_epoch, logger)
            await enrich_stake_account_metrics(client, stake_accounts, slots_in_epoch, average_slot_time_seconds, logger)
        else:
            logger.warning(
                "No active stake accounts detected via stake program; continuing with historical wallet scan."
            )

        await fetch_delegation_history(
            client,
            stake_accounts,
            wallet_address,
            safe_int(config.get("max_stake_signatures")),
            safe_int(config.get("max_wallet_signatures")),
            transaction_cache,
            transaction_backoff_seconds,
            logger,
        )

        if not stake_accounts:
            logger.warning("No stake activity detected for wallet %s", wallet_address)

        vote_accounts = await client.request("getVoteAccounts") or {}
        validator_lookup = build_validator_lookup(vote_accounts)

        validator_rows, stake_rows = analyze_validators(
            stake_accounts,
            validator_lookup,
            inflation_rate or {},
            slots_in_epoch,
        )

    pipeline_context = {
        "rpc_endpoint": config.get("rpc_endpoint"),
        "rpc_endpoints": config.get("rpc_endpoints"),
        "rpc_endpoint_groups": config.get("rpc_endpoint_groups"),
        "current_epoch": current_epoch,
        "absolute_slot": absolute_slot,
        "slots_in_epoch": slots_in_epoch,
        "average_slot_time_seconds": average_slot_time_seconds,
        "network_inflation": inflation_rate,
        "stake_account_count": len(stake_accounts),
        "validator_count": len(validator_rows),
    }

    return validator_rows, stake_rows, pipeline_context


async def run() -> None:
    base_dir = Path(__file__).resolve().parent
    config = load_config(base_dir / CONFIG_FILENAME, base_dir / "wallet_config.json")

    logging.basicConfig(
        level=getattr(logging, str(config.get("log_level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("solana_staking_analyzer")

    wallet_address = config["wallet_address"]
    logger.info("Starting staking analysis for wallet %s", wallet_address)

    data_source = config.get("data_source", "solscan")
    if data_source == "solscan":
        validator_rows, stake_rows, pipeline_context = await run_with_solscan(config, logger)
    else:
        validator_rows, stake_rows, pipeline_context = await run_with_rpc(config, logger)

    timestamp = datetime.now(tz=timezone.utc).isoformat()
    context: Dict[str, Any] = {
        "wallet_address": wallet_address,
        "generated_at": timestamp,
        "data_source": data_source,
    }
    for key, value in pipeline_context.items():
        if value is not None:
            context[key] = value

    formats = [fmt.lower() for fmt in config.get("output_format", [])]
    if "console" in formats:
        render_console_output(validator_rows, stake_rows, logger)

    if "json" in formats:
        json_path = base_dir / config["json_output_file"]
        write_json_output(json_path, context, validator_rows, stake_rows, logger)

    if "csv" in formats:
        validator_csv_path = base_dir / config["csv_output_file"]
        stake_csv_path = base_dir / config["stake_accounts_csv_output_file"]
        write_csv_outputs(validator_csv_path, stake_csv_path, validator_rows, stake_rows, logger)


def main() -> None:
    try:
        asyncio.run(run())
    except (ConfigurationError, RPCError) as exc:
        logging.getLogger("solana_staking_analyzer").error("Fatal error: %s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logging.getLogger("solana_staking_analyzer").warning("Execution interrupted by user")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

