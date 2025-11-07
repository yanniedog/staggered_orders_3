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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from solana.publickey import PublicKey


LAMPORTS_PER_SOL = 1_000_000_000
STAKE_PROGRAM_ID = "Stake11111111111111111111111111111111111111"
MAX_RPC_RETRIES = 5
RETRY_BACKOFF_SECONDS = 0.5


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

    @property
    def delegated_sol(self) -> float:
        return lamports_to_sol(self.delegated_lamports)

    @property
    def total_rewards_sol(self) -> float:
        return lamports_to_sol(self.total_rewards_lamports)


class SolanaRPCClient:
    def __init__(self, endpoint: str, concurrency_limit: int, timeout: float = 35.0) -> None:
        self.endpoint = endpoint
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        self._request_id = 0
        self.logger = logging.getLogger(self.__class__.__name__)

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

        attempt = 0
        while True:
            attempt += 1
            async with self._semaphore:
                try:
                    response = await self._client.post(self.endpoint, json=payload)
                    response.raise_for_status()
                    data = response.json()
                except httpx.HTTPStatusError as exc:
                    self.logger.warning("HTTP error on %s attempt %d: %s", method, attempt, exc)
                    if attempt >= MAX_RPC_RETRIES:
                        raise RPCError(f"HTTP error on method {method}: {exc}") from exc
                except httpx.RequestError as exc:
                    self.logger.warning("Request error on %s attempt %d: %s", method, attempt, exc)
                    if attempt >= MAX_RPC_RETRIES:
                        raise RPCError(f"Request error on method {method}: {exc}") from exc
                else:
                    if "error" in data:
                        message = data["error"].get("message", "Unknown RPC error")
                        self.logger.warning("RPC error on %s attempt %d: %s", method, attempt, message)
                        if attempt >= MAX_RPC_RETRIES:
                            raise RPCError(f"RPC error on method {method}: {message}")
                    else:
                        return data.get("result")

            await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        try:
            config = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON in configuration file: {exc}") from exc

    wallet_address = config.get("wallet_address")
    if not wallet_address:
        raise ConfigurationError("Configuration must include 'wallet_address'.")
    try:
        PublicKey(wallet_address)
    except Exception as exc:  # pylint: disable=broad-except
        raise ConfigurationError(f"Invalid wallet address: {wallet_address}") from exc

    rpc_endpoint = config.get("rpc_endpoint") or "https://api.mainnet-beta.solana.com"
    config["rpc_endpoint"] = rpc_endpoint

    concurrency_limit = safe_int(config.get("concurrency_limit")) or 8
    config["concurrency_limit"] = max(1, concurrency_limit)

    config.setdefault("output_format", ["console", "json", "csv"])
    config.setdefault("json_output_file", "validator_analysis.json")
    config.setdefault("csv_output_file", "validator_analysis.csv")
    config.setdefault("stake_accounts_csv_output_file", "stake_account_analysis.csv")
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
        validator_vote = delegation.get("voteAccount")
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
    for account in stake_accounts.values():
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
    pd.DataFrame(validator_rows).to_csv(validator_path, index=False)
    pd.DataFrame(stake_rows).to_csv(stake_path, index=False)
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


async def run() -> None:
    base_dir = Path(__file__).resolve().parent
    config = load_config(base_dir / "config.json")

    logging.basicConfig(
        level=getattr(logging, str(config.get("log_level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("solana_staking_analyzer")

    wallet_address = config["wallet_address"]
    logger.info("Starting staking analysis for wallet %s", wallet_address)

    async with SolanaRPCClient(config["rpc_endpoint"], config["concurrency_limit"]) as client:
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
        if not stake_accounts:
            logger.warning("No stake accounts detected. Nothing to analyze.")
            return

        activation_epochs = [acc.activation_epoch for acc in stake_accounts.values() if acc.activation_epoch is not None]
        start_epoch = min(activation_epochs) if activation_epochs else current_epoch
        await fetch_inflation_rewards(client, stake_accounts, start_epoch, current_epoch, logger)

        await enrich_stake_account_metrics(client, stake_accounts, slots_in_epoch, average_slot_time_seconds, logger)

        vote_accounts = await client.request("getVoteAccounts") or {}
        validator_lookup = build_validator_lookup(vote_accounts)

        validator_rows, stake_rows = analyze_validators(stake_accounts, validator_lookup, inflation_rate or {}, slots_in_epoch)

        timestamp = datetime.now(tz=timezone.utc).isoformat()
        context = {
            "wallet_address": wallet_address,
            "generated_at": timestamp,
            "rpc_endpoint": config["rpc_endpoint"],
            "current_epoch": current_epoch,
            "absolute_slot": absolute_slot,
            "slots_in_epoch": slots_in_epoch,
            "average_slot_time_seconds": average_slot_time_seconds,
            "network_inflation": inflation_rate,
            "stake_account_count": len(stake_accounts),
            "validator_count": len(validator_rows),
        }

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

