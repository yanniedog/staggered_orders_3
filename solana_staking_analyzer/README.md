# Solana Staking Performance Analyzer

This utility inspects all native Solana stake accounts that a wallet controls, summarizes validator performance, and highlights underperforming validators relative to expected staking yield.

## Features

- Discovers stake accounts where the configured wallet is the authorized staker or withdrawer.
- Aggregates delegation information (validator, stake size, activation/deactivation epochs).
- Retrieves historical inflation rewards for every stake account since activation.
- Calculates realized yield (total and annualized) per stake account and per validator.
- Evaluates validator health using current commission, activated stake, delinquency, uptime, and vote credit trends.
- Compares realized yield against an expected benchmark derived from network inflation and validator commission to flag underperformers.
- Outputs a structured report to the console, JSON, and CSV files.

## Prerequisites

- Python 3.10 or later.
- Access to a Solana RPC endpoint with `getProgramAccounts`, `getInflationReward`, and other standard methods enabled. The public mainnet-beta endpoint works for light usage, but a dedicated provider is recommended for heavy analysis.

## Installation

```bash
cd solana_staking_analyzer
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Edit `config.cfg` and `wallet_config.json` before running the script.

- `wallet_config.json` → `wallet_address`: Your Solana wallet public key in base58 form.
- `config.cfg` → `rpc_endpoint`/`rpc_endpoints`: Primary RPC URL or ordered list of preferred providers. When a list is supplied it is treated as the first priority group.
- `config.cfg` → `rpc_fallback_endpoints`: Optional secondary group of providers that will be attempted after the primary list is exhausted.
- `config.cfg` → `max_stake_signatures`: Maximum historical signatures fetched per stake account (increase if older delegations are missing).
- `config.cfg` → `max_wallet_signatures`: Maximum historical signatures fetched when scanning the root wallet (increase to capture very old stake activity).
- `concurrency_limit`: Maximum number of concurrent RPC calls (tune based on provider limits).
- `output_format`: Any combination of `console`, `json`, `csv`.
- `json_output_file`, `csv_output_file`: Filenames for saved reports relative to the project directory.
- `log_level`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

## Usage

```bash
python solana_staking_analyzer.py
```

The script prints a validator summary table, stake account details, and paths to generated output files. JSON and CSV exports include both validator-level and stake-account-level metrics for downstream analysis.

## Error Handling & Fallbacks

The script uses structured logging, retries transient RPC errors automatically, and exits with a non-zero status for unrecoverable issues. It maintains multiple tiers of RPC endpoints:

- User-specified primary endpoints (`rpc_endpoints`).
- Optional user-specified fallback endpoints (`rpc_fallback_endpoints`).
- Built-in public fallbacks (Solana Foundation, GenesysGo, Project Serum, Ankr, and solana.public-rpc.com).

If an endpoint throttles, errors, or becomes unhealthy, the client rotates within the current tier before promoting to the next tier; exhausted tiers are removed from service for the remainder of the run. Review the logs for warnings or errors, adjust configuration, and rerun as needed.

## Data Sources

- Solana JSON-RPC (`getProgramAccounts`, `getInflationReward`, `getVoteAccounts`, `getEpochInfo`, `getInflationRate`, `getEpochSchedule`, `getRecentPerformanceSamples`, `getBlockTime`).
- No third-party analytics APIs are required, ensuring all on-chain data comes directly from the configured endpoint.

## License

This project is provided without a license by default. Add one if distribution or collaboration is required.

