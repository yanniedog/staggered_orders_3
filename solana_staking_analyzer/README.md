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
- `config.cfg` → `rpc_endpoint`: RPC URL (defaults to `https://api.mainnet-beta.solana.com`).
- `concurrency_limit`: Maximum number of concurrent RPC calls (tune based on provider limits).
- `output_format`: Any combination of `console`, `json`, `csv`.
- `json_output_file`, `csv_output_file`: Filenames for saved reports relative to the project directory.
- `log_level`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

## Usage

```bash
python solana_staking_analyzer.py
```

The script prints a validator summary table, stake account details, and paths to generated output files. JSON and CSV exports include both validator-level and stake-account-level metrics for downstream analysis.

## Error Handling

The script uses structured logging, retries transient RPC errors automatically, and exits with a non-zero status for unrecoverable issues. Review the logs for warnings or errors, adjust configuration, and rerun as needed.

## Data Sources

- Solana JSON-RPC (`getProgramAccounts`, `getInflationReward`, `getVoteAccounts`, `getEpochInfo`, `getInflationRate`, `getEpochSchedule`, `getRecentPerformanceSamples`, `getBlockTime`).
- No third-party analytics APIs are required, ensuring all on-chain data comes directly from the configured endpoint.

## License

This project is provided without a license by default. Add one if distribution or collaboration is required.

