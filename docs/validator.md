# Validator Guide

This guide now recommends Docker-first validator operation.

## Docker-First Quickstart

```bash
cd docker
cp validator.env.example validator.env
cp compose.env.example compose.env
# edit both files, then:
chmod 600 validator.env compose.env
docker compose --env-file compose.env -f docker-compose.validator.yml up -d
```

Useful commands:

```bash
docker logs -f nexis-validator
docker logs -f nexis-watchtower
docker inspect --format '{{index .RepoDigests 0}}' nexis-validator
```

### Validator-Only vs Owner-Validator

- **Validator-only**:
  - validates miner submissions
  - submits chain weights
  - reads shared `record_info` overlap index
- **Owner-validator** (hotkey matches `NEXIS_OWNER_VALIDATOR_HOTKEY`):
  - publishes accepted metadata bundles (`dataset.parquet`, `manifest.json`) to record-info bucket
  - runs optional independent worker (`nexis sync-owner-datasets`) to copy full assets to owner dataset bucket
  - writes updated overlap index snapshot

## Environment Matrix

### Required

- `NEXIS_NETUID`
- `BT_NETWORK`
- `BT_WALLET_NAME`
- `BT_WALLET_HOTKEY`
- `BT_WALLET_PATH`
- `R2_ACCOUNT_ID`
- `R2_REGION`
- `NEXIS_BLOCK_POLL_SEC`
- `NEXIS_WORKDIR`
- `NEXIS_RECORD_INFO_BUCKET`
- `NEXIS_RECORD_INFO_ACCOUNT_ID`
- `NEXIS_RECORD_INFO_READ_ACCESS_KEY`
- `NEXIS_RECORD_INFO_READ_SECRET_KEY`

### Recommended

- `NEXIS_LOG_LEVEL`
- `NEXIS_DATASET_CATEGORY` (default `nature_landscape_scenery`)
- `NEXIS_VALIDATOR_ENABLED_SPECS` (default `video_v1`)
- `NEXIS_VALIDATOR_SEMANTIC_CHECK_ENABLED` (default `true`)
- `NEXIS_VALIDATOR_SEMANTIC_MODEL`
- `NEXIS_VALIDATOR_SEMANTIC_TIMEOUT_SEC`
- `NEXIS_VALIDATOR_SEMANTIC_MAX_SAMPLES`
- `NEXIS_VALIDATOR_CATEGORY_CHECK_ENABLED` (default `true`)
- `NEXIS_VALIDATOR_CATEGORY_MODEL`
- `NEXIS_VALIDATOR_CATEGORY_TIMEOUT_SEC`
- `NEXIS_VALIDATOR_CATEGORY_MAX_SAMPLES`
- `OPENAI_API_KEY` (preferred; uses `gpt-4o`)
- `GEMINI_API_KEY` (optional fallback for category checks; uses `gemini-3.1-flash-lite-preview` when OpenAI key is unset)
- `NEXIS_VALIDATION_API_URL` (optional evidence API endpoint)
- `NEXIS_VALIDATION_API_TIMEOUT_SEC`

### Owner-Only

- `NEXIS_OWNER_VALIDATOR_HOTKEY`
- `NEXIS_OWNER_DB_BUCKET`
- `NEXIS_OWNER_DB_ACCOUNT_ID`
- `NEXIS_OWNER_DB_READ_ACCESS_KEY`
- `NEXIS_OWNER_DB_READ_SECRET_KEY`
- `NEXIS_OWNER_DB_WRITE_ACCESS_KEY`
- `NEXIS_OWNER_DB_WRITE_SECRET_KEY`
- `NEXIS_RECORD_INFO_WRITE_ACCESS_KEY`
- `NEXIS_RECORD_INFO_WRITE_SECRET_KEY`
- `NEXIS_RECORD_INFO_OBJECT_KEY`

## Runtime Behavior

`nexis validate` runs continuously and:

- validates each closed 50-block interval (`[start, start+50)`) after reserve (`2` blocks)
- runs hard checks on full record sets
- verifies sampled clip/frame assets against row SHA256 fields
- enforces sampled clip resolution (`1280x720`)
- runs optional semantic caption-vs-multi-frame checks on sampled rows
- runs optional category validation (nature/landscape/scenery):
  - caption gate first
  - strict middle-3 frame check for borderline cases
  - rejects when manifest category metadata is missing or unsupported
- prunes row-level overlaps using shared `nexis-record-info` index
- arbitrates same-interval cross-miner overlaps via earliest manifest `created_at`
- fetches invalid hotkeys from API for `[interval_id-500, interval_id]`
- accumulates scores and submits `set_weights` every 250 blocks
- zeros invalid hotkeys before `set_weights`
- owner validator publishes metadata to `nexis-record-info` and updates overlap snapshot

`nexis validate-source-auth` runs source-auth-only validation:

- uses same interval/miner/row sampling logic as main validator
- validates source authenticity and posts invalid hotkeys to API
- anti-bot/download errors are fail-open (miner treated as valid for source-auth stage)

The API-backed blacklist is always applied. Stop with `Ctrl+C`.

## Category Failure Codes

When `NEXIS_VALIDATOR_CATEGORY_CHECK_ENABLED=true`, category-related failures can appear in
validator decision payloads:

- `missing_category_metadata`
  - manifest does not include `category`
- `unsupported_category:<value>`
  - manifest category is present but not currently supported by validator category logic
- `category_caption_reject:<clip_id>`
  - caption-gate stage rejects sampled clip category before strict vision pass
- `category_strict_reject:<clip_id>`
  - strict middle-frame vision evaluation rejects sampled clip category
- `category_strict_frames_missing:<clip_id>`
  - strict stage could not build the required middle-frame set from available semantic frames
- `category_strict_api_key_missing:<clip_id>`
  - strict stage had no usable API key configured for category checks
- `category_strict_client_unavailable:<clip_id>`
  - Gemini OpenAI-compatible client could not be initialized
- `category_strict_response_invalid:<clip_id>`
  - strict stage response was malformed/unparseable or missing required fields

## Local (Non-Docker) Commands

```bash
nexis validate
nexis validate --specs video_v1
# source-auth sidecar
nexis validate-source-auth
# independent owner copy worker
nexis sync-owner-datasets --poll-sec 60
# optional debug logging:
# nexis validate --debug
# optional poll override:
# nexis validate --poll-sec 4
# optional extra excludes for this run:
# nexis validate --exclude-hotkeys hotkey1,hotkey2
```

## Troubleshooting

- Wallet mount issues:
  - verify `BT_WALLET_HOST_PATH` in `compose.env` points to the host wallet directory
  - ensure `BT_WALLET_PATH=/wallets` in `validator.env` matches compose mount target
- S3 credential failures:
  - confirm `R2_ACCOUNT_ID` and `R2_REGION`
  - confirm `NEXIS_RECORD_INFO_*` and (owner mode) `NEXIS_OWNER_DB_*` values
  - confirm miners committed valid `account_id + read credentials`
- LLM provider behavior:
  - validator semantic checks use OpenAI (`gpt-4o`) when `OPENAI_API_KEY` is set
  - if OpenAI is unset but `GEMINI_API_KEY` is set, checks use Gemini (`gemini-3.1-flash-lite-preview`)
  - if neither key is set, disable semantic checks with `NEXIS_VALIDATOR_SEMANTIC_CHECK_ENABLED=false`
  - category checks follow the same provider preference:
    OpenAI first, then Gemini fallback
- Weight submission retries:
  - inspect validator logs for `set_weights failed`
  - confirm wallet hotkey/key material and chain access (`BT_NETWORK`)
  - resolve root cause and keep the validator process running to allow automatic backoff retries

## Watchtower Operations

- production nodes should track `:stable` image tags
- canary nodes can track `:latest` (via `docker-compose.validator.canary.yml`)
- for rollback, pin `NEXIS_VALIDATOR_IMAGE` to a known digest and redeploy