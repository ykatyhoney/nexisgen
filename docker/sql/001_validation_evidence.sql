CREATE TABLE IF NOT EXISTS validation_results (
    validator_hotkey TEXT NOT NULL,
    interval_id BIGINT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    accepted BOOLEAN NOT NULL,
    failure_reasons TEXT[] NOT NULL DEFAULT '{}',
    record_count INTEGER NOT NULL DEFAULT 0,
    global_overlap_pruned_count INTEGER NOT NULL DEFAULT 0,
    cross_miner_overlap_pruned_count INTEGER NOT NULL DEFAULT 0,
    signature TEXT NOT NULL,
    signature_timestamp BIGINT NOT NULL,
    signature_nonce TEXT NOT NULL,
    body_sha256 TEXT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (validator_hotkey, interval_id, miner_hotkey)
);

CREATE INDEX IF NOT EXISTS idx_validation_results_validator_interval
    ON validation_results (validator_hotkey, interval_id);

CREATE TABLE IF NOT EXISTS validator_request_nonces (
    validator_hotkey TEXT NOT NULL,
    nonce TEXT NOT NULL,
    signature_timestamp BIGINT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (validator_hotkey, nonce)
);

CREATE INDEX IF NOT EXISTS idx_validator_request_nonces_received_at
    ON validator_request_nonces (received_at);

CREATE TABLE IF NOT EXISTS blacklisted_hotkeys (
    hotkey TEXT PRIMARY KEY,
    reason TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

