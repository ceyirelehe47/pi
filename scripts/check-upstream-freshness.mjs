#!/usr/bin/env node
/**
 * R0-Pi (#120): MECHANICAL upstream freshness gate.
 *
 * The audited upstream SHA is recorded in docs/iris-fork/production-lock.json
 * (sync.lastVerifiedUpstreamCommit). This gate fetches the CURRENT live
 * upstream (earendil-works/pi main) and compares:
 *
 *   live upstream SHA == audited SHA  -> fresh, exit 0
 *   live upstream SHA != audited SHA  -> STALE, exit 1  (R0/release gate FAIL)
 *
 * A human editing lastVerifiedUpstreamCommit by hand does not make this
 * gate pass — the gate ALWAYS compares against the live remote. It is wired
 * into CI as a required check (see .github/workflows/upstream-freshness.yml).
 */
import { execSync } from "node:child_process";
import * as fs from "node:fs";
import { join } from "node:path";

const REPO_ROOT = join(import.meta.dirname, "..");
const LOCK = join(REPO_ROOT, "docs", "iris-fork", "production-lock.json");
const UPSTREAM_REMOTE = "earendil";
const UPSTREAM_BRANCH = "main";

function sh(cmd, opts = {}) {
  return execSync(cmd, { cwd: REPO_ROOT, stdio: "pipe", encoding: "utf8", ...opts }).trim();
}

const lock = JSON.parse(fs.readFileSync(LOCK, "utf8"));
const audited = lock.sync?.lastVerifiedUpstreamCommit;
if (typeof audited !== "string" || audited.length !== 40) {
  console.error(`UPSTREAM FRESHNESS: no audited SHA in ${LOCK} (sync.lastVerifiedUpstreamCommit)`);
  process.exit(1);
}

// Fetch the live upstream (never assume the recorded SHA is still current).
try {
  sh(`git fetch ${UPSTREAM_REMOTE} ${UPSTREAM_BRANCH} --prune`);
} catch (error) {
  console.error(`UPSTREAM FRESHNESS: cannot fetch ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}: ${error.message}`);
  process.exit(1);
}
const live = sh(`git rev-parse ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}`);

if (live !== audited) {
  console.error(
    `UPSTREAM FRESHNESS: STALE\n  audited: ${audited} (${lock.sync.lastVerifiedAt})\n  live:    ${live} (${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH})\n` +
      `R0/release gate FAIL — run the upstream sync + carried-patch audit, then update production-lock.json.`,
  );
  process.exit(1);
}

console.log(`UPSTREAM FRESHNESS: OK (audited ${audited} == live ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH})`);
