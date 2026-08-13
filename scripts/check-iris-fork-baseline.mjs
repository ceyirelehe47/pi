// Iris fork provenance gate.
//
// Validates docs/iris-fork/production-lock.json and docs/iris-fork/carried-patches.json.
// Fail-closed: any malformed or unsupported value is an error and the check exits non-zero.
// No third-party dependencies. Schema authority lives here (single source of truth).
//
// Usage:
//   node scripts/check-iris-fork-baseline.mjs [productionLockPath] [carriedPatchesPath]
// Defaults to the repo's docs/iris-fork/ manifests.

import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const SCRIPTS_DIR = dirname(fileURLToPath(import.meta.url));
const DEFAULT_LOCK_PATH = join(SCRIPTS_DIR, "..", "docs", "iris-fork", "production-lock.json");
const DEFAULT_PATCHES_PATH = join(SCRIPTS_DIR, "..", "docs", "iris-fork", "carried-patches.json");
const DEFAULT_MIGRATION_MANIFEST_PATH = join(
	SCRIPTS_DIR,
	"..",
	"packages",
	"storage",
	"sqlite-node",
	"src",
	"sqlite",
	"migrations",
	"release-manifest.json",
);
const DEFAULT_MIGRATIONS_DIR = join(SCRIPTS_DIR, "..", "packages", "storage", "sqlite-node", "src", "sqlite", "migrations");
const MIGRATION_MANIFEST_SCHEMA_VERSION = 1;
const MIGRATION_SHA256_PATTERN = /^[0-9a-f]{64}$/;

export const SUPPORTED_SCHEMA_VERSION = 2;

export const FORK_REPOSITORY = "blueforst/pi";
export const UPSTREAM_REPOSITORY = "earendil-works/pi";
export const PACKAGE_IDENTITY_STATUSES = ["inherits_upstream_package_names", "independent_identity", "independent_package_identity"];
export const PUBLISH_STATUSES = ["not_published", "publish_forbidden", "ready_for_fork_release"];
export const DEPENDENCY_DIRECTIONS = ["upstream_only"];
export const SYNC_STRATEGIES = [
	"manual_review_gate",
	"manual_review_gate + mechanical freshness gate (scripts/check-upstream-freshness.mjs)",
];
export const PATCH_STATUSES = ["proposed", "carried", "upstreamed", "removable", "removed"];
export const UPSTREAM_STATUSES = ["not_filed", "filed", "open", "merged", "rejected", "superseded"];

const SHA_PATTERN = /^[0-9a-f]{40}$/;
const ZERO_SHA = "0000000000000000000000000000000000000000";
const ISO_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
const PLACEHOLDER_PATTERN = /(?:^|\b)(?:TBD|TODO|unknown)(?:\b|$)/i;

const PRODUCTION_LOCK_REQUIRED_FIELDS = [
	"schemaVersion",
	"fork.repository",
	"fork.defaultBranch",
	"fork.baselineCommit",
	"acceptedRuntime.repository",
	"acceptedRuntime.commit",
	"acceptedRuntime.tree",
	"acceptedRuntime.verifiedAt",
	"upstream.repository",
	"upstream.baseCommit",
	"upstream.verifiedAt",
	"runtime.node",
	"runtime.packageManager",
	"runtime.lockfile",
	"distribution.packageIdentityStatus",
	"distribution.publishStatus",
	"dependencyDirection",
	"sync.strategy",
	"sync.lastVerifiedUpstreamCommit",
	"sync.lastVerifiedAt",
];

// firstForkCommit / latestForkCommit are optional: a proposed patch may not
// have landed any fork commit yet. When present they must be full SHAs.
const PATCH_REQUIRED_FIELDS = [
	"id",
	"title",
	"status",
	"genericRuntimeRationale",
	"affectedPackages",
	"affectedSurfaces",
	"tests",
	"upstream.issue",
	"upstream.pullRequest",
	"upstream.status",
	"removalCondition",
	"compatibilityRisk",
	"notes",
];

function isObject(value) {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function getPath(obj, path) {
	return path.split(".").reduce((acc, key) => (acc == null ? undefined : acc[key]), obj);
}

function isNonEmptyString(value) {
	return typeof value === "string" && value.length > 0;
}

function isSha(value) {
	return typeof value === "string" && SHA_PATTERN.test(value);
}

function isZeroSha(value) {
	return value === ZERO_SHA;
}

function isIsoDate(value) {
	if (typeof value !== "string" || !ISO_DATE_PATTERN.test(value)) return false;
	const [year, month, day] = value.split("-").map(Number);
	if (month < 1 || month > 12 || day < 1 || day > 31) return false;
	const date = new Date(Date.UTC(year, month - 1, day));
	return date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 && date.getUTCDate() === day;
}

function isNullishOrPositiveInt(value) {
	return value === null || (Number.isInteger(value) && value > 0);
}

function collectStrings(value, path, out) {
	if (typeof value === "string") {
		out.push([path, value]);
	} else if (Array.isArray(value)) {
		value.forEach((item, index) => collectStrings(item, `${path}[${index}]`, out));
	} else if (isObject(value)) {
		for (const [key, child] of Object.entries(value)) {
			collectStrings(child, path === "" ? key : `${path}.${key}`, out);
		}
	}
}

function placeholderErrors(value, label) {
	const strings = [];
	collectStrings(value, "", strings);
	const errors = [];
	for (const [path, text] of strings) {
		if (PLACEHOLDER_PATTERN.test(text)) {
			const where = path === "" ? "(root)" : path;
			errors.push(`${label}: ${where}: placeholder value not allowed: ${JSON.stringify(text)}`);
		}
	}
	return errors;
}

function checkRequiredFields(obj, requiredFields, label) {
	const errors = [];
	for (const path of requiredFields) {
		if (getPath(obj, path) === undefined) {
			errors.push(`${label}: missing required field: ${path}`);
		}
	}
	return errors;
}

function checkShaField(obj, path, label) {
	const errors = [];
	const value = getPath(obj, path);
	if (!isSha(value)) {
		errors.push(`${label}: ${path}: expected full 40-character hex commit SHA, got ${JSON.stringify(value)}`);
	} else if (isZeroSha(value)) {
		errors.push(`${label}: ${path}: zero SHA is not a valid baseline`);
	}
	return errors;
}

function checkIsoDateField(obj, path, label) {
	const errors = [];
	const value = getPath(obj, path);
	if (!isIsoDate(value)) {
		errors.push(`${label}: ${path}: expected ISO date YYYY-MM-DD, got ${JSON.stringify(value)}`);
	}
	return errors;
}

function checkEnumField(obj, path, allowed, label) {
	const errors = [];
	const value = getPath(obj, path);
	if (!allowed.includes(value)) {
		errors.push(`${label}: ${path}: unsupported value ${JSON.stringify(value)}, expected one of ${allowed.map((v) => JSON.stringify(v)).join(", ")}`);
	}
	return errors;
}

function checkNonEmptyStringField(obj, path, label) {
	const errors = [];
	const value = getPath(obj, path);
	if (!isNonEmptyString(value)) {
		errors.push(`${label}: ${path}: expected non-empty string, got ${JSON.stringify(value)}`);
	}
	return errors;
}

function checkNonEmptyStringArray(obj, path, label) {
	const errors = [];
	const value = getPath(obj, path);
	if (!Array.isArray(value) || value.length === 0) {
		errors.push(`${label}: ${path}: expected non-empty array of strings, got ${JSON.stringify(value)}`);
		return errors;
	}
	for (let index = 0; index < value.length; index++) {
		if (!isNonEmptyString(value[index])) {
			errors.push(`${label}: ${path}[${index}]: expected non-empty string, got ${JSON.stringify(value[index])}`);
		}
	}
	return errors;
}

// Cross-field invariants for schema v1. Only checked when both values exist;
// missing values are reported by the required-field checks instead.
function checkLockInvariants(lock, label) {
	const errors = [];
	const defaultBranch = getPath(lock, "fork.defaultBranch");
	if (defaultBranch !== undefined && defaultBranch !== "main") {
		errors.push(`${label}: fork.defaultBranch: schema v${SUPPORTED_SCHEMA_VERSION} requires "main", got ${JSON.stringify(defaultBranch)}`);
	}
	const packageManager = getPath(lock, "runtime.packageManager");
	if (packageManager !== undefined && packageManager !== "npm") {
		errors.push(`${label}: runtime.packageManager: schema v${SUPPORTED_SCHEMA_VERSION} requires "npm", got ${JSON.stringify(packageManager)}`);
	}
	const lockfile = getPath(lock, "runtime.lockfile");
	if (lockfile !== undefined && lockfile !== "package-lock.json") {
		errors.push(`${label}: runtime.lockfile: schema v${SUPPORTED_SCHEMA_VERSION} requires "package-lock.json", got ${JSON.stringify(lockfile)}`);
	}
	const upstreamBase = getPath(lock, "upstream.baseCommit");
	const lastVerifiedUpstream = getPath(lock, "sync.lastVerifiedUpstreamCommit");
	if (upstreamBase !== undefined && lastVerifiedUpstream !== undefined && lastVerifiedUpstream !== upstreamBase) {
		errors.push(`${label}: sync.lastVerifiedUpstreamCommit must equal upstream.baseCommit, got ${JSON.stringify(lastVerifiedUpstream)} vs ${JSON.stringify(upstreamBase)}`);
	}
	const verifiedAt = getPath(lock, "upstream.verifiedAt");
	const lastVerifiedAt = getPath(lock, "sync.lastVerifiedAt");
	if (verifiedAt !== undefined && lastVerifiedAt !== undefined && lastVerifiedAt < verifiedAt) {
		errors.push(`${label}: sync.lastVerifiedAt must be equal to or later than upstream.verifiedAt, got ${JSON.stringify(lastVerifiedAt)} vs ${JSON.stringify(verifiedAt)}`);
	}
	return errors;
}

export function validateProductionLock(lock, label = "production-lock.json") {
	const errors = [];
	if (!isObject(lock)) {
		errors.push(`${label}: expected a JSON object, got ${JSON.stringify(lock)}`);
		return errors;
	}
	errors.push(...checkRequiredFields(lock, PRODUCTION_LOCK_REQUIRED_FIELDS, label));
	if (lock.schemaVersion !== SUPPORTED_SCHEMA_VERSION) {
		errors.push(`${label}: schemaVersion: unsupported value ${JSON.stringify(lock.schemaVersion)}, expected ${SUPPORTED_SCHEMA_VERSION}`);
	}
	checkNonEmptyStringField(lock, "fork.defaultBranch", label).forEach((e) => errors.push(e));
	if (getPath(lock, "fork.repository") !== FORK_REPOSITORY) {
		errors.push(`${label}: fork.repository: expected ${FORK_REPOSITORY}, got ${JSON.stringify(getPath(lock, "fork.repository"))}`);
	}
	if (getPath(lock, "upstream.repository") !== UPSTREAM_REPOSITORY) {
		errors.push(`${label}: upstream.repository: expected ${UPSTREAM_REPOSITORY}, got ${JSON.stringify(getPath(lock, "upstream.repository"))}`);
	}
	errors.push(...checkShaField(lock, "fork.baselineCommit", label));
	errors.push(...checkShaField(lock, "upstream.baseCommit", label));
	errors.push(...checkShaField(lock, "sync.lastVerifiedUpstreamCommit", label));
	errors.push(...checkIsoDateField(lock, "upstream.verifiedAt", label));
	errors.push(...checkIsoDateField(lock, "sync.lastVerifiedAt", label));
	// acceptedRuntime: the immutable identity consumers are allowed to run.
	// It must be a real commit AND a real tree of that commit; it must not be
	// the initial fork baseline (issue iris_agent#41: the lock must express an
	// accepted runtime identity beyond the bootstrap baseline).
	if (getPath(lock, "acceptedRuntime.repository") !== FORK_REPOSITORY) {
		errors.push(`${label}: acceptedRuntime.repository: expected ${FORK_REPOSITORY}, got ${JSON.stringify(getPath(lock, "acceptedRuntime.repository"))}`);
	}
	errors.push(...checkShaField(lock, "acceptedRuntime.commit", label));
	errors.push(...checkShaField(lock, "acceptedRuntime.tree", label));
	errors.push(...checkIsoDateField(lock, "acceptedRuntime.verifiedAt", label));
	const baselineCommit = getPath(lock, "fork.baselineCommit");
	const acceptedCommit = getPath(lock, "acceptedRuntime.commit");
	if (acceptedCommit !== undefined && baselineCommit !== undefined && acceptedCommit === baselineCommit) {
		errors.push(`${label}: acceptedRuntime.commit must differ from fork.baselineCommit (the accepted runtime identity cannot be only the initial bootstrap baseline)`);
	}
	["runtime.node", "runtime.packageManager", "runtime.lockfile"].forEach((path) => checkNonEmptyStringField(lock, path, label).forEach((e) => errors.push(e)));
	errors.push(...checkEnumField(lock, "distribution.packageIdentityStatus", PACKAGE_IDENTITY_STATUSES, label));
	errors.push(...checkEnumField(lock, "distribution.publishStatus", PUBLISH_STATUSES, label));
	errors.push(...checkEnumField(lock, "dependencyDirection", DEPENDENCY_DIRECTIONS, label));
	errors.push(...checkEnumField(lock, "sync.strategy", SYNC_STRATEGIES, label));
	errors.push(...checkLockInvariants(lock, label));
	errors.push(...placeholderErrors(lock, label));
	return errors;
}

export function validateCarriedPatches(doc, label = "carried-patches.json") {
	const errors = [];
	if (!isObject(doc)) {
		errors.push(`${label}: expected a JSON object, got ${JSON.stringify(doc)}`);
		return errors;
	}
	if (doc.schemaVersion !== SUPPORTED_SCHEMA_VERSION) {
		errors.push(`${label}: schemaVersion: unsupported value ${JSON.stringify(doc.schemaVersion)}, expected ${SUPPORTED_SCHEMA_VERSION}`);
	}
	errors.push(...checkShaField(doc, "upstreamBaseCommit", label));
	if (!Array.isArray(doc.patches)) {
		errors.push(`${label}: patches: expected an array, got ${JSON.stringify(doc.patches)}`);
		return errors;
	}
	const seenIds = new Set();
	for (let index = 0; index < doc.patches.length; index++) {
		const patch = doc.patches[index];
		const patchLabel = `${label}: patches[${index}]`;
		if (!isObject(patch)) {
			errors.push(`${patchLabel}: expected a patch object, got ${JSON.stringify(patch)}`);
			continue;
		}
		errors.push(...checkRequiredFields(patch, PATCH_REQUIRED_FIELDS, patchLabel));
		checkNonEmptyStringField(patch, "id", patchLabel).forEach((e) => errors.push(e));
		// Duplicate detection only applies once the id is known to be non-empty.
		if (isNonEmptyString(patch.id)) {
			if (seenIds.has(patch.id)) {
				errors.push(`${patchLabel}: id: duplicate patch id: ${JSON.stringify(patch.id)}`);
			}
			seenIds.add(patch.id);
		}
		checkNonEmptyStringField(patch, "title", patchLabel).forEach((e) => errors.push(e));
		checkNonEmptyStringField(patch, "genericRuntimeRationale", patchLabel).forEach((e) => errors.push(e));
		checkNonEmptyStringField(patch, "removalCondition", patchLabel).forEach((e) => errors.push(e));
		checkNonEmptyStringField(patch, "compatibilityRisk", patchLabel).forEach((e) => errors.push(e));
		errors.push(...checkNonEmptyStringArray(patch, "affectedPackages", patchLabel));
		errors.push(...checkNonEmptyStringArray(patch, "affectedSurfaces", patchLabel));
		errors.push(...checkNonEmptyStringArray(patch, "tests", patchLabel));
		errors.push(...checkEnumField(patch, "status", PATCH_STATUSES, patchLabel));
		errors.push(...checkEnumField(patch, "upstream.status", UPSTREAM_STATUSES, patchLabel));
		for (const path of ["upstream.issue", "upstream.pullRequest"]) {
			const value = getPath(patch, path);
			if (!isNullishOrPositiveInt(value)) {
				errors.push(`${patchLabel}: ${path}: expected null or a positive integer, got ${JSON.stringify(value)}`);
			}
		}
		const upstreamStatus = getPath(patch, "upstream.status");
		if (upstreamStatus === "not_filed" && (getPath(patch, "upstream.issue") !== null || getPath(patch, "upstream.pullRequest") !== null)) {
			errors.push(`${patchLabel}: upstream.status is "not_filed" but upstream.issue/upstream.pullRequest are not null`);
		}
		// Commit identity: only "proposed" may omit fork commit ids; every other
		// status must point at real, non-zero fork commits.
		const patchStatus = getPath(patch, "status");
		const requiresCommitIdentity = patchStatus !== undefined && patchStatus !== "proposed";
		for (const path of ["firstForkCommit", "latestForkCommit"]) {
			const value = getPath(patch, path);
			if (requiresCommitIdentity && value === undefined) {
				errors.push(`${patchLabel}: ${path}: required for status ${JSON.stringify(patchStatus)}; only "proposed" may omit commit identity`);
			} else if (value !== undefined) {
				errors.push(...checkShaField(patch, path, patchLabel));
			}
		}
		errors.push(...placeholderErrors(patch, patchLabel));
	}
	return errors;
}

export function validateBaselineConsistency(lock, patchesDoc, lockLabel = "production-lock.json", patchesLabel = "carried-patches.json") {
	const errors = [];
	if (!isObject(lock) || !isObject(patchesDoc)) return errors;
	const lockBase = getPath(lock, "upstream.baseCommit");
	const patchesBase = getPath(patchesDoc, "upstreamBaseCommit");
	// Missing values are reported by the per-manifest checks; only compare
	// when both are present so partial documents never crash this pass.
	if (lockBase !== undefined && patchesBase !== undefined && patchesBase !== lockBase) {
		errors.push(
			`${patchesLabel}: upstreamBaseCommit ${JSON.stringify(patchesBase)} does not match ${lockLabel}: upstream.baseCommit ${JSON.stringify(lockBase)}`
		);
	}
	return errors;
}

// Git-backed provenance validation (issue iris_agent#41).
// Requires a git repository at repoDir; verifies the acceptedRuntime identity
// really exists in that repository and that its tree matches the lock.
// Optionally verifies carried patches are ancestors of the accepted commit.
export function validateAcceptedRuntimeInGit(lock, repoDir, { verifyPatches = false, patchesDoc = null } = {}) {
	const errors = [];
	const acceptedCommit = getPath(lock, "acceptedRuntime.commit");
	const acceptedTree = getPath(lock, "acceptedRuntime.tree");
	if (!isSha(acceptedCommit) || !isSha(acceptedTree)) {
		// Shape errors are reported by validateProductionLock; nothing to do here.
		return errors;
	}
	try {
		// rev-parse --verify accepts any 40-hex string even when the object
		// does not exist; cat-file -e is the existence check.
		const head = spawnSync("git", ["-C", repoDir, "cat-file", "-e", `${acceptedCommit}^{commit}`], {
			encoding: "utf8",
		});
		if (head.status !== 0) {
			errors.push(`acceptedRuntime.commit ${acceptedCommit} does not exist in git repository ${repoDir}`);
			return errors;
		}
	} catch (error) {
		errors.push(`acceptedRuntime git check failed for ${repoDir}: ${error.message}`);
		return errors;
	}
	const treeCheck = spawnSync("git", ["-C", repoDir, "rev-parse", `${acceptedCommit}^{tree}`], {
		encoding: "utf8",
	});
	if (treeCheck.status !== 0) {
		errors.push(`cannot resolve tree of acceptedRuntime.commit ${acceptedCommit}: ${treeCheck.stderr?.trim()}`);
		return errors;
	}
	const actualTree = treeCheck.stdout.trim();
	if (actualTree !== acceptedTree) {
		errors.push(
			`acceptedRuntime.tree mismatch: lock records ${acceptedTree}, git resolves ${actualTree} for commit ${acceptedCommit}`
		);
	}
	if (verifyPatches && isObject(patchesDoc) && Array.isArray(patchesDoc.patches)) {
		for (let index = 0; index < patchesDoc.patches.length; index++) {
			const patch = patchesDoc.patches[index];
			if (!isObject(patch)) continue;
			// Only patches whose identity is fixed in the fork must be
			// contained in the accepted runtime commit.
			const status = patch.status;
			if (!["carried", "upstreamed", "removable", "removed"].includes(status)) continue;
			const latest = patch.latestForkCommit;
			if (!isSha(latest)) continue;
			const ancestorCheck = spawnSync("git", ["-C", repoDir, "merge-base", "--is-ancestor", latest, acceptedCommit], {
				encoding: "utf8",
			});
			if (ancestorCheck.status !== 0) {
				errors.push(
					`carried-patches.json: patches[${index}] ${JSON.stringify(patch.id)}: latestForkCommit ${latest} is not an ancestor of acceptedRuntime.commit ${acceptedCommit}`
				);
			}
		}
	}
	return errors;
}

// Migration release manifest (iris_agent#67/#78): the release-owned checksum
// trust root for the SQLite migration set. It is the SINGLE source shared
// with the runtime (`packages/storage/sqlite-node/src/sqlite/migrations.ts`),
// this provenance gate and CI. The manifest array order IS the release order.
// Any mismatch between the manifest and the packaged migration files fails
// closed here so a release can never silently self-heal from mutable SQL.
export function validateReleaseManifest(manifest, label = "release-manifest.json") {
	const errors = [];
	if (!isObject(manifest)) {
		errors.push(`${label}: expected a JSON object, got ${JSON.stringify(manifest)}`);
		return errors;
	}
	if (manifest.schemaVersion !== MIGRATION_MANIFEST_SCHEMA_VERSION) {
		errors.push(
			`${label}: schemaVersion: unsupported value ${JSON.stringify(manifest.schemaVersion)}, expected ${MIGRATION_MANIFEST_SCHEMA_VERSION}`
		);
	}
	if (!Array.isArray(manifest.migrations)) {
		errors.push(`${label}: migrations: expected an array, got ${JSON.stringify(manifest.migrations)}`);
		return errors;
	}
	const seenIds = new Set();
	for (let index = 0; index < manifest.migrations.length; index++) {
		const entry = manifest.migrations[index];
		const entryLabel = `${label}: migrations[${index}]`;
		if (!isObject(entry)) {
			errors.push(`${entryLabel}: expected a migration object, got ${JSON.stringify(entry)}`);
			continue;
		}
		checkNonEmptyStringField(entry, "id", entryLabel).forEach((e) => errors.push(e));
		const sha256 = getPath(entry, "sha256");
		if (typeof sha256 !== "string" || !MIGRATION_SHA256_PATTERN.test(sha256)) {
			errors.push(
				`${entryLabel}: sha256: expected a 64-character lowercase hex sha256, got ${JSON.stringify(sha256)}`
			);
		}
		if (isNonEmptyString(entry.id)) {
			if (seenIds.has(entry.id)) {
				errors.push(`${entryLabel}: id: duplicate migration id: ${JSON.stringify(entry.id)}`);
			}
			seenIds.add(entry.id);
		}
	}
	return errors;
}

export function checkMigrationReleaseManifest(
	manifestPath = DEFAULT_MIGRATION_MANIFEST_PATH,
	migrationsDir = DEFAULT_MIGRATIONS_DIR
) {
	const errors = [];
	let manifest;
	try {
		manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
	} catch (error) {
		return { ok: false, errors: [`${manifestPath}: cannot read or parse: ${error.message}`] };
	}
	errors.push(...validateReleaseManifest(manifest, manifestPath));
	if (!Array.isArray(manifest.migrations)) {
		return { ok: false, errors };
	}
	let files;
	try {
		files = readdirSync(migrationsDir)
			.filter((name) => name.endsWith(".sql"))
			.sort();
	} catch (error) {
		return { ok: false, errors: [`${migrationsDir}: cannot list packaged migration files: ${error.message}`] };
	}
	const manifestIds = manifest.migrations.map((entry) => entry.id);
	if (manifestIds.join("\u0000") !== files.join("\u0000")) {
		errors.push(
			`${manifestPath}: packaged migration files do not match the manifest entries: manifest lists ` +
				`[${manifestIds.join(", ")}], package ships [${files.join(", ")}]`
		);
	}
	const checksumsById = new Map(manifest.migrations.map((entry) => [entry.id, entry.sha256]));
	for (const file of files) {
		const expected = checksumsById.get(file);
		if (expected === undefined) continue; // set mismatch already reported
		let digest;
		try {
			digest = createHash("sha256").update(readFileSync(join(migrationsDir, file))).digest("hex");
		} catch (error) {
			errors.push(`${migrationsDir}/${file}: cannot read packaged migration: ${error.message}`);
			continue;
		}
		if (digest !== expected) {
			errors.push(
				`${manifestPath}: packaged migration ${file} sha256 ${digest} does not match its release-owned ` +
					`manifest checksum ${expected}`
			);
		}
	}
	return { ok: errors.length === 0, errors };
}

export function checkIrisForkBaseline(lockPath, patchesPath) {
	const lockLabel = lockPath;
	const patchesLabel = patchesPath;
	let lock;
	let patchesDoc;
	try {
		lock = JSON.parse(readFileSync(lockPath, "utf8"));
	} catch (error) {
		return { ok: false, errors: [`${lockLabel}: cannot read or parse: ${error.message}`] };
	}
	try {
		patchesDoc = JSON.parse(readFileSync(patchesPath, "utf8"));
	} catch (error) {
		return { ok: false, errors: [`${patchesLabel}: cannot read or parse: ${error.message}`] };
	}
	const errors = [
		...validateProductionLock(lock, lockLabel),
		...validateCarriedPatches(patchesDoc, patchesLabel),
		...validateBaselineConsistency(lock, patchesDoc, lockLabel, patchesLabel),
		...checkMigrationReleaseManifest().errors,
	];
	return { ok: errors.length === 0, errors };
}

function main() {
	const args = process.argv.slice(2);
	const verifyGit = args.includes("--verify-git");
	const positional = args.filter((arg) => arg !== "--verify-git");
	// Optional third positional: git repository to verify against.
	// Defaults to the repository hosting this script (the Pi fork checkout).
	const [lockArg, patchesArg, gitRepoArg] = positional;
	const lockPath = lockArg ?? DEFAULT_LOCK_PATH;
	const patchesPath = patchesArg ?? DEFAULT_PATCHES_PATH;
	const result = checkIrisForkBaseline(lockPath, patchesPath);
	const errors = [...result.errors];
	if (verifyGit) {
		// Verify the accepted runtime identity against the repository that
		// hosts this manifest (the Pi fork checkout itself). Repo root is the
		// script's repo root, independent of the manifest paths passed in.
		let lock;
		try {
			lock = JSON.parse(readFileSync(lockPath, "utf8"));
		} catch {
			lock = null;
		}
		let patchesDoc = null;
		try {
			patchesDoc = JSON.parse(readFileSync(patchesPath, "utf8"));
		} catch {
			patchesDoc = null;
		}
		if (lock !== null) {
			const repoDir = gitRepoArg ?? dirname(SCRIPTS_DIR); // <repo-root>/scripts -> <repo-root>
			errors.push(...validateAcceptedRuntimeInGit(lock, repoDir, { verifyPatches: true, patchesDoc }));
		}
	}
	if (errors.length === 0) {
		console.log("OK: iris fork baseline manifests are valid" + (verifyGit ? " (git-verified)" : ""));
		return;
	}
	console.error("FAIL: iris fork baseline manifests are invalid:");
	for (const error of errors) {
		console.error(`- ${error}`);
	}
	process.exit(1);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
	main();
}
