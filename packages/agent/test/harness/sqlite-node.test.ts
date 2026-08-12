import { join } from "node:path";
import { createModels, type FauxProviderHandle, fauxProvider } from "@iris/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import {
	createNodeSqliteFactory,
	createSqliteSessionSearch,
	SqliteSessionBackend,
	type SqliteSessionMetadata,
	SqliteSessionRepository,
} from "../../../storage/sqlite-node/src/index.ts";
import { AgentHarness, computeMessageContentHash } from "../../src/harness/agent-harness.ts";
import { NodeExecutionEnv } from "../../src/harness/env/nodejs.ts";
import { JsonlSessionRepository } from "../../src/harness/session/jsonl-repo.ts";
import { createScanningSessionSearch } from "../../src/harness/session/search.ts";
import type { Session } from "../../src/harness/session/session.ts";
import type {
	MessageFinalizedEvent,
	SessionSearch,
	SessionSearchHit,
	SessionSearchOptions,
} from "../../src/harness/types.ts";
import { createTempDir, createUserMessage } from "./session-test-utils.ts";

const models = createModels();
let fauxCount = 0;
function newFaux(): FauxProviderHandle {
	const faux = fauxProvider({ provider: `seam-faux-${++fauxCount}` });
	models.setProvider(faux.provider);
	return faux;
}

const ownedRepositories: AsyncDisposable[] = [];

afterEach(async () => {
	for (const repository of ownedRepositories.splice(0)) await repository[Symbol.asyncDispose]();
});

function createSqliteFixture(options: ConstructorParameters<typeof SqliteSessionRepository>[0]) {
	const repository = new SqliteSessionRepository(options);
	ownedRepositories.push(repository);
	return { repository, search: createSqliteSessionSearch(options) };
}

function createJsonlFixture(options: ConstructorParameters<typeof JsonlSessionRepository>[0]) {
	const repository = new JsonlSessionRepository(options);
	ownedRepositories.push(repository);
	return { repository, search: createScanningSessionSearch(repository) };
}

describe("JsonlSessionBackend with scanning search", () => {
	it("searches canonical session entries by scanning", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const { repository: repo, search } = createJsonlFixture({
			fs: env,
			sessionsRoot: join(root, "sessions"),
		});
		const included = await repo.create({ cwd: root, id: "included" });
		const excluded = await repo.create({
			cwd: `${root}/other`,
			id: "excluded",
		});
		const entryId = await included.appendMessage(createUserMessage("Find the auth defect"));
		await excluded.appendMessage(createUserMessage("Find the auth defect"));

		await expect(search.search({ text: "AUTH", cwd: root })).resolves.toEqual([
			expect.objectContaining({
				entryId,
				metadata: expect.objectContaining({ id: "included" }),
			}),
		]);
	});
});

describe("SqliteSessionBackend with explicit SQLite FTS5 search", () => {
	it("uses SQLite FTS5 when composed with its search implementation", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const sqlite = createNodeSqliteFactory();
		const databasePath = join(root, "sessions.sqlite");
		const { repository: repo, search } = createSqliteFixture({
			env,
			sqlite,
			databasePath,
		});
		const included = await repo.create({ cwd: root, id: "included" });
		const excluded = await repo.create({
			cwd: `${root}/other`,
			id: "excluded",
		});
		const metadata = await included.getMetadata();
		const entryId = await included.appendMessage(createUserMessage("Find the auth defect"));
		await excluded.appendMessage(createUserMessage("Find the auth defect"));

		await expect(search.search({ text: "auth", cwd: root })).resolves.toEqual([
			expect.objectContaining({
				entryId,
				metadata: expect.objectContaining({ id: "included" }),
			}),
		]);
		await expect(search.search({ text: "uth", cwd: root })).resolves.toEqual([
			expect.objectContaining({
				entryId,
				metadata: expect.objectContaining({ id: "included" }),
			}),
		]);

		const db = await sqlite.open(databasePath);
		try {
			const tables = await db
				.prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
				.all<{ name: string }>();
			expect(tables.map((row) => row.name)).toContain("session_search_fts");
			expect(tables.map((row) => row.name)).not.toContain("session_search_records");
		} finally {
			await db.close();
		}

		await repo.delete(metadata);
		await expect(search.search({ text: "auth", cwd: root })).resolves.toEqual([]);
	});

	it("creates an empty canonical session without initializing FTS", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const sqlite = createNodeSqliteFactory();
		const databasePath = join(root, "sessions.sqlite");
		const { repository: repo } = createSqliteFixture({
			env,
			sqlite,
			databasePath,
		});
		const session = await repo.create({ cwd: root, id: "session-1" });

		const db = await sqlite.open(databasePath);
		try {
			const fts = await db
				.prepare("SELECT 1 AS found FROM sqlite_master WHERE type = 'table' AND name = 'session_search_fts'")
				.get<{ found: number }>();
			expect(fts).toBeUndefined();
		} finally {
			await db.close();
		}
		await expect(session.appendMessage(createUserMessage("still writable"))).resolves.toBeTypeOf("string");
	});

	it("rolls back canonical appends when co-located FTS trigger writes fail", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const sqlite = createNodeSqliteFactory();
		const databasePath = join(root, "sessions.sqlite");
		const { repository: repo, search } = createSqliteFixture({
			env,
			sqlite,
			databasePath,
		});
		await search.search({ text: "initialize" });
		const session = await repo.create({ cwd: root, id: "session-1" });

		const db = await sqlite.open(databasePath);
		try {
			await db.exec("DROP TABLE session_search_fts");
		} finally {
			await db.close();
		}

		await expect(session.appendMessage(createUserMessage("must roll back"))).rejects.toThrow();
		await expect(session.getEntries()).resolves.toEqual([]);
	});

	it("rolls back canonical deletion when co-located FTS cleanup fails", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const sqlite = createNodeSqliteFactory();
		const databasePath = join(root, "sessions.sqlite");
		const { repository: repo, search } = createSqliteFixture({
			env,
			sqlite,
			databasePath,
		});
		await search.search({ text: "initialize" });
		const session = await repo.create({ cwd: root, id: "session-1" });
		await session.appendMessage(createUserMessage("must remain"));
		const metadata = await session.getMetadata();

		const db = await sqlite.open(databasePath);
		try {
			await db.exec("DROP TABLE session_search_fts");
		} finally {
			await db.close();
		}

		await expect(repo.delete(metadata)).rejects.toThrow();
		const reopened = await repo.open(metadata);
		await expect(reopened.getEntries()).resolves.toHaveLength(1);
	});

	it("initializes canonical storage when searched before the first session is created", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const { repository: repo, search } = createSqliteFixture({
			env,
			sqlite: createNodeSqliteFactory(),
			databasePath: join(root, "sessions.sqlite"),
		});

		await expect(search.search({ text: "auth" })).resolves.toEqual([]);
		const session = await repo.create({ cwd: root, id: "session-1" });
		const entryId = await session.appendMessage(createUserMessage("Find the auth defect"));

		await expect(search.search({ text: "auth" })).resolves.toEqual([
			expect.objectContaining({
				entryId,
				metadata: expect.objectContaining({ id: "session-1" }),
			}),
		]);
		await expect(session.appendMessage(createUserMessage("Still writable"))).resolves.toBeTypeOf("string");
	});
});

describe("SqliteSessionRepository with custom search", () => {
	it("uses an independently supplied search implementation", async () => {
		const root = createTempDir();
		const searches: SessionSearchOptions[] = [];
		const search: SessionSearch<SqliteSessionMetadata> = {
			async search(options): Promise<SessionSearchHit<SqliteSessionMetadata>[]> {
				searches.push(options);
				return [];
			},
		};
		const env = new NodeExecutionEnv({ cwd: root });
		const repo = new SqliteSessionRepository({
			env,
			sqlite: createNodeSqliteFactory(),
			databasePath: join(root, "sessions.sqlite"),
		});
		ownedRepositories.push(repo);
		const session = await repo.create({ cwd: root, id: "session-1" });
		await session.appendMessage(createUserMessage("stored canonically"));

		await expect(search.search({ text: "custom query" })).resolves.toEqual([]);
		expect(searches).toEqual([{ text: "custom query" }]);
	});
});

describe("SQLite crash-consistent commit journal (iris_agent#40 Feature 2)", () => {
	it("persists pending receipts in a transaction and replays them after restart", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const databasePath = join(root, "sessions.sqlite");
		const sqlite = createNodeSqliteFactory();

		// First process: create session + record entry + pending receipt, then
		// "crash" by disposing without publishing (no ack).
		const repo = new SqliteSessionRepository({ env, sqlite, databasePath });
		const session = await repo.create({ cwd: root, id: "session-1" });
		const metadata = await session.getMetadata();
		const message = createUserMessage("crash window");
		const contentHash = await computeMessageContentHash(message);
		await session.appendMessageWithCommitReceipt(message, (entryId) => ({
			sessionId: metadata.id,
			entryId,
			contentHash,
			committedAt: new Date().toISOString(),
		}));
		await repo[Symbol.asyncDispose]();

		// Restart: reopen the same database file.
		const reopenedRepo = new SqliteSessionRepository({
			env,
			sqlite,
			databasePath,
		});
		ownedRepositories.push(reopenedRepo);
		const reopened = await reopenedRepo.open(metadata);
		expect((await reopened.readPendingCommitReceipts()).length).toBe(1);

		const finalized: MessageFinalizedEvent[] = [];
		const harness = new AgentHarness({
			models,
			session: reopened as unknown as Session,
			model: newFaux().getModel(),
			systemPrompt: "You are helpful.",
		});
		harness.subscribe((event) => {
			if (event.type === "message_finalized") finalized.push(event as MessageFinalizedEvent);
		});

		expect(await harness.recoverPendingCommitReceipts()).toBe(1);
		expect(finalized.length).toBe(1);
		expect(finalized[0]!.receipt.contentHash).toBe(contentHash);
		expect(await reopened.readPendingCommitReceipts()).toEqual([]);

		// Third restart must not re-emit (acked row persisted).
		const thirdRepo = new SqliteSessionRepository({
			env,
			sqlite,
			databasePath,
		});
		ownedRepositories.push(thirdRepo);
		const third = await thirdRepo.open(metadata);
		const harness2 = new AgentHarness({
			models,
			session: third as unknown as Session,
			model: newFaux().getModel(),
			systemPrompt: "You are helpful.",
		});
		const secondRun: MessageFinalizedEvent[] = [];
		harness2.subscribe((event) => {
			if (event.type === "message_finalized") secondRun.push(event as MessageFinalizedEvent);
		});
		expect(await harness2.recoverPendingCommitReceipts()).toBe(0);
		expect(secondRun.length).toBe(0);
	});

	it("acknowledges receipts after normal publication (no pending rows remain)", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const repo = new SqliteSessionRepository({
			env,
			sqlite: createNodeSqliteFactory(),
			databasePath: join(root, "sessions.sqlite"),
		});
		ownedRepositories.push(repo);
		const session = await repo.create({ cwd: root, id: "session-1" });
		const finalized: MessageFinalizedEvent[] = [];
		const harness = new AgentHarness({
			models,
			session,
			model: newFaux().getModel(),
			systemPrompt: "You are helpful.",
		});
		harness.subscribe((event) => {
			if (event.type === "message_finalized") finalized.push(event as MessageFinalizedEvent);
		});

		await harness.appendMessage(createUserMessage("normal path"));

		expect(finalized.length).toBe(1);
		expect(await session.readPendingCommitReceipts()).toEqual([]);
		expect(await harness.recoverPendingCommitReceipts()).toBe(0);
		expect(finalized.length).toBe(1);
	});

	it("iris_agent#50: quarantines receipts whose entry is missing or not a message, and messages with unknown roles", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const databasePath = join(root, "sessions.sqlite");
		const sqlite = createNodeSqliteFactory();

		// Process 1: three committed messages (real hashes), then tamper the
		// PERSISTED entries directly: delete the middle entry row and flip the
		// third entry's payload role to an unknown role (leaving its receipt
		// untouched).
		const repo = new SqliteSessionRepository({ env, sqlite, databasePath });
		const session = await repo.create({ cwd: root, id: "session-branches" });
		const metadata = await session.getMetadata();
		const messages = [createUserMessage("one"), createUserMessage("two"), createUserMessage("three")];
		const hashes = await Promise.all(messages.map((message) => computeMessageContentHash(message)));
		const entryIds: string[] = [];
		for (let i = 0; i < messages.length; i++) {
			const { entryId } = await session.appendMessageWithCommitReceipt(messages[i]!, (id) => ({
				sessionId: metadata.id,
				entryId: id,
				contentHash: hashes[i]!,
				committedAt: new Date().toISOString(),
			}));
			entryIds.push(entryId);
		}
		await repo[Symbol.asyncDispose]();

		const tamperDb = await sqlite.open(databasePath);
		await tamperDb
			.prepare("DELETE FROM session_entries WHERE session_id = ? AND id = ?")
			.run(metadata.id, entryIds[1]);
		// Flip the third entry's message role to an unknown role, preserving
		// the stored payload shape (the entry wrapper with .message).
		const thirdRow = (await tamperDb
			.prepare("SELECT payload FROM session_entries WHERE session_id = ? AND id = ?")
			.get(metadata.id, entryIds[2])) as { payload: string };
		const thirdEntry = JSON.parse(thirdRow.payload) as {
			message: { role: string; content: unknown; timestamp: number };
		};
		await tamperDb.prepare("UPDATE session_entries SET payload = ? WHERE session_id = ? AND id = ?").run(
			JSON.stringify({
				...thirdEntry,
				message: { ...thirdEntry.message, role: "evil" },
			}),
			metadata.id,
			entryIds[2],
		);
		await tamperDb.close();

		// Process 2: recovery must quarantine entry 2 (missing_message_entry)
		// and entry 3 (invalid_role), and still emit entry 1 exactly once.
		const reopenedRepo = new SqliteSessionRepository({
			env,
			sqlite,
			databasePath,
		});
		ownedRepositories.push(reopenedRepo);
		const reopened = await reopenedRepo.open(metadata);
		const finalized: MessageFinalizedEvent[] = [];
		const harness = new AgentHarness({
			models,
			session: reopened as unknown as Session,
			model: newFaux().getModel(),
			systemPrompt: "You are helpful.",
		});
		harness.subscribe((event) => {
			if (event.type === "message_finalized") finalized.push(event as MessageFinalizedEvent);
		});

		expect(await harness.recoverPendingCommitReceipts()).toBe(1);
		expect(finalized.map((event) => event.entryId)).toEqual([entryIds[0]]);

		const quarantined = await reopened.readQuarantinedCommitReceipts();
		const byId = new Map(quarantined.map((q) => [q.entryId, q.reason]));
		expect(byId.get(entryIds[1])).toContain("missing_message_entry");
		expect(byId.get(entryIds[2])).toContain("invalid_role");

		// Idempotent across retry: nothing new emits, nothing re-quarantines.
		expect(await harness.recoverPendingCommitReceipts()).toBe(0);
		expect(finalized.length).toBe(1);
		expect((await reopened.readQuarantinedCommitReceipts()).length).toBe(2);
	});

	it("iris_agent#50: receipts sharing one committed_at replay in exact append order (authoritative receipt_seq)", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const databasePath = join(root, "sessions.sqlite");
		const sqlite = createNodeSqliteFactory();

		const repo = new SqliteSessionRepository({ env, sqlite, databasePath });
		const session = await repo.create({ cwd: root, id: "session-ms-tie" });
		const metadata = await session.getMetadata();
		const sameTimestamp = "2026-08-07T00:00:00.000Z";
		const texts = ["first", "second", "third"];
		const entryIds: string[] = [];
		for (const text of texts) {
			const message = createUserMessage(text);
			const contentHash = await computeMessageContentHash(message);
			const { entryId } = await session.appendMessageWithCommitReceipt(message, (id) => ({
				sessionId: metadata.id,
				entryId: id,
				contentHash,
				committedAt: sameTimestamp, // identical millisecond timestamp
			}));
			entryIds.push(entryId);
		}
		await repo[Symbol.asyncDispose]();

		// Restart and replay: order must follow receipt_seq (append order),
		// NOT committed_at (all identical) and NOT entry_id (opaque).
		const reopenedRepo = new SqliteSessionRepository({
			env,
			sqlite,
			databasePath,
		});
		ownedRepositories.push(reopenedRepo);
		const reopened = await reopenedRepo.open(metadata);
		const pending = await reopened.readPendingCommitReceipts();
		expect(pending.map((receipt) => receipt.entryId)).toEqual(entryIds);
		expect(pending.map((receipt) => receipt.entrySeq)).toEqual([1, 2, 3]);

		const finalized: MessageFinalizedEvent[] = [];
		const harness = new AgentHarness({
			models,
			session: reopened as unknown as Session,
			model: newFaux().getModel(),
			systemPrompt: "You are helpful.",
		});
		harness.subscribe((event) => {
			if (event.type === "message_finalized") finalized.push(event as MessageFinalizedEvent);
		});
		expect(await harness.recoverPendingCommitReceipts()).toBe(3);
		expect(finalized.map((event) => event.entryId)).toEqual(entryIds);
	});

	it("iris_agent#50: a tampered persisted receipt is quarantined, never emitted, never acked; valid receipts still emit", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const databasePath = join(root, "sessions.sqlite");
		const sqlite = createNodeSqliteFactory();

		// Process 1: three committed messages with pending receipts (crash
		// window), using REAL canonical hashes.
		const repo = new SqliteSessionRepository({ env, sqlite, databasePath });
		const session = await repo.create({ cwd: root, id: "session-tamper" });
		const metadata = await session.getMetadata();
		const messages = [createUserMessage("first"), createUserMessage("second"), createUserMessage("third")];
		const hashes = await Promise.all(messages.map((message) => computeMessageContentHash(message)));
		const entryIds: string[] = [];
		for (let i = 0; i < messages.length; i++) {
			const { entryId } = await session.appendMessageWithCommitReceipt(messages[i]!, (id) => ({
				sessionId: metadata.id,
				entryId: id,
				contentHash: hashes[i]!,
				committedAt: new Date().toISOString(),
			}));
			entryIds.push(entryId);
		}
		await repo[Symbol.asyncDispose]();

		// Tamper the SECOND receipt's content hash directly in the database
		// (simulates a corrupted/tampered persisted row).
		const tamperDb = await sqlite.open(databasePath);
		await tamperDb
			.prepare("UPDATE session_commit_receipts SET content_hash = ? WHERE session_id = ? AND entry_id = ?")
			.run("f".repeat(64), metadata.id, entryIds[1]);
		await tamperDb.close();

		// Process 2: recovery must emit 2/3 (valid ones), quarantine the
		// tampered one with a typed reason, and never ack it.
		const reopenedRepo = new SqliteSessionRepository({
			env,
			sqlite,
			databasePath,
		});
		ownedRepositories.push(reopenedRepo);
		const reopened = await reopenedRepo.open(metadata);
		const finalized: MessageFinalizedEvent[] = [];
		const harness = new AgentHarness({
			models,
			session: reopened as unknown as Session,
			model: newFaux().getModel(),
			systemPrompt: "You are helpful.",
		});
		harness.subscribe((event) => {
			if (event.type === "message_finalized") finalized.push(event as MessageFinalizedEvent);
		});

		expect(await harness.recoverPendingCommitReceipts()).toBe(2);
		expect(finalized.map((event) => event.entryId)).toEqual([entryIds[0], entryIds[2]]);

		// Quarantined row visible in health diagnostics with typed reason.
		const quarantined = await reopened.readQuarantinedCommitReceipts();
		expect(quarantined.length).toBe(1);
		expect(quarantined[0]!.entryId).toBe(entryIds[1]);
		expect(quarantined[0]!.reason).toContain("content_hash_mismatch");

		// Retry/restart idempotent: second recovery replays nothing new and
		// the tampered row stays quarantined (never re-emitted, never acked).
		expect(await harness.recoverPendingCommitReceipts()).toBe(0);
		expect(finalized.length).toBe(2);
		expect(await reopened.readPendingCommitReceipts()).toEqual([]);
		expect((await reopened.readQuarantinedCommitReceipts()).length).toBe(1);

		// Third process: quarantine persists across restart.
		const thirdRepo = new SqliteSessionRepository({
			env,
			sqlite,
			databasePath,
		});
		ownedRepositories.push(thirdRepo);
		const third = await thirdRepo.open(metadata);
		const thirdHarness = new AgentHarness({
			models,
			session: third as unknown as Session,
			model: newFaux().getModel(),
			systemPrompt: "You are helpful.",
		});
		const thirdFinalized: MessageFinalizedEvent[] = [];
		thirdHarness.subscribe((event) => {
			if (event.type === "message_finalized") thirdFinalized.push(event as MessageFinalizedEvent);
		});
		expect(await thirdHarness.recoverPendingCommitReceipts()).toBe(0);
		expect(thirdFinalized.length).toBe(0);
		expect((await third.readQuarantinedCommitReceipts()).length).toBe(1);
	});

	it("iris_agent#62: a failed receipt transaction leaves NO phantom in-memory Session entry (cache publishes only after commit)", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const databasePath = join(root, "sessions.sqlite");
		const sqlite = createNodeSqliteFactory();

		// Process 1: create the session so the schema + session row exist.
		const repo = new SqliteSessionRepository({ env, sqlite, databasePath });
		const session = await repo.create({ cwd: root, id: "session-phantom" });
		const metadata = await session.getMetadata();
		await repo[Symbol.asyncDispose]();

		// Fault injection: make the receipt INSERT fail (entry SQL has already
		// succeeded inside the same transaction). A BEFORE INSERT trigger on
		// session_commit_receipts aborts the outer transaction exactly where
		// the issue's failure point is.
		const faultDb = await sqlite.open(databasePath);
		await faultDb.exec(`
		CREATE TRIGGER fail_receipt_insert
		BEFORE INSERT ON session_commit_receipts
		BEGIN
		SELECT RAISE(ABORT, 'injected receipt insert failure');
		END;
		`);
		await faultDb.close();

		// Process 2: same database, receipt insert always fails. Use the
		// backend DIRECTLY with an explicit entry id so the in-memory caches
		// (byId / materializedState) are observable after the rollback.
		const failedRepo = new SqliteSessionRepository({
			env,
			sqlite,
			databasePath,
		});
		ownedRepositories.push(failedRepo);
		await failedRepo.open(metadata);
		const backend = new SqliteSessionBackend({ env, sqlite, databasePath });
		const failedStorage = await backend.open(metadata);
		const message = createUserMessage("must never become a phantom");
		const contentHash = await computeMessageContentHash(message);
		const phantomEntry = {
			id: "phantom-entry-1",
			parentId: null,
			timestamp: new Date().toISOString(),
			type: "message" as const,
			message,
		};
		await expect(
			failedStorage.appendEntryWithReceipt!(phantomEntry, {
				sessionId: metadata.id,
				entryId: phantomEntry.id,
				contentHash,
				committedAt: new Date().toISOString(),
			}),
		).rejects.toThrow(/injected receipt insert failure/);

		// After the rollback the in-memory connection must behave EXACTLY as if
		// the append never happened:
		//  - readEntry (byId cache) finds nothing;
		//  - readEntries (DB) finds nothing;
		//  - no pending receipt;
		//  - a retry of the SAME entry id is NOT rejected as a duplicate
		//    (a phantom byId entry would make the retry fail with a duplicate).
		expect(await failedStorage.readEntry?.("phantom-entry-1")).toBeUndefined();
		expect((await failedStorage.readEntries?.())?.length ?? 0).toBe(0);
		expect(await failedStorage.readPendingCommitReceipts?.()).toEqual([]);

		// Remove the fault, retry the SAME logical append exactly once.
		const healDb = await sqlite.open(databasePath);
		await healDb.exec("DROP TRIGGER fail_receipt_insert");
		await healDb.close();
		await failedStorage.appendEntryWithReceipt!(phantomEntry, {
			sessionId: metadata.id,
			entryId: phantomEntry.id,
			contentHash,
			committedAt: new Date().toISOString(),
		});
		expect((await failedStorage.readEntries?.())?.length ?? 0).toBe(1);
		const pendingAfterRetry = await failedStorage.readPendingCommitReceipts?.();
		expect(pendingAfterRetry?.length ?? 0).toBe(1);
		// The retry got seq 1 — the failed transaction did NOT advance the
		// cached sequence (a phantom would have consumed seq 1).
		expect(pendingAfterRetry?.[0]?.entrySeq).toBe(1);
		expect(pendingAfterRetry?.[0]?.entryId).toBe("phantom-entry-1");

		// Restart: same-process and restarted observations agree.
		await backend[Symbol.asyncDispose]();
		await failedRepo[Symbol.asyncDispose]();
		const restartedRepo = new SqliteSessionRepository({
			env,
			sqlite,
			databasePath,
		});
		ownedRepositories.push(restartedRepo);
		const restarted = await restartedRepo.open(metadata);
		expect((await restarted.getEntries()).length).toBe(1);
		expect((await restarted.readPendingCommitReceipts()).map((r) => r.entryId)).toEqual(["phantom-entry-1"]);
	});

	it("iris_agent#62: successful append+receipt still publishes caches once, in authoritative seq order", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const databasePath = join(root, "sessions.sqlite");
		const sqlite = createNodeSqliteFactory();

		const repo = new SqliteSessionRepository({ env, sqlite, databasePath });
		ownedRepositories.push(repo);
		const session = await repo.create({ cwd: root, id: "session-cache-once" });
		const metadata = await session.getMetadata();
		const first = createUserMessage("first");
		const second = createUserMessage("second");
		const hash1 = await computeMessageContentHash(first);
		const hash2 = await computeMessageContentHash(second);
		const { entryId: entryId1 } = await session.appendMessageWithCommitReceipt(first, (id) => ({
			sessionId: metadata.id,
			entryId: id,
			contentHash: hash1,
			committedAt: new Date().toISOString(),
		}));
		await session.appendMessageWithCommitReceipt(second, (id) => ({
			sessionId: metadata.id,
			entryId: id,
			contentHash: hash2,
			committedAt: new Date().toISOString(),
		}));

		// Caches visible immediately (same-process reads).
		expect((await session.getEntries()).length).toBe(2);
		expect((await session.getEntry(entryId1))?.type).toBe("message");
		// Authoritative order: entry_seq 1 then 2 (same millisecond timestamps
		// would still order by seq, not committed_at).
		expect((await session.readPendingCommitReceipts()).map((r) => r.entrySeq)).toEqual([1, 2]);

		// Duplicate-id rejection still works from the cache after the retry
		// (single publish): replay the same entry id through the backend.
		const backend = new SqliteSessionBackend({ env, sqlite, databasePath });
		const storage = await backend.open(metadata);
		await expect(
			storage.appendEntryWithReceipt!(
				{
					id: entryId1,
					parentId: null,
					timestamp: new Date().toISOString(),
					type: "message",
					message: first,
				},
				{
					sessionId: metadata.id,
					entryId: entryId1,
					contentHash: hash1,
					committedAt: new Date().toISOString(),
				},
			),
		).rejects.toThrow(/Failed to append SQLite session entry/);
		// The rejected duplicate attempt did not create a third entry.
		expect((await storage.readEntries()).length).toBe(2);
		await backend[Symbol.asyncDispose]();
	});
});
