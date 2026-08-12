import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { createModels, type FauxProviderHandle, fauxProvider } from "@iris/pi-ai";
import { describe, expect, it } from "vitest";
import { AgentHarness, computeMessageContentHash } from "../../src/harness/agent-harness.ts";
import { NodeExecutionEnv } from "../../src/harness/env/nodejs.ts";
import {
	JsonlSessionBackend,
	JsonlSessionRepository,
	loadJsonlSessionMetadata,
} from "../../src/harness/session/jsonl-repo.ts";
import { InMemorySessionRepository } from "../../src/harness/session/memory-repo.ts";
import type { Session } from "../../src/harness/session/session.ts";
import type { MessageFinalizedEvent } from "../../src/harness/types.ts";
import { createAssistantMessage, createTempDir, createUserMessage } from "./session-test-utils.ts";

const models = createModels();
let fauxCount = 0;
function newFaux(): FauxProviderHandle {
	const faux = fauxProvider({ provider: `seam-faux-${++fauxCount}` });
	models.setProvider(faux.provider);
	return faux;
}

async function appendUsageEntries(session: Session) {
	const assistant = createAssistantMessage("reply");
	if (assistant.role !== "assistant") throw new Error("Expected assistant test message");
	assistant.usage = {
		input: 10,
		output: 20,
		cacheRead: 30,
		cacheWrite: 40,
		totalTokens: 100,
		cost: { input: 0.1, output: 0.2, cacheRead: 0.3, cacheWrite: 0.4, total: 1 },
	};
	await session.appendMessage(assistant);
	await session.appendCompaction("summary", undefined, 1234, undefined, undefined, {
		input: 1,
		output: 2,
		cacheRead: 3,
		cacheWrite: 4,
		totalTokens: 10,
		cost: { input: 0.01, output: 0.02, cacheRead: 0.03, cacheWrite: 0.04, total: 0.1 },
	});
	await session.moveTo(await session.getLeafId(), {
		summary: "branch",
		usage: {
			input: 5,
			output: 6,
			cacheRead: 7,
			cacheWrite: 8,
			totalTokens: 26,
			cost: { input: 0.05, output: 0.06, cacheRead: 0.07, cacheWrite: 0.08, total: 0.26 },
		},
	});
}

describe("Session aggregate", () => {
	it("repository disposal closes its owned storage", async () => {
		let session: Session;
		{
			await using repository = new InMemorySessionRepository();
			session = await repository.create({ id: "session-1" });
		}

		await expect(session!.appendMessage(createUserMessage("late"))).rejects.toThrow(
			"In-memory session repository is disposed",
		);
	});

	it("owns leaf navigation, labels, names, stats, and branch traversal", async () => {
		const repo = new InMemorySessionRepository();
		const session = await repo.create({ id: "session-1" });
		const root = await session.appendMessage(createUserMessage("root"));
		const child = await session.appendMessage(createAssistantMessage("child"));
		await session.appendLabel(root, "checkpoint");
		await session.appendSessionName(" review\nname ");

		expect(await session.getLeafId()).not.toBeNull();
		expect(await session.getLabel(root)).toBe("checkpoint");
		expect(await session.getSessionName()).toBe("review name");
		expect((await session.getBranch(child)).map((entry) => entry.id)).toEqual([root, child]);

		await session.moveTo(root);
		expect(await session.getLeafId()).toBe(root);
		expect((await session.getEntries()).at(-1)).toMatchObject({ type: "leaf", targetId: root });
		await expect(session.moveTo("missing")).rejects.toThrow("Entry missing not found");
	});

	it("serializes concurrent appends into one parent chain", async () => {
		const session = await new InMemorySessionRepository().create({});
		await Promise.all(
			Array.from({ length: 20 }, (_, index) => session.appendMessage(createUserMessage(`message ${index}`))),
		);
		const entries = await session.getEntries();
		expect(entries).toHaveLength(20);
		for (let index = 0; index < entries.length; index++) {
			expect(entries[index]!.parentId).toBe(index === 0 ? null : entries[index - 1]!.id);
		}
	});

	it("includes assistant and summary usage in statistics", async () => {
		const session = await new InMemorySessionRepository().create({});
		await appendUsageEntries(session);
		expect(await session.getSessionStats()).toEqual({
			messageCount: 1,
			cachedTokens: 40,
			uncachedTokens: 68,
			totalTokens: 136,
			costTotal: 1.36,
		});
	});

	it("stops branch traversal at retained-tail compaction", async () => {
		const session = await new InMemorySessionRepository().create({});
		await session.appendMessage(createUserMessage("root"));
		const child = await session.appendMessage(createAssistantMessage("child"));
		const compaction = await session.appendCompaction("summary", child, 1234, undefined, undefined, undefined, [
			createAssistantMessage("child"),
		]);
		const tail = await session.appendMessage(createUserMessage("after"));
		expect((await session.getBranch(tail)).map((entry) => entry.id)).toEqual([compaction, tail]);
	});
});

describe("JsonlSessionBackend", () => {
	it("writes headers and entries and reopens the aggregate", async () => {
		const root = createTempDir();
		const repo = new JsonlSessionRepository({ fs: new NodeExecutionEnv({ cwd: root }), sessionsRoot: root });
		const session = await repo.create({ cwd: root, id: "session-1", metadata: { profile: "reviewer" } });
		const metadata = await session.getMetadata();
		const entryId = await session.appendMessage(createUserMessage("one"));
		const reopened = await repo.open(metadata);

		expect(existsSync(metadata.path)).toBe(true);
		expect((await reopened.getEntries()).map((entry) => entry.id)).toEqual([entryId]);
		expect((await loadJsonlSessionMetadata(new NodeExecutionEnv({ cwd: root }), metadata.path)).metadata).toEqual({
			profile: "reviewer",
		});
		const lines = readFileSync(metadata.path, "utf8").trim().split("\n");
		expect(JSON.parse(lines[0]!)).toMatchObject({ type: "session", version: 3, id: "session-1" });
		expect(JSON.parse(lines[1]!)).toMatchObject({ id: entryId, type: "message" });
	});

	it("fails loudly for malformed headers and entries", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const repo = new JsonlSessionRepository({ fs: env, sessionsRoot: root });
		const session = await repo.create({ cwd: root, id: "session-1" });
		const metadata = await session.getMetadata();
		writeFileSync(metadata.path, "not json\n");
		await expect(repo.open(metadata)).rejects.toMatchObject({ code: "invalid_session" });

		const header = { type: "session", version: 3, id: "session-1", timestamp: metadata.createdAt, cwd: root };
		// A malformed LAST line is a torn tail (iris_agent#51): quarantined with
		// a typed diagnostic, not a fatal invalid_entry — an interrupted append
		// must not make the whole session unreadable. Mid-file corruption below
		// still fails closed.
		writeFileSync(metadata.path, `${JSON.stringify(header)}\nnot json\n`);
		const reopened = await repo.open(metadata);
		expect((await reopened.getEntries()).length).toBe(0);
		expect(await reopened.journalDiagnostics()).toEqual([
			"torn_tail: line 2: line is not valid JSON (Unexpected token 'o', \"not json\" is not valid JSON); quarantined",
		]);
		// Mid-file corruption stays fail-closed.
		writeFileSync(
			metadata.path,
			`${JSON.stringify(header)}\nnot json\n${JSON.stringify({ type: "message", id: "e2", parentId: null, timestamp: metadata.createdAt, message: createUserMessage("x") })}\n`,
		);
		await expect(repo.open(metadata)).rejects.toMatchObject({ code: "invalid_entry" });
	});

	it("enforces entry uniqueness and does not recreate deleted files", async () => {
		const root = createTempDir();
		const backend = new JsonlSessionBackend({ fs: new NodeExecutionEnv({ cwd: root }), sessionsRoot: root });
		const snapshot = await backend.create({ cwd: root, id: "session-1" });
		const entry = {
			type: "message" as const,
			id: "entry-1",
			parentId: null,
			timestamp: "2026-01-01T00:00:00.000Z",
			message: createUserMessage("one"),
		};
		await snapshot.appendEntry(entry);
		await expect(snapshot.appendEntry(entry)).rejects.toThrow("Entry entry-1 already exists");
		await backend.delete(snapshot.metadata);
		await expect(snapshot.appendEntry({ ...entry, id: "entry-2" })).rejects.toThrow("Session not found");
		expect(existsSync(snapshot.metadata.path)).toBe(false);
	});

	it("scopes entry uniqueness to the session path", async () => {
		const root = createTempDir();
		const backend = new JsonlSessionBackend({ fs: new NodeExecutionEnv({ cwd: root }), sessionsRoot: root });
		const first = await backend.create({ cwd: "/tmp/first", id: "shared-session-id" });
		const second = await backend.create({ cwd: "/tmp/second", id: "shared-session-id" });
		const entry = {
			type: "message" as const,
			id: "shared-entry-id",
			parentId: null,
			timestamp: "2026-01-01T00:00:00.000Z",
			message: createUserMessage("one"),
		};

		await first.appendEntry(entry);
		await expect(second.appendEntry(entry)).resolves.toBeUndefined();
	});

	it("rejects non-object header metadata", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const repo = new JsonlSessionRepository({ fs: env, sessionsRoot: root });
		const session = await repo.create({ cwd: root, id: "session-1" });
		const metadata = await session.getMetadata();
		const header = {
			type: "session",
			version: 3,
			id: metadata.id,
			timestamp: metadata.createdAt,
			cwd: root,
			metadata: "profile",
		};
		writeFileSync(metadata.path, `${JSON.stringify(header)}\n`);
		await expect(repo.open(metadata)).rejects.toThrow("session header metadata must be an object");
	});
});

describe("JSONL crash-consistent commit journal (iris_agent#40 Feature 2)", () => {
	it("survives a simulated crash: pending receipt persists and replays after reopening", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const repo = new JsonlSessionRepository({ fs: env, sessionsRoot: root });
		const session = await repo.create({ cwd: root, id: "session-crash-1" });
		const metadata = await session.getMetadata();

		// Simulate a crash between the durable append and publication: record
		// the entry + pending receipt at the storage level, then "die" by
		// disposing without ever publishing message_finalized.
		const message = createUserMessage("crash window");
		const contentHash = await computeMessageContentHash(message);
		await session.appendMessageWithCommitReceipt(message, (entryId) => ({
			sessionId: metadata.id,
			entryId,
			contentHash,
			committedAt: new Date().toISOString(),
		}));
		await repo[Symbol.asyncDispose]();

		// Reopen the same session file with a fresh repository (process restart).
		const reopenedRepo = new JsonlSessionRepository({ fs: env, sessionsRoot: root });
		const reopened = await reopenedRepo.open(metadata);
		// The journal survived the restart; the entry is readable and the
		// receipt is still pending (no phantom ack was written).
		expect((await reopened.readPendingCommitReceipts()).length).toBe(1);

		const finalized: MessageFinalizedEvent[] = [];
		const harness = new AgentHarness({
			models,
			session: reopened,
			model: newFaux().getModel(),
			systemPrompt: "You are helpful.",
		});
		harness.subscribe((event) => {
			if (event.type === "message_finalized") finalized.push(event as MessageFinalizedEvent);
		});

		expect(await harness.recoverPendingCommitReceipts()).toBe(1);
		expect(finalized.length).toBe(1);
		expect(finalized[0]!.receipt.contentHash).toBe(contentHash);
		expect(finalized[0]!.receipt.entryId).toBe(await reopened.getLeafId());
		expect(await reopened.readPendingCommitReceipts()).toEqual([]);

		// Reopening again must not re-emit (ack marker persisted).
		const reopenedAgainRepo = new JsonlSessionRepository({ fs: env, sessionsRoot: root });
		const reopenedAgain = await reopenedAgainRepo.open(metadata);
		const harness2 = new AgentHarness({
			models,
			session: reopenedAgain,
			model: newFaux().getModel(),
			systemPrompt: "You are helpful.",
		});
		const secondRun: MessageFinalizedEvent[] = [];
		harness2.subscribe((event) => {
			if (event.type === "message_finalized") secondRun.push(event as MessageFinalizedEvent);
		});
		expect(await harness2.recoverPendingCommitReceipts()).toBe(0);
		expect(secondRun.length).toBe(0);
		await reopenedAgainRepo[Symbol.asyncDispose]();
	});

	it("loadJsonlSession skips journal marker lines (they are not entries)", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const repo = new JsonlSessionRepository({ fs: env, sessionsRoot: root });
		const session = await repo.create({ cwd: root, id: "session-journal-1" });
		const metadata = await session.getMetadata();
		const message = createUserMessage("marker test");
		const contentHash = await computeMessageContentHash(message);
		await session.appendMessageWithCommitReceipt(message, (entryId) => ({
			sessionId: metadata.id,
			entryId,
			contentHash,
			committedAt: new Date().toISOString(),
		}));
		await repo[Symbol.asyncDispose]();

		// The file now contains entry + receipt marker (+ no ack yet). Opening
		// must yield exactly one session entry (the message), not the marker.
		const reopenedRepo = new JsonlSessionRepository({ fs: env, sessionsRoot: root });
		const reopened = await reopenedRepo.open(metadata);
		const entries = await reopened.getEntries();
		expect(entries.length).toBe(1);
		expect(entries[0]!.type).toBe("message");
		await reopenedRepo[Symbol.asyncDispose]();
	});

	it("does not mistake entry text containing journal marker strings for marker lines", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const repo = new JsonlSessionRepository({ fs: env, sessionsRoot: root });
		const session = await repo.create({ cwd: root, id: "session-marker-fp" });
		const metadata = await session.getMetadata();

		// Plain append path (no journal): the message text contains both marker
		// strings. A substring-based marker filter would drop this entry on
		// reload; the structural check must keep it.
		const entryId = await session.appendMessage(
			createUserMessage("please log __piReceiptAck and __piReceipt in the output"),
		);
		await repo[Symbol.asyncDispose]();

		const reopenedRepo = new JsonlSessionRepository({ fs: env, sessionsRoot: root });
		const reopened = await reopenedRepo.open(metadata);
		const entries = await reopened.getEntries();
		expect(entries.length).toBe(1);
		expect(entries[0]!.id).toBe(entryId);
		expect(entries[0]!.type).toBe("message");
		await reopenedRepo[Symbol.asyncDispose]();
	});

	it("crash recovery does not deadlock when the pending receipt message text contains marker strings", async () => {
		const root = createTempDir();
		const env = new NodeExecutionEnv({ cwd: root });
		const repo = new JsonlSessionRepository({ fs: env, sessionsRoot: root });
		const session = await repo.create({ cwd: root, id: "session-marker-fp-recovery" });
		const metadata = await session.getMetadata();

		// Journal path with marker-looking text in the message.
		const message = createUserMessage("mention __piReceiptAck inside the committed message");
		const contentHash = await computeMessageContentHash(message);
		await session.appendMessageWithCommitReceipt(message, (entryId) => ({
			sessionId: metadata.id,
			entryId,
			contentHash,
			committedAt: new Date().toISOString(),
		}));
		await repo[Symbol.asyncDispose]();

		// Reopen: entry must survive and recovery must replay, not throw.
		const reopenedRepo = new JsonlSessionRepository({ fs: env, sessionsRoot: root });
		const reopened = await reopenedRepo.open(metadata);
		expect((await reopened.getEntries()).length).toBe(1);

		const finalized: MessageFinalizedEvent[] = [];
		const harness = new AgentHarness({
			models,
			session: reopened,
			model: newFaux().getModel(),
			systemPrompt: "You are helpful.",
		});
		harness.subscribe((event) => {
			if (event.type === "message_finalized") finalized.push(event as MessageFinalizedEvent);
		});
		expect(await harness.recoverPendingCommitReceipts()).toBe(1);
		expect(finalized.length).toBe(1);
		await reopenedRepo[Symbol.asyncDispose]();
	});
});
