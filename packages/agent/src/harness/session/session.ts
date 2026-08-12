import { type ImageContent, type TextContent, type Usage, uuidv7 } from "@iris/pi-ai";
import type { AgentMessage } from "../../types.ts";
import { createBranchSummaryMessage, createCompactionSummaryMessage, createCustomMessage } from "../messages.ts";
import type {
	ActiveToolsChangeEntry,
	BranchSummaryEntry,
	CompactionEntry,
	CustomEntry,
	CustomMessageEntry,
	LabelEntry,
	LeafEntry,
	MessageEntry,
	ModelChangeEntry,
	QuarantinedCommitReceipt,
	SessionBranchQuery,
	SessionCommitReceipt,
	SessionContext,
	SessionEntryCursorOptions,
	SessionInfoEntry,
	SessionMetadata,
	SessionStats,
	SessionStorage,
	SessionTreeEntry,
	ThinkingLevelChangeEntry,
} from "../types.ts";
import { SessionError } from "../types.ts";

export type ContextEntryTransform = (entries: readonly SessionTreeEntry[]) => readonly SessionTreeEntry[];

export type CustomEntryContextMessageProjector = (
	entry: CustomEntry,
	index: number,
	entries: readonly SessionTreeEntry[],
) => readonly AgentMessage[] | undefined;

export interface SessionContextBuildOptions {
	/** Additional entry transforms applied after the default compaction transform. */
	entryTransforms?: readonly ContextEntryTransform[];
	/** Optional custom-entry projectors. Custom entries are omitted from model context by default. */
	entryProjectors?: Readonly<Record<string, CustomEntryContextMessageProjector>>;
}

function deriveSessionContextState(pathEntries: readonly SessionTreeEntry[]): Omit<SessionContext, "messages"> {
	let thinkingLevel = "off";
	let model: { provider: string; modelId: string } | null = null;
	let activeToolNames: string[] | null = null;

	for (const entry of pathEntries) {
		if (entry.type === "thinking_level_change") {
			thinkingLevel = entry.thinkingLevel;
		} else if (entry.type === "model_change") {
			model = { provider: entry.provider, modelId: entry.modelId };
		} else if (entry.type === "message" && entry.message.role === "assistant") {
			model = { provider: entry.message.provider, modelId: entry.message.model };
		} else if (entry.type === "active_tools_change") {
			activeToolNames = [...entry.activeToolNames];
		}
	}

	return { thinkingLevel, model, activeToolNames };
}

export function defaultContextEntryTransform(pathEntries: readonly SessionTreeEntry[]): SessionTreeEntry[] {
	let compaction: CompactionEntry | null = null;
	for (const entry of pathEntries) {
		if (entry.type === "compaction") {
			compaction = entry;
		}
	}
	if (!compaction) {
		return [...pathEntries];
	}

	const entries: SessionTreeEntry[] = [compaction];
	const compactionIdx = pathEntries.findIndex((entry) => entry.type === "compaction" && entry.id === compaction.id);
	if (compaction.retainedTail) {
		for (let i = compactionIdx + 1; i < pathEntries.length; i++) {
			entries.push(pathEntries[i]!);
		}
		return entries;
	}
	if (compaction.firstKeptEntryId) {
		let foundFirstKept = false;
		for (let i = 0; i < compactionIdx; i++) {
			const entry = pathEntries[i]!;
			if (entry.id === compaction.firstKeptEntryId) foundFirstKept = true;
			if (foundFirstKept) entries.push(entry);
		}
	}
	for (let i = compactionIdx + 1; i < pathEntries.length; i++) {
		entries.push(pathEntries[i]!);
	}
	return entries;
}

export function buildContextEntries(
	pathEntries: readonly SessionTreeEntry[],
	options: SessionContextBuildOptions = {},
): SessionTreeEntry[] {
	let entries = defaultContextEntryTransform(pathEntries);
	for (const transform of options.entryTransforms ?? []) {
		entries = [...transform(entries)];
	}
	return entries;
}

export function sessionEntryToContextMessages(
	entry: SessionTreeEntry,
	index: number,
	entries: readonly SessionTreeEntry[],
	options: SessionContextBuildOptions = {},
): AgentMessage[] {
	if (entry.type === "message") {
		return [entry.message as AgentMessage];
	}
	if (entry.type === "custom_message") {
		return [
			createCustomMessage(
				entry.customType,
				entry.content as string | (TextContent | ImageContent)[],
				entry.display,
				entry.details,
				entry.timestamp,
			),
		];
	}
	if (entry.type === "compaction") {
		return [
			createCompactionSummaryMessage(entry.summary, entry.tokensBefore, entry.timestamp),
			...(entry.retainedTail ?? []),
		];
	}
	if (entry.type === "branch_summary" && entry.summary) {
		return [createBranchSummaryMessage(entry.summary, entry.fromId, entry.timestamp)];
	}
	if (entry.type === "custom") {
		return [...(options.entryProjectors?.[entry.customType]?.(entry, index, entries) ?? [])];
	}
	return [];
}

export function buildSessionContext(
	pathEntries: readonly SessionTreeEntry[],
	options: SessionContextBuildOptions = {},
): SessionContext {
	const state = deriveSessionContextState(pathEntries);
	const contextEntries = buildContextEntries(pathEntries, options);
	const messages = contextEntries.flatMap((entry, index) =>
		sessionEntryToContextMessages(entry, index, contextEntries, options),
	);
	return { ...state, messages };
}

export class Session<TMetadata extends SessionMetadata = SessionMetadata> {
	private readonly storage: SessionStorage<TMetadata>;
	private readonly metadata: TMetadata;
	private leafId: string | null;
	private readonly contextBuildOptions: SessionContextBuildOptions;
	private appendTail: Promise<void> = Promise.resolve();

	/** @internal Construct sessions through SessionRepository. */
	constructor(
		storage: SessionStorage<TMetadata>,
		leafId: string | null,
		contextBuildOptions: SessionContextBuildOptions = {},
	) {
		this.storage = storage;
		this.metadata = storage.metadata;
		this.leafId = leafId;
		this.contextBuildOptions = contextBuildOptions;
	}

	async getMetadata(): Promise<TMetadata> {
		return this.metadata;
	}
	async getLeafId(): Promise<string | null> {
		return this.leafId;
	}
	async getEntry(id: string): Promise<SessionTreeEntry | undefined> {
		return this.storage.readEntry(id);
	}
	async getEntries(options?: SessionEntryCursorOptions): Promise<SessionTreeEntry[]> {
		return [...(await this.storage.readEntries(options))];
	}

	async getBranch(fromId?: string | null): Promise<SessionTreeEntry[]> {
		return [...(await this.storage.readPathToRootOrCompaction(fromId === undefined ? this.leafId : fromId))];
	}

	async findEntriesOnBranch(query: SessionBranchQuery = {}): Promise<SessionTreeEntry[]> {
		return [
			...(await this.storage.findEntriesOnBranch({
				...query,
				start: query.start === undefined ? this.leafId : query.start,
			})),
		];
	}

	async findEntryOnBranch(query: SessionBranchQuery = {}): Promise<SessionTreeEntry | undefined> {
		return (await this.findEntriesOnBranch({ ...query, limit: 1 }))[0];
	}

	async buildContextEntries(options: SessionContextBuildOptions = {}): Promise<SessionTreeEntry[]> {
		return buildContextEntries(await this.getBranch(), this.mergeContextBuildOptions(options));
	}

	async buildContext(options: SessionContextBuildOptions = {}): Promise<SessionContext> {
		return buildSessionContext(await this.getBranch(), this.mergeContextBuildOptions(options));
	}

	private mergeContextBuildOptions(options: SessionContextBuildOptions): SessionContextBuildOptions {
		return {
			entryTransforms: [...(this.contextBuildOptions.entryTransforms ?? []), ...(options.entryTransforms ?? [])],
			entryProjectors: {
				...(this.contextBuildOptions.entryProjectors ?? {}),
				...(options.entryProjectors ?? {}),
			},
		};
	}

	async getLabel(id: string): Promise<string | undefined> {
		return this.storage.getLabel(id);
	}
	async getSessionStats(): Promise<SessionStats> {
		return this.storage.getStats();
	}
	async getSessionName(): Promise<string | undefined> {
		return this.storage.getName();
	}

	private async createEntryId(): Promise<string> {
		for (let i = 0; i < 100; i++) {
			const id = uuidv7().slice(-8);
			if (!(await this.getEntry(id))) return id;
		}
		return uuidv7();
	}

	private enqueueAppend<TEntry extends SessionTreeEntry>(
		createEntry: (base: Pick<SessionTreeEntry, "id" | "parentId" | "timestamp">) => TEntry,
	): Promise<TEntry> {
		const commit = this.appendTail.then(async () => {
			const entry = createEntry({
				id: await this.createEntryId(),
				parentId: this.leafId,
				timestamp: new Date().toISOString(),
			});
			await this.storage.appendEntry(entry);
			this.leafId = entry.type === "leaf" ? entry.targetId : entry.id;
			return entry;
		});
		this.appendTail = commit.then(
			() => undefined,
			() => undefined,
		);
		return commit;
	}

	/**
	 * Crash-consistent append variant (iris_agent#40 / Feature 2): persists the
	 * entry and its pending commit receipt atomically when the storage backend
	 * implements the commit journal, and falls back to the plain append path
	 * otherwise. The returned receipt is exactly the one the caller must
	 * publish (e.g. inside `message_finalized`); after publication the caller
	 * acknowledges it with {@link ackCommitReceipt} so crash recovery does not
	 * replay it.
	 */
	private enqueueAppendWithReceipt<TEntry extends SessionTreeEntry>(
		createEntry: (base: Pick<SessionTreeEntry, "id" | "parentId" | "timestamp">) => TEntry,
		createReceipt: (entry: TEntry) => SessionCommitReceipt,
	): Promise<{ entry: TEntry; receipt: SessionCommitReceipt }> {
		const commit = this.appendTail.then(async () => {
			const entry = createEntry({
				id: await this.createEntryId(),
				parentId: this.leafId,
				timestamp: new Date().toISOString(),
			});
			const receipt = createReceipt(entry);
			// iris_agent#51: the journal is only used when the backend both
			// implements it AND explicitly supports crash-recoverable receipts
			// (e.g. JSONL requires an fsync/syncFile capability). Otherwise fall
			// back to the plain append (receipts are publish-only, not durable)
			// rather than pretending an unsupported durability boundary exists.
			if (
				this.storage.appendEntryWithReceipt !== undefined &&
				this.storage.supportsCrashRecoverableReceipts?.() !== false
			) {
				await this.storage.appendEntryWithReceipt(entry, receipt);
			} else {
				await this.storage.appendEntry(entry);
			}
			this.leafId = entry.type === "leaf" ? entry.targetId : entry.id;
			return { entry, receipt };
		});
		this.appendTail = commit.then(
			() => undefined,
			() => undefined,
		);
		return commit;
	}

	/**
	 * Durable message append that atomically records its pending commit
	 * receipt when the storage backend supports the commit journal. Returns
	 * the committed entry id and the receipt to publish. A crash between this
	 * call and publishing the receipt leaves the receipt pending so
	 * {@link AgentHarness.recoverPendingCommitReceipts} can replay it.
	 */
	async appendMessageWithCommitReceipt(
		message: AgentMessage,
		createReceipt: (entryId: string, entry: MessageEntry) => SessionCommitReceipt,
	): Promise<{ entryId: string; receipt: SessionCommitReceipt }> {
		const { entry, receipt } = await this.enqueueAppendWithReceipt(
			(base) =>
				({
					...base,
					type: "message",
					message,
				}) satisfies MessageEntry,
			(entry) => createReceipt(entry.id, entry),
		);
		return { entryId: entry.id, receipt };
	}

	/** Pending (durably recorded, not yet acknowledged) commit receipts, in commit order. */
	async readPendingCommitReceipts(): Promise<readonly SessionCommitReceipt[]> {
		return (await this.storage.readPendingCommitReceipts?.()) ?? [];
	}

	/** Marks a commit receipt as published; crash recovery skips it afterwards. */
	async ackCommitReceipt(entryId: string): Promise<void> {
		await this.storage.ackCommitReceipt?.(entryId);
	}

	/** iris_agent#50: permanently quarantine an integrity-failed receipt. */
	async quarantineCommitReceipt(entryId: string, reason: string): Promise<void> {
		await this.storage.quarantineCommitReceipt?.(entryId, reason);
	}

	/** iris_agent#50: quarantined receipts (health diagnostics; never re-emitted). */
	async readQuarantinedCommitReceipts(): Promise<readonly QuarantinedCommitReceipt[]> {
		return (await this.storage.readQuarantinedCommitReceipts?.()) ?? [];
	}

	/**
	 * iris_agent#51: whether this Session's storage draws an explicit
	 * durability boundary for entry+receipt journal commits (SQLite
	 * transaction, memory ordering, framed JSONL with fsync). Production
	 * configurations that require crash recovery must check this and fail
	 * closed when false.
	 */
	supportsCrashRecoverableReceipts(): boolean {
		return (
			this.storage.appendEntryWithReceipt !== undefined &&
			this.storage.supportsCrashRecoverableReceipts?.() !== false
		);
	}

	/** Diagnostics from the most recent load (torn tails, checksum failures). */
	async journalDiagnostics(): Promise<readonly string[]> {
		return [...((await this.storage.journalDiagnostics?.()) ?? [])];
	}

	private async setLeafId(leafId: string | null): Promise<LeafEntry> {
		if (leafId !== null && !(await this.getEntry(leafId))) {
			throw new SessionError("not_found", `Entry ${leafId} not found`);
		}
		return this.enqueueAppend((base) => {
			return { ...base, type: "leaf", targetId: leafId };
		});
	}

	private async appendTypedEntry<TEntry extends SessionTreeEntry>(
		createEntry: (base: Pick<SessionTreeEntry, "id" | "parentId" | "timestamp">) => TEntry,
	): Promise<string> {
		return (await this.enqueueAppend(createEntry)).id;
	}

	async appendMessage(message: AgentMessage): Promise<string> {
		return this.appendTypedEntry(
			(base) =>
				({
					...base,
					type: "message",
					message,
				}) satisfies MessageEntry,
		);
	}

	async appendThinkingLevelChange(thinkingLevel: string): Promise<string> {
		return this.appendTypedEntry(
			(base) =>
				({
					...base,
					type: "thinking_level_change",
					thinkingLevel,
				}) satisfies ThinkingLevelChangeEntry,
		);
	}

	async appendModelChange(provider: string, modelId: string): Promise<string> {
		return this.appendTypedEntry(
			(base) =>
				({
					...base,
					type: "model_change",
					provider,
					modelId,
				}) satisfies ModelChangeEntry,
		);
	}

	async appendActiveToolsChange(activeToolNames: string[]): Promise<string> {
		return this.appendTypedEntry(
			(base) =>
				({
					...base,
					type: "active_tools_change",
					activeToolNames: [...activeToolNames],
				}) satisfies ActiveToolsChangeEntry,
		);
	}

	async appendCompaction<T = unknown>(
		summary: string,
		firstKeptEntryId: string | undefined,
		tokensBefore: number,
		details?: T,
		fromHook?: boolean,
		usage?: Usage,
		retainedTail?: AgentMessage[],
	): Promise<string> {
		return this.appendTypedEntry(
			(base) =>
				({
					...base,
					type: "compaction",
					summary,
					firstKeptEntryId,
					tokensBefore,
					retainedTail,
					details,
					usage,
					fromHook,
				}) satisfies CompactionEntry<T>,
		);
	}

	async appendCustomEntry(customType: string, data?: unknown): Promise<string> {
		return this.appendTypedEntry(
			(base) =>
				({
					...base,
					type: "custom",
					customType,
					data,
				}) satisfies CustomEntry,
		);
	}

	async appendCustomMessageEntry<T = unknown>(
		customType: string,
		content: string | (TextContent | ImageContent)[],
		display: boolean,
		details?: T,
	): Promise<string> {
		return this.appendTypedEntry(
			(base) =>
				({
					...base,
					type: "custom_message",
					customType,
					content,
					display,
					details,
				}) satisfies CustomMessageEntry<T>,
		);
	}

	async appendLabel(targetId: string, label: string | undefined): Promise<string> {
		if (!(await this.getEntry(targetId))) {
			throw new SessionError("not_found", `Entry ${targetId} not found`);
		}
		return this.appendTypedEntry(
			(base) =>
				({
					...base,
					type: "label",
					targetId,
					label,
				}) satisfies LabelEntry,
		);
	}

	async appendSessionName(name: string): Promise<string> {
		const sanitizedName = name.replace(/[\r\n]+/g, " ").trim();
		return this.appendTypedEntry(
			(base) =>
				({
					...base,
					type: "session_info",
					name: sanitizedName,
				}) satisfies SessionInfoEntry,
		);
	}

	async moveTo(
		entryId: string | null,
		summary?: { summary: string; details?: unknown; usage?: Usage; fromHook?: boolean },
	): Promise<string | undefined> {
		if (entryId !== null && !(await this.getEntry(entryId))) {
			throw new SessionError("not_found", `Entry ${entryId} not found`);
		}
		await this.setLeafId(entryId);
		if (!summary) return undefined;
		return this.appendTypedEntry(
			(base) =>
				({
					...base,
					type: "branch_summary",
					fromId: entryId ?? "root",
					summary: summary.summary,
					details: summary.details,
					usage: summary.usage,
					fromHook: summary.fromHook,
				}) satisfies BranchSummaryEntry,
		);
	}
}

/** Wraps an opened storage connection for a SessionRepository implementation. */
export async function createSession<TMetadata extends SessionMetadata>(
	storage: SessionStorage<TMetadata>,
	contextBuildOptions: SessionContextBuildOptions = {},
): Promise<Session<TMetadata>> {
	return new Session(storage, (await storage.readHead()).leafId, contextBuildOptions);
}
