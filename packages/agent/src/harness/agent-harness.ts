import {
	type AssistantMessage,
	contentText,
	type ImageContent,
	type Model,
	type Models,
	type RetryCallbacks,
	type RetryPolicy,
	type UserMessage,
} from "@iris/pi-ai";
import { runAgentLoop } from "../agent-loop.ts";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	QueueMode,
	StreamFn,
	ThinkingLevel,
} from "../types.ts";
import { collectEntriesForBranchSummary, generateBranchSummary } from "./compaction/branch-summarization.ts";
import { compact, DEFAULT_COMPACTION_SETTINGS, prepareCompaction } from "./compaction/compaction.ts";
import { convertToLlm } from "./messages.ts";
import { formatPromptTemplateInvocation } from "./prompt-templates.ts";
import { formatSkillInvocation } from "./skills.ts";
import type {
	AbortResult,
	AgentHarnessContextController,
	AgentHarnessEvent,
	AgentHarnessEventResultMap,
	AgentHarnessOptions,
	AgentHarnessOwnEvent,
	AgentHarnessPhase,
	AgentHarnessResources,
	AgentHarnessStreamOptions,
	AgentHarnessStreamOptionsPatch,
	AgentHarnessSystemPrompt,
	AgentHarnessTool,
	AgentHarnessToolContextSource,
	CompactResult,
	NavigateTreeResult,
	PendingSessionWrite,
	PromptTemplate,
	Session,
	Skill,
} from "./types.ts";
import { AgentHarnessError, BranchSummaryError, CompactionError, SessionError, toError } from "./types.ts";

function createUserMessage(text: string, images?: ImageContent[]): UserMessage {
	const content: Array<{ type: "text"; text: string } | ImageContent> = [{ type: "text", text }];
	if (images) content.push(...images);
	return { role: "user", content, timestamp: Date.now() };
}

/**
 * Deterministic content hash for a committed message entry (PI-016 receipt
 * attribution). Uses a stable key-sorted JSON serialization so identical
 * messages always produce identical hashes regardless of key order.
 *
 * Uses the global Web Crypto API (available in Node >=19 and browsers) so the
 * harness keeps bundling for browser platforms (browser-smoke gate).
 */
export async function computeMessageContentHash(message: AgentMessage): Promise<string> {
	const sorted = (value: unknown): unknown => {
		if (Array.isArray(value)) return value.map(sorted);
		if (value !== null && typeof value === "object") {
			const record = value as Record<string, unknown>;
			return Object.fromEntries(
				Object.keys(record)
					.sort()
					.map((key) => [key, sorted(record[key])]),
			);
		}
		return value;
	};
	const data = new TextEncoder().encode(JSON.stringify(sorted(message)));
	const digest = await crypto.subtle.digest("SHA-256", data);
	return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function createFailureMessage(model: Model<any>, error: unknown, aborted: boolean): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text: "" }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		stopReason: aborted ? "aborted" : "error",
		errorMessage: error instanceof Error ? error.message : String(error),
		timestamp: Date.now(),
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
	};
}

function cloneStreamOptions(streamOptions?: AgentHarnessStreamOptions): AgentHarnessStreamOptions {
	return {
		...streamOptions,
		headers: streamOptions?.headers ? { ...streamOptions.headers } : undefined,
		metadata: streamOptions?.metadata ? { ...streamOptions.metadata } : undefined,
	};
}

function findDuplicateNames(names: string[]): string[] {
	const seen = new Set<string>();
	const duplicates = new Set<string>();
	for (const name of names) {
		if (seen.has(name)) duplicates.add(name);
		seen.add(name);
	}
	return [...duplicates];
}

function applyStreamOptionsPatch(
	base: AgentHarnessStreamOptions,
	patch?: AgentHarnessStreamOptionsPatch,
): AgentHarnessStreamOptions {
	const result = cloneStreamOptions(base);
	if (!patch) return result;

	if (Object.hasOwn(patch, "transport")) result.transport = patch.transport;
	if (Object.hasOwn(patch, "timeoutMs")) result.timeoutMs = patch.timeoutMs;
	if (Object.hasOwn(patch, "maxRetries")) result.maxRetries = patch.maxRetries;
	if (Object.hasOwn(patch, "maxRetryDelayMs")) result.maxRetryDelayMs = patch.maxRetryDelayMs;
	if (Object.hasOwn(patch, "cacheRetention")) result.cacheRetention = patch.cacheRetention;

	if (Object.hasOwn(patch, "headers")) {
		if (patch.headers === undefined) {
			result.headers = undefined;
		} else {
			const headers = { ...(result.headers ?? {}) };
			for (const [key, value] of Object.entries(patch.headers)) {
				if (value === undefined) delete headers[key];
				else headers[key] = value;
			}
			result.headers = Object.keys(headers).length > 0 ? headers : undefined;
		}
	}

	if (Object.hasOwn(patch, "metadata")) {
		if (patch.metadata === undefined) {
			result.metadata = undefined;
		} else {
			const metadata = { ...(result.metadata ?? {}) };
			for (const [key, value] of Object.entries(patch.metadata)) {
				if (value === undefined) delete metadata[key];
				else metadata[key] = value;
			}
			result.metadata = Object.keys(metadata).length > 0 ? metadata : undefined;
		}
	}

	return result;
}

const SUBSCRIBER_EVENT_TYPE = "*";

/**
 * iris_agent#50: known AgentMessage roles (base Message roles "user" /
 * "assistant" / "toolResult" plus the harness custom message roles declared
 * in messages.ts). Used to fail closed on a corrupted/tampered entry whose
 * role matches no known shape during receipt recovery.
 */
const VALID_MESSAGE_ROLES: ReadonlySet<string> = new Set([
	"user",
	"assistant",
	"toolResult",
	"bashExecution",
	"custom",
	"branchSummary",
	"compactionSummary",
]);

type AgentHarnessHandler = (event: any, signal?: AbortSignal) => Promise<any> | any;

type TrackedTaskKind = "operation" | "mutation";

function normalizeHarnessError(error: unknown, fallbackCode: AgentHarnessError["code"]): AgentHarnessError {
	if (error instanceof AgentHarnessError) return error;
	const cause = toError(error);
	if (cause instanceof SessionError) return new AgentHarnessError("session", cause.message, cause);
	if (cause instanceof CompactionError) return new AgentHarnessError("compaction", cause.message, cause);
	if (cause instanceof BranchSummaryError) return new AgentHarnessError("branch_summary", cause.message, cause);
	return new AgentHarnessError(fallbackCode, cause.message, cause);
}

function normalizeHookError(error: unknown): AgentHarnessError {
	return normalizeHarnessError(error, "hook");
}

interface AgentHarnessTurnState<
	TContext extends object | undefined,
	TSkill extends Skill = Skill,
	TPromptTemplate extends PromptTemplate = PromptTemplate,
	TTool extends AgentHarnessTool<TContext> = AgentHarnessTool<TContext>,
> {
	messages: AgentMessage[];
	resources: AgentHarnessResources<TSkill, TPromptTemplate>;
	toolContext: TContext;
	streamOptions: AgentHarnessStreamOptions;
	sessionId: string;
	systemPrompt: string;
	model: Model<any>;
	thinkingLevel: ThinkingLevel;
	tools: TTool[];
	activeTools: TTool[];
}

export class AgentHarness<
	TContext extends object | undefined = undefined,
	TSkill extends Skill = Skill,
	TPromptTemplate extends PromptTemplate = PromptTemplate,
	TTool extends AgentHarnessTool<TContext> = AgentHarnessTool<TContext>,
> {
	private session: Session;
	readonly models: Models;
	private phase: AgentHarnessPhase = "idle";
	private activeAbortController?: AbortController;
	private readonly activeTasks = new Map<Promise<void>, TrackedTaskKind>();
	private shutdownPromise?: Promise<void>;
	private isShutdown = false;
	private pendingSessionWrites: PendingSessionWrite[] = [];
	private model: Model<any>;
	private thinkingLevel: ThinkingLevel;
	private systemPrompt: AgentHarnessSystemPrompt<TContext, TSkill, TPromptTemplate, TTool> | undefined;
	private contextController: AgentHarnessContextController<TContext, TSkill, TPromptTemplate, TTool> | undefined;
	private toolContext: AgentHarnessToolContextSource<TContext> | undefined;
	private streamOptions: AgentHarnessStreamOptions;
	private retry: RetryPolicy | undefined;
	private resources: AgentHarnessResources<TSkill, TPromptTemplate>;
	private tools = new Map<string, TTool>();
	private activeToolNames: string[];
	private steerQueue: UserMessage[] = [];
	private steeringQueueMode: QueueMode;
	private followUpQueue: UserMessage[] = [];
	private followUpQueueMode: QueueMode;
	private nextTurnQueue: AgentMessage[] = [];
	private handlers = new Map<string, Set<AgentHarnessHandler>>();

	constructor(options: AgentHarnessOptions<TContext, TSkill, TPromptTemplate, TTool>) {
		this.session = options.session;
		this.models = options.models;
		this.resources = options.resources ?? {};
		this.streamOptions = cloneStreamOptions(options.streamOptions);
		this.retry = options.retry;
		this.systemPrompt = options.systemPrompt;
		this.contextController = options.contextController;
		this.toolContext = options.toolContext;
		this.validateUniqueNames(
			(options.tools ?? []).map((tool) => tool.name),
			"Duplicate tool name(s)",
		);
		for (const tool of options.tools ?? []) {
			this.tools.set(tool.name, tool);
		}
		this.model = options.model;
		this.thinkingLevel = options.thinkingLevel ?? "off";
		this.activeToolNames = options.activeToolNames
			? [...options.activeToolNames]
			: (options.tools ?? []).map((tool) => tool.name);
		this.validateUniqueNames(this.activeToolNames, "Duplicate active tool name(s)");
		this.validateToolNames(this.activeToolNames);
		this.steeringQueueMode = options.steeringMode ?? "one-at-a-time";
		this.followUpQueueMode = options.followUpMode ?? "one-at-a-time";
	}

	private assertNotShutDown(): void {
		if (this.isShutdown) throw new AgentHarnessError("invalid_state", "AgentHarness has been shut down");
	}

	private getHandlers(type: string): Set<AgentHarnessHandler> | undefined {
		return this.handlers.get(type);
	}

	private async emitOwn(event: AgentHarnessOwnEvent<TSkill, TPromptTemplate>, signal?: AbortSignal): Promise<void> {
		for (const listener of this.getHandlers(SUBSCRIBER_EVENT_TYPE) ?? []) {
			try {
				await listener(event, signal);
			} catch (error) {
				throw normalizeHookError(error);
			}
		}
	}

	private async emitAny(event: AgentHarnessEvent<TSkill, TPromptTemplate>, signal?: AbortSignal): Promise<void> {
		for (const listener of this.getHandlers(SUBSCRIBER_EVENT_TYPE) ?? []) {
			try {
				await listener(event, signal);
			} catch (error) {
				throw normalizeHookError(error);
			}
		}
	}

	private async emitHook<TType extends keyof AgentHarnessEventResultMap>(
		event: Extract<AgentHarnessOwnEvent, { type: TType }>,
	): Promise<AgentHarnessEventResultMap[TType] | undefined> {
		const handlers = this.getHandlers(event.type as TType);
		if (!handlers || handlers.size === 0) return undefined;
		let lastResult: AgentHarnessEventResultMap[TType] | undefined;
		for (const handler of handlers) {
			try {
				const result = await handler(event);
				if (result !== undefined) {
					lastResult = result;
				}
			} catch (error) {
				throw normalizeHookError(error);
			}
		}
		return lastResult;
	}

	private retryCallbacks(operation: "compaction" | "branch_summary"): RetryCallbacks {
		return {
			onRetryScheduled: (attempt, maxAttempts, delayMs, errorMessage) =>
				this.emitOwn({ type: "retry_scheduled", operation, attempt, maxAttempts, delayMs, errorMessage }),
			onRetryAttemptStart: () => this.emitOwn({ type: "retry_attempt_start", operation }),
			onRetryFinished: () => this.emitOwn({ type: "retry_finished", operation }),
		};
	}

	private async emitBeforeProviderRequest(
		model: Model<any>,
		sessionId: string,
		streamOptions: AgentHarnessStreamOptions,
	): Promise<AgentHarnessStreamOptions> {
		const handlers = this.getHandlers("before_provider_request");
		let current = cloneStreamOptions(streamOptions);
		if (!handlers || handlers.size === 0) return current;
		for (const handler of handlers) {
			try {
				const result = await handler({
					type: "before_provider_request",
					model,
					sessionId,
					streamOptions: cloneStreamOptions(current),
				});
				if (result?.streamOptions) {
					current = applyStreamOptionsPatch(current, result.streamOptions);
				}
			} catch (error) {
				throw normalizeHookError(error);
			}
		}
		return current;
	}

	private async emitBeforeProviderPayload(model: Model<any>, payload: unknown): Promise<unknown> {
		const handlers = this.getHandlers("before_provider_payload");
		let current = payload;
		if (!handlers || handlers.size === 0) return current;
		for (const handler of handlers) {
			try {
				const result = await handler({ type: "before_provider_payload", model, payload: current });
				if (result !== undefined) {
					current = result.payload;
				}
			} catch (error) {
				throw normalizeHookError(error);
			}
		}
		return current;
	}

	private async emitQueueUpdate(): Promise<void> {
		await this.emitOwn({
			type: "queue_update",
			steer: [...this.steerQueue],
			followUp: [...this.followUpQueue],
			nextTurn: [...this.nextTurnQueue],
		});
	}

	private startOperation(): { signal: AbortSignal; finish: () => void } {
		const abortController = new AbortController();
		let finish = () => {};
		this.activeAbortController = abortController;
		void this.track(
			"operation",
			() =>
				new Promise<void>((resolve) => {
					finish = resolve;
				}),
		);
		return {
			signal: abortController.signal,
			finish: () => {
				this.activeAbortController = undefined;
				finish();
			},
		};
	}

	private async track<T>(kind: TrackedTaskKind, operation: () => Promise<T>): Promise<T> {
		let settle = () => {};
		const settled = new Promise<void>((resolve) => {
			settle = resolve;
		});
		this.activeTasks.set(settled, kind);
		try {
			return await operation();
		} finally {
			this.activeTasks.delete(settled);
			settle();
		}
	}

	private async waitForTasks(kind?: TrackedTaskKind): Promise<void> {
		while (true) {
			const tasks = [...this.activeTasks].flatMap(([task, taskKind]) =>
				kind === undefined || kind === taskKind ? [task] : [],
			);
			if (tasks.length === 0) return;
			await Promise.all(tasks);
		}
	}

	private async resolveToolContext(): Promise<TContext> {
		if (typeof this.toolContext === "function") {
			return await (this.toolContext as () => TContext | Promise<TContext>)();
		}
		return this.toolContext as TContext;
	}

	private bindToolContext(tool: TTool, context: TContext): AgentTool {
		return {
			...tool,
			execute: (toolCallId, params, signal, onUpdate) => tool.execute(toolCallId, params, signal, onUpdate, context),
		};
	}

	private async createTurnState(): Promise<AgentHarnessTurnState<TContext, TSkill, TPromptTemplate, TTool>> {
		this.assertNotShutDown();
		// PI-015: when no context controller is present, keep the EXACT native
		// await order (Session.buildContext() first) so the default path stays
		// byte-compatible. The controller path never forces buildContext().
		const nativeContext = this.contextController === undefined ? await this.session.buildContext() : undefined;
		const resources = this.getResources();
		const sessionMetadata = await this.session.getMetadata();
		const toolContext = await this.resolveToolContext();
		const tools = [...this.tools.values()];
		const activeTools = this.activeToolNames
			.map((name) => this.tools.get(name))
			.filter((tool): tool is TTool => tool !== undefined);
		let systemPrompt = "You are a helpful assistant.";
		let messages: AgentMessage[];
		if (this.contextController !== undefined) {
			// PI-015: Provider Context Controller decides systemPrompt/messages
			// completely BEFORE provider conversion; Session.buildContext() is
			// NOT forced on this path.
			const controlled = await this.contextController({
				session: this.session,
				model: this.model,
				thinkingLevel: this.thinkingLevel,
				activeTools,
				resources,
			});
			systemPrompt = controlled.systemPrompt;
			messages = controlled.messages;
			if (controlled.activeToolNames !== undefined) {
				this.validateUniqueNames(controlled.activeToolNames, "Duplicate active tool name(s)");
				this.validateToolNames(controlled.activeToolNames);
				this.activeToolNames = [...controlled.activeToolNames];
			}
		} else {
			// Default native path: Session-derived context, byte-compatible.
			messages = nativeContext!.messages;
			if (typeof this.systemPrompt === "string") {
				systemPrompt = this.systemPrompt;
			} else if (this.systemPrompt) {
				systemPrompt = await this.systemPrompt({
					session: this.session,
					model: this.model,
					thinkingLevel: this.thinkingLevel,
					activeTools,
					resources,
				});
			}
		}
		return {
			messages,
			resources,
			toolContext,
			streamOptions: cloneStreamOptions(this.streamOptions),
			sessionId: sessionMetadata.id,
			systemPrompt,
			model: this.model,
			thinkingLevel: this.thinkingLevel,
			tools,
			activeTools,
		};
	}

	private createContext(
		turnState: AgentHarnessTurnState<TContext, TSkill, TPromptTemplate, TTool>,
		systemPrompt?: string,
	): AgentContext {
		return {
			systemPrompt: systemPrompt ?? turnState.systemPrompt,
			messages: turnState.messages.slice(),
			tools: turnState.activeTools.map((tool) => this.bindToolContext(tool, turnState.toolContext)),
		};
	}

	private createStreamFn(
		getTurnState: () => AgentHarnessTurnState<TContext, TSkill, TPromptTemplate, TTool>,
	): StreamFn {
		return async (model, context, streamOptions) => {
			const turnState = getTurnState();
			const snapshotOptions: AgentHarnessStreamOptions = { ...turnState.streamOptions };
			const requestOptions = await this.emitBeforeProviderRequest(model, turnState.sessionId, snapshotOptions);
			return this.models.streamSimple(model, context, {
				cacheRetention: requestOptions.cacheRetention,
				headers: requestOptions.headers,
				maxRetries: requestOptions.maxRetries,
				maxRetryDelayMs: requestOptions.maxRetryDelayMs,
				metadata: requestOptions.metadata,
				onPayload: async (payload) => await this.emitBeforeProviderPayload(model, payload),
				onResponse: async (response) => {
					const headers = { ...(response.headers as Record<string, string>) };
					await this.emitOwn(
						{ type: "after_provider_response", status: response.status, headers },
						streamOptions?.signal,
					);
				},
				reasoning: streamOptions?.reasoning,
				signal: streamOptions?.signal,
				sessionId: turnState.sessionId,
				timeoutMs: requestOptions.timeoutMs,
				transport: requestOptions.transport,
			});
		};
	}

	private async drainQueuedMessages(queue: AgentMessage[], mode: QueueMode): Promise<AgentMessage[]> {
		const messages = mode === "all" ? queue.splice(0) : queue.splice(0, 1);
		if (messages.length === 0) return messages;
		try {
			await this.emitQueueUpdate();
			return messages;
		} catch (error) {
			queue.unshift(...messages);
			throw normalizeHookError(error);
		}
	}

	private createLoopConfig(
		getTurnState: () => AgentHarnessTurnState<TContext, TSkill, TPromptTemplate, TTool>,
		setTurnState: (turnState: AgentHarnessTurnState<TContext, TSkill, TPromptTemplate, TTool>) => void,
	): AgentLoopConfig {
		const turnState = getTurnState();
		return {
			model: turnState.model,
			reasoning: turnState.thinkingLevel === "off" ? undefined : turnState.thinkingLevel,
			convertToLlm,
			transformContext: async (messages) => {
				const result = await this.emitHook({ type: "context", messages: [...messages] });
				return result?.messages ?? messages;
			},
			beforeToolCall: async ({ toolCall, args }) => {
				const result = await this.emitHook({
					type: "tool_call",
					toolCallId: toolCall.id,
					toolName: toolCall.name,
					input: args as Record<string, unknown>,
				});
				return result ? { block: result.block, reason: result.reason } : undefined;
			},
			afterToolCall: async ({ toolCall, args, result, isError }) => {
				const patch = await this.emitHook({
					type: "tool_result",
					toolCallId: toolCall.id,
					toolName: toolCall.name,
					input: args as Record<string, unknown>,
					content: result.content,
					details: result.details,
					isError,
					usage: result.usage,
				});
				// PI-017: tool execution committed with its result.
				await this.emitOwn(
					{
						type: "tool_execution_committed",
						toolCallId: toolCall.id,
						toolName: toolCall.name,
						input: args as Record<string, unknown>,
						isError: patch ? (patch.isError ?? isError) : isError,
					},
					this.activeAbortController?.signal,
				);
				return patch
					? {
							content: patch.content,
							details: patch.details,
							isError: patch.isError,
							usage: patch.usage,
							terminate: patch.terminate,
						}
					: undefined;
			},
			prepareNextTurn: async () => {
				await this.flushPendingSessionWrites();
				const nextTurnState = await this.createTurnState();
				setTurnState(nextTurnState);
				return {
					context: this.createContext(nextTurnState),
					model: nextTurnState.model,
					thinkingLevel: nextTurnState.thinkingLevel,
				};
			},
			getSteeringMessages: async () => this.drainQueuedMessages(this.steerQueue, this.steeringQueueMode),
			getFollowUpMessages: async () => this.drainQueuedMessages(this.followUpQueue, this.followUpQueueMode),
		};
	}

	private validateUniqueNames(names: string[], message: string): void {
		const duplicates = findDuplicateNames(names);
		if (duplicates.length > 0)
			throw new AgentHarnessError("invalid_argument", `${message}: ${duplicates.join(", ")}`);
	}

	private validateToolNames(toolNames: string[], tools: Map<string, TTool> = this.tools): void {
		this.validateUniqueNames(toolNames, "Duplicate active tool name(s)");
		const missing = toolNames.filter((name) => !tools.has(name));
		if (missing.length > 0) throw new AgentHarnessError("invalid_argument", `Unknown tool(s): ${missing.join(", ")}`);
	}

	private async flushPendingSessionWrites(): Promise<void> {
		while (this.pendingSessionWrites.length > 0) {
			const write = this.pendingSessionWrites[0]!;
			if (write.type === "message") {
				await this.appendAndCommitMessage(write.message);
			} else if (write.type === "model_change") {
				await this.session.appendModelChange(write.provider, write.modelId);
			} else if (write.type === "thinking_level_change") {
				await this.session.appendThinkingLevelChange(write.thinkingLevel);
			} else if (write.type === "active_tools_change") {
				await this.session.appendActiveToolsChange(write.activeToolNames);
			} else if (write.type === "custom") {
				await this.session.appendCustomEntry(write.customType, write.data);
			} else if (write.type === "custom_message") {
				await this.session.appendCustomMessageEntry(write.customType, write.content, write.display, write.details);
			} else if (write.type === "label") {
				await this.session.appendLabel(write.targetId, write.label);
			} else if (write.type === "session_info") {
				await this.session.appendSessionName(write.name ?? "");
			} else if (write.type === "leaf") {
				await this.session.moveTo(write.targetId);
			}
			this.pendingSessionWrites.shift();
		}
	}

	/**
	 * Shared generic commit primitive (iris_agent#40): every supported durable
	 * message append path funnels through here so each committed message yields
	 * exactly one SessionCommitReceipt and exactly one canonical lifecycle
	 * event (message_finalized). Durable append happens first; if it fails the
	 * method throws and no receipt/event is produced (no phantom receipts).
	 * afterAppend runs between the durable append and the receipt publication
	 * so callers can preserve their existing observer-event ordering.
	 *
	 * Crash consistency (iris_agent#40 / Feature 2): when the session storage
	 * implements the commit journal, the durable append and the pending receipt
	 * are persisted atomically; the receipt is acknowledged only after
	 * `message_finalized` has been published. A crash between the durable
	 * append and the publication leaves the receipt pending, and
	 * {@link AgentHarness.recoverPendingCommitReceipts} replays it on restart.
	 * Replay never re-appends and never re-runs afterAppend, so it cannot
	 * duplicate semantic commits; receipt identity (sessionId + entryId +
	 * contentHash) stays stable across replay.
	 */
	private async appendAndCommitMessage(
		message: AgentMessage,
		signal?: AbortSignal,
		afterAppend?: () => Promise<void>,
	): Promise<void> {
		const sessionMetadata = await this.session.getMetadata();
		const contentHash = await computeMessageContentHash(message);
		const { entryId, receipt } = await this.session.appendMessageWithCommitReceipt(message, (committedEntryId) => ({
			sessionId: sessionMetadata.id,
			entryId: committedEntryId,
			contentHash,
			committedAt: new Date().toISOString(),
		}));
		await afterAppend?.();
		await this.emitOwn(
			{
				type: "message_finalized",
				entryId,
				role: message.role,
				contentHash,
				message,
				receipt,
			},
			signal,
		);
		// Published: the journal entry can be replayed no more. A crash between
		// the durable append and this ack leaves the receipt pending so restart
		// recovery replays it (consumer dedupes on stable receipt identity).
		await this.session.ackCommitReceipt(entryId);
	}

	/**
	 * Crash recovery (iris_agent#40 / Feature 2): replays `message_finalized`
	 * for every durably recorded receipt that was not yet acknowledged (i.e.
	 * published) before a crash. Replay emits the same canonical event with the
	 * same stable receipt identity (sessionId + entryId + contentHash) and
	 * never re-appends the message, so it cannot duplicate a semantic commit.
	 * Pending receipts are replayed in commit order. Returns the number of
	 * receipts replayed. Call after subscribing, before running new turns, when
	 * reopening a session after a crash.
	 *
	 * iris_agent#50 integrity validation: BEFORE emitting or ACKing, every
	 * pending receipt is verified against the durable Session:
	 * - receipt.sessionId MUST equal the reopened Session id;
	 * - the entry MUST exist, be a message, and carry a valid role;
	 * - the content hash is RECOMPUTED from the entry's message via the same
	 *   canonical hashing implementation as the normal append path and MUST
	 *   match the receipt.
	 * Any mismatch fails closed with a typed reason and PERMANENTLY
	 * QUARANTINES the row (never emitted, never acked, visible via
	 * {@link Session.readQuarantinedCommitReceipts}); recovery continues with
	 * the remaining receipts in commit order, so valid receipts emit exactly
	 * once and invalid receipts emit zero times (retry/restart idempotent).
	 */
	async recoverPendingCommitReceipts(): Promise<number> {
		const sessionMetadata = await this.session.getMetadata();
		const pending = await this.session.readPendingCommitReceipts();
		let replayed = 0;
		for (const receipt of pending) {
			if (receipt.sessionId !== sessionMetadata.id) {
				await this.session.quarantineCommitReceipt(
					receipt.entryId,
					`session_id_mismatch: receipt sessionId ${JSON.stringify(receipt.sessionId)} != reopened session ${JSON.stringify(sessionMetadata.id)}`,
				);
				continue;
			}
			const entry = await this.session.getEntry(receipt.entryId);
			if (!entry || entry.type !== "message") {
				await this.session.quarantineCommitReceipt(
					receipt.entryId,
					`missing_message_entry: no recoverable message entry (got ${entry?.type ?? "missing"})`,
				);
				continue;
			}
			const role = entry.message.role;
			// iris_agent#50: the role must be a known AgentMessage role (base
			// Message roles + the harness custom message roles). A corrupted or
			// tampered entry whose role does not match any known shape fails
			// closed instead of being replayed into the lifecycle.
			if (!VALID_MESSAGE_ROLES.has(role)) {
				await this.session.quarantineCommitReceipt(
					receipt.entryId,
					`invalid_role: message role ${JSON.stringify(role)} is not a known AgentMessage role`,
				);
				continue;
			}
			const recomputed = await computeMessageContentHash(entry.message);
			if (recomputed !== receipt.contentHash) {
				await this.session.quarantineCommitReceipt(
					receipt.entryId,
					`content_hash_mismatch: entry message hashes to ${recomputed} but the receipt records ${receipt.contentHash}`,
				);
				continue;
			}
			await this.emitOwn({
				type: "message_finalized",
				entryId: receipt.entryId,
				role: entry.message.role,
				contentHash: receipt.contentHash,
				message: entry.message,
				receipt,
			});
			await this.session.ackCommitReceipt(receipt.entryId);
			replayed++;
		}
		return replayed;
	}

	private async handleAgentEvent(event: AgentEvent, signal?: AbortSignal): Promise<void> {
		if (event.type === "message_end") {
			// PI-016/PI-017: the agent-loop append path shares the same generic
			// commit primitive as direct appends and pending-writes flush, so
			// receipts/events are emitted exactly once per durable append.
			// Order kept as before: durable append, then the message_end
			// observer event, then the message_finalized commit event.
			await this.appendAndCommitMessage(event.message, signal, async () => {
				await this.emitAny(event, signal);
			});
			return;
		}
		if (event.type === "turn_end") {
			let eventError: unknown;
			try {
				await this.emitAny(event, signal);
			} catch (error) {
				eventError = error;
			}
			const hadPendingMutations = this.pendingSessionWrites.length > 0;
			await this.flushPendingSessionWrites();
			if (eventError) throw eventError;
			await this.emitOwn({ type: "save_point", hadPendingMutations });
			// PI-017: a full turn (assistant response + tool results) committed.
			await this.emitOwn(
				{ type: "turn_committed", toolResultCount: event.toolResults.length, hadPendingMutations },
				signal,
			);
			return;
		}
		if (event.type === "agent_end") {
			await this.flushPendingSessionWrites();
			this.phase = "idle";
			await this.emitAny(event, signal);
			await this.emitOwn({ type: "settled", nextTurnCount: this.nextTurnQueue.length }, signal);
			return;
		}
		await this.emitAny(event, signal);
	}

	private async emitRunFailure(
		model: Model<any>,
		error: unknown,
		aborted: boolean,
		signal: AbortSignal,
	): Promise<AgentMessage[]> {
		const failureMessage = createFailureMessage(model, error, aborted);
		await this.handleAgentEvent({ type: "message_start", message: failureMessage }, signal);
		await this.handleAgentEvent({ type: "message_end", message: failureMessage }, signal);
		await this.handleAgentEvent({ type: "turn_end", message: failureMessage, toolResults: [] }, signal);
		await this.handleAgentEvent({ type: "agent_end", messages: [failureMessage] }, signal);
		return [failureMessage];
	}

	private async executeTurn(
		turnState: AgentHarnessTurnState<TContext, TSkill, TPromptTemplate, TTool>,
		text: string,
		signal: AbortSignal,
		options?: { images?: ImageContent[] },
	): Promise<AssistantMessage> {
		this.assertNotShutDown();
		let activeTurnState = turnState;
		let messages: AgentMessage[] = [createUserMessage(text, options?.images)];
		if (this.nextTurnQueue.length > 0) {
			const queuedMessages = this.nextTurnQueue.splice(0);
			try {
				await this.emitQueueUpdate();
			} catch (error) {
				this.nextTurnQueue.unshift(...queuedMessages);
				throw normalizeHookError(error);
			}
			messages = [...queuedMessages, messages[0]!];
		}
		const beforeResult = await this.emitHook({
			type: "before_agent_start",
			prompt: text,
			images: options?.images,
			systemPrompt: turnState.systemPrompt,
			resources: turnState.resources,
		});
		this.assertNotShutDown();
		if (beforeResult?.messages) messages = [...messages, ...beforeResult.messages];

		const getTurnState = () => activeTurnState;
		const setTurnState = (nextTurnState: AgentHarnessTurnState<TContext, TSkill, TPromptTemplate, TTool>) => {
			activeTurnState = nextTurnState;
		};
		const runResultPromise = (async () => {
			try {
				return await runAgentLoop(
					messages,
					this.createContext(turnState, beforeResult?.systemPrompt),
					this.createLoopConfig(getTurnState, setTurnState),
					(event) => this.handleAgentEvent(event, signal),
					signal,
					this.createStreamFn(getTurnState),
				);
			} catch (error) {
				try {
					return await this.emitRunFailure(activeTurnState.model, error, signal.aborted, signal);
				} catch (failureError) {
					const cause = new AggregateError(
						[toError(error), toError(failureError)],
						"Agent run failed and failure reporting failed",
					);
					throw new AgentHarnessError("unknown", cause.message, cause);
				}
			}
		})();
		try {
			const newMessages = await runResultPromise;
			for (let i = newMessages.length - 1; i >= 0; i--) {
				const message = newMessages[i]!;
				if (message.role === "assistant") {
					return message;
				}
			}
			throw new AgentHarnessError("invalid_state", "AgentHarness prompt completed without an assistant message");
		} finally {
			await this.flushPendingSessionWrites();
		}
	}

	async prompt(text: string, options?: { images?: ImageContent[] }): Promise<AssistantMessage> {
		this.assertNotShutDown();
		if (this.phase !== "idle") throw new AgentHarnessError("busy", "AgentHarness is busy");
		this.phase = "turn";
		const operation = this.startOperation();
		try {
			const turnState = await this.createTurnState();
			return await this.executeTurn(turnState, text, operation.signal, options);
		} catch (error) {
			this.phase = "idle";
			throw normalizeHarnessError(error, "unknown");
		} finally {
			operation.finish();
		}
	}

	async skill(name: string, additionalInstructions?: string): Promise<AssistantMessage> {
		this.assertNotShutDown();
		if (this.phase !== "idle") throw new AgentHarnessError("busy", "AgentHarness is busy");
		this.phase = "turn";
		const operation = this.startOperation();
		try {
			const turnState = await this.createTurnState();
			const skill = (turnState.resources.skills ?? []).find((candidate) => candidate.name === name);
			if (!skill) throw new AgentHarnessError("invalid_argument", `Unknown skill: ${name}`);
			return await this.executeTurn(
				turnState,
				formatSkillInvocation(skill, additionalInstructions),
				operation.signal,
			);
		} catch (error) {
			this.phase = "idle";
			throw normalizeHarnessError(error, "unknown");
		} finally {
			operation.finish();
		}
	}

	async promptFromTemplate(name: string, args: string[] = []): Promise<AssistantMessage> {
		this.assertNotShutDown();
		if (this.phase !== "idle") throw new AgentHarnessError("busy", "AgentHarness is busy");
		this.phase = "turn";
		const operation = this.startOperation();
		try {
			const turnState = await this.createTurnState();
			const template = (turnState.resources.promptTemplates ?? []).find((candidate) => candidate.name === name);
			if (!template) throw new AgentHarnessError("invalid_argument", `Unknown prompt template: ${name}`);
			return await this.executeTurn(turnState, formatPromptTemplateInvocation(template, args), operation.signal);
		} catch (error) {
			this.phase = "idle";
			throw normalizeHarnessError(error, "unknown");
		} finally {
			operation.finish();
		}
	}

	async steer(text: string, options?: { images?: ImageContent[] }): Promise<void> {
		this.assertNotShutDown();
		if (this.phase === "idle") throw new AgentHarnessError("invalid_state", "Cannot steer while idle");
		this.steerQueue.push(createUserMessage(text, options?.images));
		await this.emitQueueUpdate();
	}

	async followUp(text: string, options?: { images?: ImageContent[] }): Promise<void> {
		this.assertNotShutDown();
		if (this.phase === "idle") throw new AgentHarnessError("invalid_state", "Cannot follow up while idle");
		this.followUpQueue.push(createUserMessage(text, options?.images));
		await this.emitQueueUpdate();
	}

	async nextTurn(text: string, options?: { images?: ImageContent[] }): Promise<void> {
		this.assertNotShutDown();
		this.nextTurnQueue.push(createUserMessage(text, options?.images));
		await this.emitQueueUpdate();
	}

	async appendMessage(message: AgentMessage): Promise<void> {
		this.assertNotShutDown();
		return this.track("mutation", async () => {
			try {
				if (this.phase === "idle") {
					// iris_agent#40: direct appends use the shared commit
					// primitive so they emit exactly one receipt/event too.
					await this.appendAndCommitMessage(message);
				} else {
					this.pendingSessionWrites.push({ type: "message", message });
				}
			} catch (error) {
				throw normalizeHarnessError(error, "session");
			}
		});
	}

	async compact(customInstructions?: string): Promise<CompactResult> {
		this.assertNotShutDown();
		if (this.phase !== "idle") throw new AgentHarnessError("busy", "compact() requires idle harness");
		this.phase = "compaction";
		const operation = this.startOperation();
		try {
			const model = this.model;
			if (!model) throw new AgentHarnessError("invalid_state", "No model set for compaction");
			const branchEntries = await this.session.getBranch();
			const preparationResult = prepareCompaction(branchEntries, DEFAULT_COMPACTION_SETTINGS);
			if (!preparationResult.ok) throw preparationResult.error;
			const preparation = preparationResult.value;
			if (!preparation) throw new AgentHarnessError("compaction", "Nothing to compact");
			const hookResult = await this.emitHook({
				type: "session_before_compact",
				preparation,
				branchEntries,
				customInstructions,
				signal: operation.signal,
			});
			if (hookResult?.cancel) throw new AgentHarnessError("compaction", "Compaction cancelled");
			const provided = hookResult?.compaction;
			const compactResult = provided
				? { ok: true as const, value: provided }
				: await compact(
						preparation,
						this.models,
						model,
						customInstructions,
						operation.signal,
						this.thinkingLevel,
						this.retry,
						this.retryCallbacks("compaction"),
					);
			if (!compactResult.ok) throw compactResult.error;
			const result = compactResult.value;
			this.assertNotShutDown();
			const entryId = await this.session.appendCompaction(
				result.summary,
				result.firstKeptEntryId,
				result.tokensBefore,
				result.details,
				provided !== undefined,
				result.usage,
				result.retainedTail,
			);
			const entry = await this.session.getEntry(entryId);
			if (entry?.type === "compaction") {
				await this.emitOwn({ type: "session_compact", compactionEntry: entry, fromHook: provided !== undefined });
			}
			return result;
		} catch (error) {
			throw normalizeHarnessError(error, "compaction");
		} finally {
			this.phase = "idle";
			operation.finish();
		}
	}

	async navigateTree(
		targetId: string,
		options?: { summarize?: boolean; customInstructions?: string; replaceInstructions?: boolean; label?: string },
	): Promise<NavigateTreeResult> {
		this.assertNotShutDown();
		if (this.phase !== "idle") throw new AgentHarnessError("busy", "navigateTree() requires idle harness");
		this.phase = "branch_summary";
		const operation = this.startOperation();
		try {
			const oldLeafId = await this.session.getLeafId();
			if (oldLeafId === targetId) return { cancelled: false };
			const targetEntry = await this.session.getEntry(targetId);
			if (!targetEntry) throw new AgentHarnessError("invalid_argument", `Entry ${targetId} not found`);
			const { entries, commonAncestorId } = await collectEntriesForBranchSummary(this.session, oldLeafId, targetId);
			const preparation = {
				targetId,
				oldLeafId,
				commonAncestorId,
				entriesToSummarize: entries,
				userWantsSummary: options?.summarize ?? false,
				customInstructions: options?.customInstructions,
				replaceInstructions: options?.replaceInstructions,
				label: options?.label,
			};
			const hookResult = await this.emitHook({
				type: "session_before_tree",
				preparation,
				signal: operation.signal,
			});
			if (hookResult?.cancel) return { cancelled: true };
			let summaryEntry: NavigateTreeResult["summaryEntry"];
			let summaryText: string | undefined = hookResult?.summary?.summary;
			let summaryDetails: unknown = hookResult?.summary?.details;
			let summaryUsage = hookResult?.summary?.usage;
			if (!summaryText && options?.summarize && entries.length > 0) {
				const model = this.model;
				if (!model) throw new AgentHarnessError("invalid_state", "No model set for branch summary");
				const branchSummary = await generateBranchSummary(entries, {
					models: this.models,
					model,
					signal: operation.signal,
					customInstructions: hookResult?.customInstructions ?? options?.customInstructions,
					replaceInstructions: hookResult?.replaceInstructions ?? options?.replaceInstructions,
					retry: this.retry,
					callbacks: this.retryCallbacks("branch_summary"),
				});
				if (!branchSummary.ok) {
					if (branchSummary.error.code === "aborted") return { cancelled: true };
					throw new AgentHarnessError("branch_summary", branchSummary.error.message, branchSummary.error);
				}
				summaryText = branchSummary.value.summary;
				summaryUsage = branchSummary.value.usage;
				summaryDetails = {
					readFiles: branchSummary.value.readFiles,
					modifiedFiles: branchSummary.value.modifiedFiles,
				};
			}
			let editorText: string | undefined;
			let newLeafId: string | null;
			if (targetEntry.type === "message" && targetEntry.message.role === "user") {
				newLeafId = targetEntry.parentId;
				editorText = contentText(targetEntry.message.content, "");
			} else if (targetEntry.type === "custom_message") {
				newLeafId = targetEntry.parentId;
				editorText = contentText(targetEntry.content, "");
			} else {
				newLeafId = targetId;
			}
			this.assertNotShutDown();
			const summaryId = await this.session.moveTo(
				newLeafId,
				summaryText
					? {
							summary: summaryText,
							details: summaryDetails,
							usage: summaryUsage,
							fromHook: hookResult?.summary !== undefined,
						}
					: undefined,
			);
			if (summaryId) {
				const entry = await this.session.getEntry(summaryId);
				if (entry?.type === "branch_summary") summaryEntry = entry;
			}
			await this.emitOwn({
				type: "session_tree",
				newLeafId: await this.session.getLeafId(),
				oldLeafId,
				summaryEntry,
				fromHook: hookResult?.summary !== undefined,
			});
			return { cancelled: false, editorText, summaryEntry };
		} catch (error) {
			throw normalizeHarnessError(error, "branch_summary");
		} finally {
			this.phase = "idle";
			operation.finish();
		}
	}

	getModel(): Model<any> {
		return this.model;
	}

	async setModel(model: Model<any>): Promise<void> {
		this.assertNotShutDown();
		return this.track("mutation", async () => {
			try {
				const previousModel = this.model;
				if (this.phase === "idle") {
					await this.session.appendModelChange(model.provider, model.id);
				} else {
					this.pendingSessionWrites.push({ type: "model_change", provider: model.provider, modelId: model.id });
				}
				this.model = model;
				await this.emitOwn({ type: "model_update", model, previousModel, source: "set" });
			} catch (error) {
				throw normalizeHarnessError(error, "session");
			}
		});
	}

	getThinkingLevel(): ThinkingLevel {
		return this.thinkingLevel;
	}

	async setThinkingLevel(level: ThinkingLevel): Promise<void> {
		this.assertNotShutDown();
		return this.track("mutation", async () => {
			try {
				const previousLevel = this.thinkingLevel;
				if (this.phase === "idle") {
					await this.session.appendThinkingLevelChange(level);
				} else {
					this.pendingSessionWrites.push({ type: "thinking_level_change", thinkingLevel: level });
				}
				this.thinkingLevel = level;
				await this.emitOwn({ type: "thinking_level_update", level, previousLevel });
			} catch (error) {
				throw normalizeHarnessError(error, "session");
			}
		});
	}

	getTools(): TTool[] {
		return [...this.tools.values()];
	}

	async setTools(tools: TTool[], activeToolNames?: string[]): Promise<void> {
		this.assertNotShutDown();
		return this.track("mutation", () => this.applyTools(tools, activeToolNames));
	}

	private async applyTools(tools: TTool[], activeToolNames?: string[]): Promise<void> {
		try {
			this.validateUniqueNames(
				tools.map((tool) => tool.name),
				"Duplicate tool name(s)",
			);
			const nextTools = new Map(tools.map((tool) => [tool.name, tool]));
			const nextActiveToolNames = activeToolNames ? [...activeToolNames] : this.activeToolNames;
			this.validateToolNames(nextActiveToolNames, nextTools);
			const previousToolNames = [...this.tools.keys()];
			const previousActiveToolNames = [...this.activeToolNames];
			if (this.phase === "idle") {
				await this.session.appendActiveToolsChange(nextActiveToolNames);
			} else {
				this.pendingSessionWrites.push({ type: "active_tools_change", activeToolNames: [...nextActiveToolNames] });
			}
			this.tools = nextTools;
			this.activeToolNames = [...nextActiveToolNames];
			await this.emitOwn({
				type: "tools_update",
				toolNames: [...this.tools.keys()],
				previousToolNames,
				activeToolNames: [...this.activeToolNames],
				previousActiveToolNames,
				source: "set",
			});
		} catch (error) {
			throw normalizeHarnessError(error, "invalid_argument");
		}
	}

	getActiveTools(): TTool[] {
		return this.activeToolNames.map((name) => this.tools.get(name)!);
	}

	async setActiveTools(toolNames: string[]): Promise<void> {
		this.assertNotShutDown();
		return this.track("mutation", () => this.applyActiveTools(toolNames));
	}

	private async applyActiveTools(toolNames: string[]): Promise<void> {
		try {
			this.validateToolNames(toolNames);
			const previousToolNames = [...this.tools.keys()];
			const previousActiveToolNames = [...this.activeToolNames];
			if (this.phase === "idle") {
				await this.session.appendActiveToolsChange(toolNames);
			} else {
				this.pendingSessionWrites.push({ type: "active_tools_change", activeToolNames: [...toolNames] });
			}
			this.activeToolNames = [...toolNames];
			await this.emitOwn({
				type: "tools_update",
				toolNames: [...this.tools.keys()],
				previousToolNames,
				activeToolNames: [...this.activeToolNames],
				previousActiveToolNames,
				source: "set",
			});
		} catch (error) {
			throw normalizeHarnessError(error, "invalid_argument");
		}
	}

	getSteeringMode(): QueueMode {
		return this.steeringQueueMode;
	}

	async setSteeringMode(mode: QueueMode): Promise<void> {
		this.assertNotShutDown();
		this.steeringQueueMode = mode;
	}

	getFollowUpMode(): QueueMode {
		return this.followUpQueueMode;
	}

	async setFollowUpMode(mode: QueueMode): Promise<void> {
		this.assertNotShutDown();
		this.followUpQueueMode = mode;
	}

	getResources(): AgentHarnessResources<TSkill, TPromptTemplate> {
		return {
			skills: this.resources.skills?.slice(),
			promptTemplates: this.resources.promptTemplates?.slice(),
		};
	}

	async setResources(resources: AgentHarnessResources<TSkill, TPromptTemplate>): Promise<void> {
		this.assertNotShutDown();
		const previousResources = this.getResources();
		this.resources = {
			skills: resources.skills?.slice(),
			promptTemplates: resources.promptTemplates?.slice(),
		};
		await this.emitOwn({ type: "resources_update", resources: this.getResources(), previousResources });
	}

	getStreamOptions(): AgentHarnessStreamOptions {
		return cloneStreamOptions(this.streamOptions);
	}

	async setStreamOptions(streamOptions: AgentHarnessStreamOptions): Promise<void> {
		this.assertNotShutDown();
		this.streamOptions = cloneStreamOptions(streamOptions);
	}

	/** Permanently stop this harness instance without deleting its durable session. */
	requestShutdown(): void {
		if (this.isShutdown) return;
		this.isShutdown = true;
		this.pendingSessionWrites = [];
		this.steerQueue = [];
		this.followUpQueue = [];
		this.nextTurnQueue = [];
		this.activeAbortController?.abort();
		this.shutdownPromise = this.waitForTasks();
	}

	/** Waits for work active when shutdown was requested to settle. */
	waitForShutdown(): Promise<void> {
		if (!this.shutdownPromise) {
			return Promise.reject(new AgentHarnessError("invalid_state", "Shutdown has not been requested"));
		}
		return this.shutdownPromise;
	}

	async abort(): Promise<AbortResult> {
		this.assertNotShutDown();
		const clearedSteer = [...this.steerQueue];
		const clearedFollowUp = [...this.followUpQueue];
		this.steerQueue = [];
		this.followUpQueue = [];
		this.activeAbortController?.abort();
		const errors: Error[] = [];
		try {
			await this.emitQueueUpdate();
		} catch (error) {
			errors.push(toError(error));
		}
		try {
			await this.waitForIdle();
		} catch (error) {
			errors.push(toError(error));
		}
		try {
			await this.emitOwn({ type: "abort", clearedSteer, clearedFollowUp });
		} catch (error) {
			errors.push(toError(error));
		}
		if (errors.length > 0) {
			const cause = errors.length === 1 ? errors[0]! : new AggregateError(errors, "Abort completed with errors");
			throw normalizeHarnessError(cause, "hook");
		}
		return { clearedSteer, clearedFollowUp };
	}

	async waitForIdle(): Promise<void> {
		await this.waitForTasks("operation");
	}

	subscribe(
		listener: (event: AgentHarnessEvent<TSkill, TPromptTemplate>, signal?: AbortSignal) => Promise<void> | void,
	): () => void {
		this.assertNotShutDown();
		let handlers = this.handlers.get(SUBSCRIBER_EVENT_TYPE);
		if (!handlers) {
			handlers = new Set();
			this.handlers.set(SUBSCRIBER_EVENT_TYPE, handlers);
		}
		handlers.add(listener as AgentHarnessHandler);
		return () => handlers!.delete(listener as AgentHarnessHandler);
	}

	on<TType extends keyof AgentHarnessEventResultMap>(
		type: TType,
		handler: (
			event: Extract<AgentHarnessOwnEvent, { type: TType }>,
		) => Promise<AgentHarnessEventResultMap[TType]> | AgentHarnessEventResultMap[TType],
	): () => void {
		this.assertNotShutDown();
		let handlers = this.handlers.get(type);
		if (!handlers) {
			handlers = new Set();
			this.handlers.set(type, handlers);
		}
		handlers.add(handler as AgentHarnessHandler);
		return () => handlers!.delete(handler as AgentHarnessHandler);
	}
}
