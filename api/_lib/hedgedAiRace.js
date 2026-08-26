import { callNvidiaChat, parseJsonLoose } from './nvidia.js';

const DEFAULT_MAX_ATTEMPTS = 3;
const DEPRECATED_HOSTED_MODELS = new Set([
  'qwen/qwen3.5-122b-a10b',
  'qwen/qwen3.5-397b-a17b',
]);
const CURRENT_RACE_PRIORITY = [
  'google/gemma-4-31b-it',
  'openai/gpt-oss-120b',
  'nvidia/nemotron-3-super-120b-a12b',
  'openai/gpt-oss-20b',
  'nvidia/llama-3.3-nemotron-super-49b-v1',
  'meta/llama-3.1-8b-instruct',
];

function delay(ms, signal) {
  if (!ms) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(resolve, ms);
    if (signal) {
      signal.addEventListener('abort', () => {
        clearTimeout(timeout);
        reject(signal.reason || new DOMException('Aborted', 'AbortError'));
      }, { once: true });
    }
  });
}

function serializeError(error) {
  return {
    name: error?.name,
    message: error?.message || String(error),
    statusCode: error?.statusCode,
    detail: error?.detail,
    model: error?.model,
  };
}

function envFlag(value, fallback = false) {
  if (value === undefined) return fallback;
  return !['0', 'false', 'no', 'off'].includes(String(value).toLowerCase());
}

function rankModel(model) {
  const index = CURRENT_RACE_PRIORITY.indexOf(model);
  return index >= 0 ? index : CURRENT_RACE_PRIORITY.length + 1;
}

function selectRaceModels(models, maxAttempts) {
  const unique = [...new Set((models || []).map((model) => String(model || '').trim()).filter(Boolean))]
    .filter((model) => !DEPRECATED_HOSTED_MODELS.has(model));

  const ordered = envFlag(process.env.NVIDIA_MODEL_STRICT_ENV, false)
    ? unique
    : unique
      .map((model, originalIndex) => ({ model, originalIndex, rank: rankModel(model) }))
      .sort((a, b) => a.rank - b.rank || a.originalIndex - b.originalIndex)
      .map((entry) => entry.model);

  return ordered.slice(0, maxAttempts);
}

export async function hedgedNvidiaRace({
  models,
  delaysMs = [0, 700, 1800],
  messages,
  apiKey,
  timeoutMs = 14000,
  temperature = 0.25,
  topP,
  seed,
  maxTokens = 1800,
  responseFormat = { type: 'json_object' },
  validate,
  maxAttempts = DEFAULT_MAX_ATTEMPTS,
}) {
  const selectedModels = selectRaceModels(models, Math.max(1, Number(maxAttempts) || DEFAULT_MAX_ATTEMPTS));
  if (selectedModels.length === 0) throw new Error('At least one current NVIDIA hosted model is required');

  const attempts = [];
  const controllers = selectedModels.map(() => new AbortController());
  const overallController = new AbortController();
  let settled = false;
  let pending = selectedModels.length;
  let timeoutId;

  const abortOthers = (winnerIndex) => {
    settled = true;
    controllers.forEach((controller, index) => {
      if (index !== winnerIndex && !controller.signal.aborted) controller.abort(new Error('Hedged race resolved'));
    });
    if (!overallController.signal.aborted) overallController.abort(new Error('Hedged race resolved'));
    if (timeoutId) clearTimeout(timeoutId);
  };

  return new Promise((resolve, reject) => {
    timeoutId = setTimeout(() => {
      if (settled) return;
      settled = true;
      controllers.forEach((controller) => {
        if (!controller.signal.aborted) controller.abort(new Error('AI Generation Timeout'));
      });
      const error = new Error('AI Generation Timeout');
      error.statusCode = 504;
      error.attempts = attempts;
      reject(error);
    }, timeoutMs);

    const maybeReject = () => {
      if (settled || pending > 0) return;
      const error = new Error('All NVIDIA model attempts failed');
      error.statusCode = 502;
      error.attempts = attempts;
      reject(error);
    };

    selectedModels.forEach((model, index) => {
      (async () => {
        const startedAt = Date.now();
        try {
          await delay(delaysMs[index] ?? delaysMs[delaysMs.length - 1] ?? 0, overallController.signal);
          if (settled) throw new DOMException('Aborted after successful hedge', 'AbortError');

          const response = await callNvidiaChat({
            apiKey,
            model,
            messages,
            temperature,
            topP,
            seed,
            maxTokens,
            responseFormat,
            signal: controllers[index].signal,
          });

          const parsed = parseJsonLoose(response.content);
          const result = validate ? validate(parsed) : parsed;
          if (!result) {
            const error = new Error(`Model ${model} returned invalid JSON payload`);
            error.model = model;
            throw error;
          }

          const latencyMs = Date.now() - startedAt;
          attempts.push({ model, ok: true, latencyMs });
          if (!settled) {
            abortOthers(index);
            resolve({ result, model, latencyMs, attempts: [...attempts] });
          }
        } catch (error) {
          const latencyMs = Date.now() - startedAt;
          if (!(settled && error?.name === 'AbortError')) {
            attempts.push({ model, ok: false, latencyMs, error: serializeError(error) });
          }
        } finally {
          pending -= 1;
          maybeReject();
        }
      })();
    });
  });
}
