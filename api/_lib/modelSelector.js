import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import {
  DEFAULT_DEEP_MODEL,
  DEFAULT_PRIMARY_FAST_MODEL,
  DEFAULT_QUALITY_MODEL_FALLBACK,
  DEFAULT_SECONDARY_FAST_MODEL,
  dedupeModels,
  filterLikelyChatModels,
  listNvidiaModels,
} from './nvidia.js';

const BENCHMARK_CACHE_PATH = resolve(process.cwd(), 'data/generated/nvidia-model-benchmarks.json');
const DEFAULT_BENCHMARK_TTL_MS = 86_400_000;
const DEFAULT_DISCOVERY_TTL_MS = 21_600_000;

// NVIDIA hosted endpoint availability changes frequently. Keep this list intentionally
// short and validate it against GET /v1/models before using it.
const CURRENT_HOSTED_MODELS = {
  primaryFast: [
    'google/gemma-4-31b-it',
    'openai/gpt-oss-20b',
    'meta/llama-3.1-8b-instruct',
  ],
  secondaryFast: [
    'openai/gpt-oss-20b',
    'openai/gpt-oss-120b',
    'google/gemma-4-31b-it',
    'meta/llama-3.1-8b-instruct',
  ],
  quality: [
    'openai/gpt-oss-120b',
    'google/gemma-4-31b-it',
    'nvidia/nemotron-3-super-120b-a12b',
    'nvidia/llama-3.3-nemotron-super-49b-v1',
  ],
  deep: [
    'nvidia/nemotron-3-super-120b-a12b',
    'google/gemma-4-31b-it',
    'openai/gpt-oss-120b',
  ],
};

// These models were previously preferred by this app, but their NVIDIA hosted
// endpoints have since been deprecated. They may still exist as downloadable or
// partner endpoints, so only exclude them from the free hosted API race.
const DEPRECATED_HOSTED_MODELS = new Set([
  'qwen/qwen3.5-122b-a10b',
  'qwen/qwen3.5-397b-a17b',
]);

let memoryBenchmarkCache = null;
let recommendedModelCache = null;

function positiveNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function envFlag(value, fallback = true) {
  if (value === undefined) return fallback;
  return !['0', 'false', 'no', 'off'].includes(String(value).toLowerCase());
}

function benchmarkTtlMs() {
  return positiveNumber(process.env.NVIDIA_MODEL_BENCHMARK_CACHE_TTL_MS, DEFAULT_BENCHMARK_TTL_MS);
}

function discoveryTtlMs() {
  return positiveNumber(process.env.NVIDIA_MODEL_DISCOVERY_CACHE_TTL_MS, DEFAULT_DISCOVERY_TTL_MS);
}

function isFresh(isoDate, ttl = benchmarkTtlMs()) {
  const time = Date.parse(isoDate || '');
  return Number.isFinite(time) && Date.now() - time < ttl;
}

function compactModels(models) {
  return dedupeModels(models).filter(Boolean);
}

function withoutDeprecatedHostedModels(models) {
  return compactModels(models).filter((model) => !DEPRECATED_HOSTED_MODELS.has(model));
}

function withCache(result) {
  recommendedModelCache = {
    value: result,
    expiresAt: Date.now() + discoveryTtlMs(),
  };
  return result;
}

export async function loadBenchmarkCache({ allowStale = false } = {}) {
  if (memoryBenchmarkCache && (allowStale || isFresh(memoryBenchmarkCache.benchmarkedAt))) {
    return memoryBenchmarkCache;
  }

  try {
    const raw = await readFile(BENCHMARK_CACHE_PATH, 'utf8');
    const parsed = JSON.parse(raw);
    if (!allowStale && !isFresh(parsed?.benchmarkedAt)) return null;
    memoryBenchmarkCache = parsed;
    return parsed;
  } catch {
    return null;
  }
}

export async function saveBenchmarkCache(cache) {
  await mkdir(dirname(BENCHMARK_CACHE_PATH), { recursive: true });
  const safeCache = { ...cache };
  delete safeCache.apiKey;
  await writeFile(BENCHMARK_CACHE_PATH, `${JSON.stringify(safeCache, null, 2)}\n`, 'utf8');
  memoryBenchmarkCache = safeCache;
  recommendedModelCache = null;
  return safeCache;
}

export function getConfiguredModels() {
  return {
    primaryFastModel: process.env.NVIDIA_PRIMARY_MODEL || process.env.NVIDIA_FAST_MODEL_1 || '',
    secondaryFastModel: process.env.NVIDIA_SECONDARY_MODEL || process.env.NVIDIA_FAST_MODEL_2 || '',
    qualityModel: process.env.NVIDIA_QUALITY_MODEL || process.env.NVIDIA_MODEL || '',
    deepModel: process.env.NVIDIA_DEEP_MODEL || '',
  };
}

function chooseFirst(available, candidates, fallback) {
  return candidates.find((model) => model && available.includes(model)) || available[0] || fallback;
}

function modelsFromBenchmark(cache) {
  if (!cache) return {};
  return {
    primaryFastModel: cache.primaryFastModel,
    secondaryFastModel: cache.secondaryFastModel,
    qualityModel: cache.qualityModel,
    deepModel: cache.deepModel,
  };
}

function normalizeRecommendation(recommended, source) {
  return {
    primaryFastModel: recommended.primaryFastModel || DEFAULT_PRIMARY_FAST_MODEL,
    secondaryFastModel: recommended.secondaryFastModel || DEFAULT_SECONDARY_FAST_MODEL,
    qualityModel: recommended.qualityModel || DEFAULT_QUALITY_MODEL_FALLBACK,
    deepModel: recommended.deepModel || DEFAULT_DEEP_MODEL,
    source,
    ...(recommended.discoveredModels ? { discoveredModels: recommended.discoveredModels } : {}),
    ...(recommended.staleConfiguredModels?.length ? { staleConfiguredModels: recommended.staleConfiguredModels } : {}),
  };
}

function recommendationFromAvailable(available, configured = {}) {
  const strictEnv = envFlag(process.env.NVIDIA_MODEL_STRICT_ENV, false);
  const configuredValues = compactModels(Object.values(configured));
  const staleConfiguredModels = configuredValues.filter((model) => !available.includes(model));

  const slotCandidates = (currentCandidates, configuredCandidate) => strictEnv
    ? [configuredCandidate, ...currentCandidates]
    : [...currentCandidates, configuredCandidate];

  return {
    primaryFastModel: chooseFirst(
      available,
      slotCandidates(CURRENT_HOSTED_MODELS.primaryFast, configured.primaryFastModel),
      DEFAULT_PRIMARY_FAST_MODEL,
    ),
    secondaryFastModel: chooseFirst(
      available,
      slotCandidates(CURRENT_HOSTED_MODELS.secondaryFast, configured.secondaryFastModel),
      DEFAULT_SECONDARY_FAST_MODEL,
    ),
    qualityModel: chooseFirst(
      available,
      slotCandidates(CURRENT_HOSTED_MODELS.quality, configured.qualityModel),
      DEFAULT_QUALITY_MODEL_FALLBACK,
    ),
    deepModel: chooseFirst(
      available,
      slotCandidates(CURRENT_HOSTED_MODELS.deep, configured.deepModel),
      DEFAULT_DEEP_MODEL,
    ),
    discoveredModels: available,
    staleConfiguredModels,
  };
}

export async function getRecommendedNvidiaModels({ forceRefresh = false } = {}) {
  if (!forceRefresh && recommendedModelCache && recommendedModelCache.expiresAt > Date.now()) {
    return recommendedModelCache.value;
  }

  const configured = getConfiguredModels();
  const hasConfiguredModels = compactModels(Object.values(configured)).length > 0;

  // Always prefer a live /models check when enabled. NVIDIA removes hosted models
  // over time, so treating environment variables as permanent truth caused the app
  // to keep racing stale endpoints and silently fall back to local copy.
  if (envFlag(process.env.NVIDIA_MODEL_DISCOVERY, true)) {
    try {
      const discovered = withoutDeprecatedHostedModels(filterLikelyChatModels(await listNvidiaModels()));
      if (discovered.length > 0) {
        const recommendation = recommendationFromAvailable(discovered, configured);
        return withCache(normalizeRecommendation(
          recommendation,
          hasConfiguredModels ? 'env+live-discovery' : 'live-discovery',
        ));
      }
    } catch (error) {
      console.warn('NVIDIA model discovery failed:', error?.message || error);
    }
  }

  // If discovery itself is temporarily unavailable, keep configured models as a
  // resilience fallback instead of disabling AI entirely.
  if (hasConfiguredModels) {
    return withCache(normalizeRecommendation(configured, 'env-unverified'));
  }

  const benchmark = await loadBenchmarkCache();
  const benchmarkModels = modelsFromBenchmark(benchmark);
  if (compactModels(Object.values(benchmarkModels)).length > 0) {
    return withCache(normalizeRecommendation(benchmarkModels, 'benchmark-cache'));
  }

  return withCache(normalizeRecommendation({}, 'defaults'));
}

export async function resolveNvidiaModelsForVerseDevotion() {
  const recommended = await getRecommendedNvidiaModels();
  const modelsForRace = compactModels([
    recommended.primaryFastModel,
    recommended.qualityModel,
    recommended.secondaryFastModel,
  ]).slice(0, 3);

  return {
    ...recommended,
    modelsForRace,
  };
}
