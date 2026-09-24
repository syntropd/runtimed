# io.syntrop.Runtime1: Varlink Interface Specification

The `io.syntrop.Runtime1` interface provides programmatic access to neural model inference, text completion, vector embeddings, and memory lifecycle management.

Socket Endpoint: `/run/syntrop/io.syntrop.Runtime1`

## 1. Interface Definition

```varlink
interface io.syntrop.Runtime1

type LoadedModel (
  name: string,
  architecture: string,
  parameter_count: int,
  memory_bytes: int,
  context_window: int,
  compute_backend: string
)

type GenerationResult (
  text: string,
  prompt_tokens: int,
  completion_tokens: int,
  finish_reason: string,
  duration_ms: int
)

method Generate(model: string, prompt: string, max_tokens: int, temperature: float) -> (result: GenerationResult)
method Embed(model: string, text: string) -> (embedding: []float)
method GetModelStatus(model: string) -> (status: string, model: ?LoadedModel)
method UnloadModel(model: string) -> (freed_bytes: int)
method ListLoadedModels() -> (models: []LoadedModel)

error ModelNotFound(model: string)
error ContextExceeded(requested: int, max: int)
error GenerationFailed(reason: string)
error InvalidParameter(parameter: string)
```

## 2. Methods

### 2.1 `Generate`
Executes token generation for a prompt against the requested model.
- Parameters:
  - `model` (string): Model identifier (e.g. `qwen2.5-coder-7b`).
  - `prompt` (string): Input prompt text.
  - `max_tokens` (int): Maximum new tokens to sample.
  - `temperature` (float): Sampling temperature (0.0 for deterministic greedy decoding).
- Returns:
  - `result` (`GenerationResult`): Text completion, token counts, and duration.

### 2.2 `Embed`
Generates an L2-normalized vector embedding for input text.
- Parameters:
  - `model` (string): Target model identifier.
  - `text` (string): Input text string.
- Returns:
  - `embedding` (`[]float`): 128-dimensional normalized embedding vector.

### 2.3 `GetModelStatus`
Inspects resident memory and device status for a model.
- Parameters:
  - `model` (string): Target model identifier.
- Returns:
  - `status` (string): "loaded" or "not_loaded".
  - `model` (?`LoadedModel`): Model metadata if loaded.

### 2.4 `UnloadModel`
Evicts an active model from memory or GPU device.
- Parameters:
  - `model` (string): Target model identifier.
- Returns:
  - `freed_bytes` (int): Bytes reclaimed.

### 2.5 `ListLoadedModels`
Lists all models currently resident in memory.
- Parameters: none
- Returns:
  - `models` (`[]LoadedModel`): Array of active model metadata records.
