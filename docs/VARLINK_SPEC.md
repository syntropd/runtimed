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

method Generate(model: string, prompt: string, max_tokens: int, temperature: float, top_k: int, top_p: float, seed: int, image: ?string) -> (result: GenerationResult)
method AttachVision(model: string, mmproj: string) -> (model: LoadedModel)
method AttachLora(model: string, lora: string) -> (fused_tensors: []string)
method GetLoad() -> (available_slots: int, max_slots: int, used_bytes: int, models: []LoadedModel)
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
  - `top_k` (int, optional): Top-k truncation, 0 disables.
  - `top_p` (float, optional): Nucleus truncation, 1.0 disables.
  - `seed` (int, optional): Sampling seed, 0 draws entropy from the clock.
  - `image` (string, optional): Base64-encoded PNG/JPEG for multimodal generation (needs `AttachVision` first; the prompt is wrapped in the chat template).
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

### 2.4 `AttachVision`
Binds a vision projector to a loaded model, enabling the `image` parameter on `Generate`. Re-attaching replaces the previous tower.
- Parameters:
  - `model` (string): Target model identifier (must be loaded).
  - `mmproj` (string): Projector file: absolute path or models-directory entry.
- Returns:
  - `model` (`LoadedModel`): The model's metadata.

### 2.5 `AttachLora`
Fuses a LoRA adapter file into a loaded model's live weights (base file untouched; reload to detach). Returns the fused base-weight names.
- Parameters:
  - `model` (string): Target model identifier (must be loaded).
  - `lora` (string): Adapter file: absolute path or models-directory entry.
- Returns:
  - `fused_tensors` (`[]string`): Fused base-weight names.

### 2.6 `UnloadModel`
Evicts an active model from memory or GPU device.
- Parameters:
  - `model` (string): Target model identifier.
- Returns:
  - `freed_bytes` (int): Bytes reclaimed.

### 2.7 `GetLoad`
Reports daemon load for fleet overflow routing: free request slots plus resident models and bytes. A router spills `Generate` calls to the least-loaded node.
- Parameters: none.
- Returns:
  - `available_slots` (int): Free request slots right now.
  - `max_slots` (int): Configured concurrency cap.
  - `used_bytes` (int): Sum of loaded-model footprints.
  - `models` (`[]LoadedModel`): Resident models.

### 2.8 `ListLoadedModels`
Lists all models currently resident in memory.
- Parameters: none
- Returns:
  - `models` (`[]LoadedModel`): Array of active model metadata records.
