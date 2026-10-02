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

type BoundingBox (
  label: string,
  x1: int,
  y1: int,
  x2: int,
  y2: int
)

type GenerationResult (
  text: string,
  prompt_tokens: int,
  completion_tokens: int,
  finish_reason: string,
  duration_ms: int
)

method Generate(model: string, prompt: string, max_tokens: int, temperature: float, top_k: int, top_p: float, seed: int, image: ?string, grammar_type: ?string, grammar: ?string, speculative_draft_model: ?string, reasoning_budget: ?int, reasoning_effort: ?string) -> (result: GenerationResult)
method Decide(model: string, prompt: string, candidates: []Candidate, temperature: ?float) -> (result: DecisionResult)
method AttachVision(model: string, mmproj: string) -> (model: LoadedModel)
method AttachLora(model: string, lora: string) -> (fused_tensors: []string)
method GetLoad() -> (available_slots: int, max_slots: int, used_bytes: int, models: []LoadedModel)
method Embed(model: string, text: string) -> (embedding: []float)
method GetModelStatus(model: string) -> (status: string, model: ?LoadedModel)
method UnloadModel(model: string) -> (freed_bytes: int)
method ListLoadedModels() -> (models: []LoadedModel)
method StreamAudioOut(text: string, voice: ?string, sink_type: ?string) -> (bytes_streamed: int, sample_rate: int, channels: int)
method GenerateVisual(prompt: string, width: ?int, height: ?int, seed: ?int, steps: ?int, lease_id: ?string) -> (image_path: string, bytes: int, width: int, height: int, format: string)
method GroundVisual(image_bytes: string, task: string) -> (text: string, regions: []BoundingBox)
method CompactKvCache() -> (freed_bytes: int)

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
  - `grammar_type` (string, optional): Optional grammar enforcement ("json" or "regex").
  - `grammar` (string, optional): JSON schema or regex pattern.
  - `speculative_draft_model` (string, optional): Optional draft model identifier for speculative decoding acceleration (e.g. `qwen2.5-0.5b`).
  - `reasoning_budget` (int, optional): Thinking token budget cap.
  - `reasoning_effort` (string, optional): Thinking effort tier ("none", "low", "medium", "high").
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

### 2.9 `StreamAudioOut`
Streams synthesized 24kHz S16LE PCM speech directly to PipeWire or returns bytes.
- Parameters:
  - `text` (string): Text content to synthesize.
  - `voice` (?string, optional): Target voice style profile.
  - `sink_type` (?string, optional): "auto", "pipewire", or "buffer".
- Returns:
  - `bytes_streamed` (int): Number of PCM bytes produced.
  - `sample_rate` (int): Audio sample rate (24000 Hz).
  - `channels` (int): Channel count (1, mono).

### 2.10 `GenerateVisual`
Executes 1-step visual generation into an atomically committed PNG file in runtime storage under compute lease gating.
- Parameters:
  - `prompt` (string): Image description prompt.
  - `width` (?int, optional): Image width in pixels (default 512).
  - `height` (?int, optional): Image height in pixels (default 512).
  - `seed` (?int, optional): Deterministic generation seed.
  - `steps` (?int, optional): Diffusion/LCM step count (default 1).
  - `lease_id` (?string, optional): Active compute lease identifier.
- Returns:
  - `image_path` (string): Path to generated PNG file on host runtime storage.
  - `bytes` (int): Size of generated PNG buffer.
  - `width` (int): Result image width.
  - `height` (int): Result image height.
  - `format` (string): Image encoding format ("png").

---

# io.syntrop.Sensory1: Varlink Interface Specification

The `io.syntrop.Sensory1` interface provides unprivileged environmental awareness, ambient sensing, audio capture with voice activity detection (VAD), video frame acquisition with silhouette detection, screen inspection, and composite operator presence estimation.

Socket Endpoint: `/run/syntrop/io.syntrop.Sensory1`

## 1. Interface Definition

```varlink
interface io.syntrop.Sensory1

type AudioResult (
  audio_pcm_base64: string,
  sample_rate: int,
  channels: int
)

type FrameResult (
  image_base64: string,
  format: string
)

type PresenceResult (
  present: bool,
  confidence: float,
  reason: string
)

method CaptureAudio(duration_ms: ?int, sample_rate: ?int) -> (audio_pcm_base64: string, sample_rate: int, channels: int)
method CaptureFrame(device: ?string, width: ?int, height: ?int) -> (image_base64: string, format: string)
method CaptureScreen(display: ?string) -> (image_base64: string, format: string)
method GetOperatorPresence() -> (present: bool, confidence: float, reason: string)

error DeviceNotFound(device: string)
error CaptureFailed(reason: string)
error PermissionDenied()
```

## 2. Methods

### 2.1 `CaptureAudio`
Records ambient audio from default system audio source via PipeWire PCM ingest (`pw-record`) or synthetic fallback and evaluates energy threshold Voice Activity Detection (VAD).
- Parameters:
  - `duration_ms` (?int, optional): Duration in milliseconds (clamped between 100ms and 10,000ms, default 1,000ms).
  - `sample_rate` (?int, optional): Sampling frequency (default 16,000 Hz, mono S16LE).
- Returns:
  - `audio_pcm_base64` (string): Base64-encoded mono S16LE PCM bytes.
  - `sample_rate` (int): Effective sampling frequency in Hz.
  - `channels` (int): Number of audio channels (1).

### 2.2 `CaptureFrame`
Captures an RGB24 frame via Linux V4L2 device (or synthetic fallback), evaluates silhouette presence heuristic, integrates spatial patch pooling, and encodes to PNG base64.
- Parameters:
  - `device` (?string, optional): V4L2 video device path (default `/dev/video0`).
  - `width` (?int, optional): Frame width in pixels (clamped 64 to 1920, default 640).
  - `height` (?int, optional): Frame height in pixels (clamped 64 to 1080, default 480).
- Returns:
  - `image_base64` (string): PNG image payload encoded as base64 string.
  - `format` (string): Encoding format ("png").

### 2.3 `CaptureScreen`
Captures display desktop screen buffer or synthetic canvas fallback into PNG base64 via unprivileged Wayland portal / KMS dumb buffer detection.
- Parameters:
  - `display` (?string, optional): Display identifier (e.g. `:0`, `wayland-0`).
- Returns:
  - `image_base64` (string): PNG image payload encoded as base64 string.
  - `format` (string): Encoding format ("png").

### 2.4 `GetOperatorPresence`
Fuses microphone audio VAD energy and webcam frame silhouette variance into a unified presence probability and confidence estimate.
- Parameters: none.
- Returns:
  - `present` (bool): True if operator presence is detected via acoustic energy or visual silhouette.
  - `confidence` (float): Detection confidence score (0.0–1.0).
  - `reason` (string): Descriptive heuristic reason detailing active sensor modalities.



