# runtimectl: Command-Line Reference

`runtimectl` is the companion CLI client for interacting with the `runtimed` daemon over Varlink.

## Global Flags

- `-s, --socket <PATH>`: Override Varlink socket path (default: `/run/syntrop/io.syntrop.Runtime1`).
- `--json`: Output raw JSON replies instead of formatted tables.
- `-h, --help`: Display help information.
- `-V, --version`: Display version.

## Subcommands

### 1. `generate`
Sample text completions from a prompt.

```bash
runtimectl generate <PROMPT> [--model <MODEL>] [--max-tokens <N>] [--temperature <FLOAT>] [--json]
```

Example:
```bash
runtimectl generate "Analyze root cause of failed nginx worker" --max-tokens 128
```

### 2. `embed`
Compute vector embeddings for text.

```bash
runtimectl embed <TEXT> [--model <MODEL>] [--json]
```

Example:
```bash
runtimectl embed "Kernel page fault at 0x0"
```

### 3. `status`
Inspect loaded memory footprint and backend details for a model.

```bash
runtimectl status <MODEL> [--json]
```

Example:
```bash
runtimectl status qwen2.5-coder-7b
```

### 4. `unload`
Evict a model from memory to reclaim RAM or GPU VRAM.

```bash
runtimectl unload <MODEL> [--json]
```

Example:
```bash
runtimectl unload qwen2.5-coder-7b
```

### 5. `list`
List all models currently active in memory.

```bash
runtimectl list [--json]
```

### 6. `info`
Introspect daemon vendor metadata and supported Varlink interfaces.

```bash
runtimectl info [--json]
```

### 7. `completions`
Generate shell tab completion script for bash, zsh, fish, or powershell.

```bash
runtimectl completions <SHELL>
```
