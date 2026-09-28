#!/bin/bash
# page_score.sh — per-page scorecard: lines, binary bytes, heat, reach.
#
# Usage: qa/page_score.sh [--bin] [--heat [--heat-runs N]] [--out FILE]
#   default: lines + reach (fast, no build)
#   --bin:   + shipped-bytes attribution via nm (builds the workspace)
#   --heat:  + CPU share sampled while test binaries run (slow, bounded);
#            --heat-runs N repeats the suite N times for fast suites
# Writes page-score.json. Best-effort fields are null with notes when
# their tools are missing. Only stock Linux tools are used.
set -u
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OUT="page-score.json"
WANT_BIN=0
WANT_HEAT=0
HEAT_RUNS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --bin) WANT_BIN=1 ;;
    --heat) WANT_HEAT=1 ;;
    --heat-runs) HEAT_RUNS="$2"; shift ;;
    --out) OUT="$2"; shift ;;
  esac
  shift
done
cd "$ROOT" || exit 1
TMP="$(mktemp -d /tmp/page-score.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# ---- pages + lines (exact) ----
find . -name '*.rs' -not -path '*/target/*' -not -path '*/.git/*' | sort > "$TMP/pages.txt"
: > "$TMP/lines.tsv"
while IFS= read -r f; do
  printf '%s\t%s\n' "$f" "$(wc -l < "$f")" >> "$TMP/lines.tsv"
done < "$TMP/pages.txt"

# ---- reach: files referencing each page's module stem (approx) ----
: > "$TMP/reach.tsv"
while IFS= read -r f; do
  stem="$(basename "$f" .rs)"
  n=$(grep -rl --include='*.rs' -E "(^|[^A-Za-z0-9_])${stem}::|use [^;]*${stem}[ ;]" . 2>/dev/null | grep -v target | grep -v "^${f}$" | wc -l)
  printf '%s\t%s\n' "$f" "$n" >> "$TMP/reach.tsv"
done < "$TMP/pages.txt"

NOTES=""
# ---- bin_bytes: .text sizes per source file via nm (approx) ----
: > "$TMP/bin.tsv"
if [ "$WANT_BIN" = 1 ]; then
  if ! command -v nm >/dev/null; then
    NOTES="$NOTES bin_bytes unavailable (no nm);"
  elif ! cargo build --workspace --offline >/dev/null 2>&1; then
    NOTES="$NOTES bin_bytes unavailable (build failed);"
  else
    : > "$TMP/nm.tsv"
    # Newest rlib per crate only (stale hashes and package-verify copies out).
    for rlib in $(ls -t target/debug/deps/*.rlib 2>/dev/null | awk -F'[-.]' '{k=""; for(i=1;i<NF-1;i++) k=k $i "-"; if (!seen[k]++) print}'); do
      [ -f "$rlib" ] || continue
      nm --print-size --line-numbers "$rlib" 2>/dev/null | awk '
        / [TtWw] / {
          size = strtonum("0x" $2)
          loc = $NF
          sub(/:[0-9]+$/, "", loc)
          if (loc != "" && loc != "??") print loc "\t" size
        }' >> "$TMP/nm.tsv"
    done
    # Map nm paths onto repo pages; packaged-copy paths (target/package)
    # match by relative path; other strays match by unique basename only.
    awk -v root="$ROOT" '
      NR == FNR { pages[$1] = 1; base[$1] = $1; sub(/.*\//, "", base[$1]); count[base[$1]]++; next }
      {
        loc = $1; size = $2
        if (index(loc, root) == 1) { loc = substr(loc, length(root) + 2); if (loc !~ /^\.\//) loc = "./" loc }
        else if (match(loc, /\/target\/package\/[^\/]+\//)) { loc = "./" substr(loc, RSTART + RLENGTH) }
        else { b = loc; sub(/.*\//, "", b); if (count[b] == 1) { for (p in pages) if (base[p] == b) { loc = p; break } } else next }
        if (loc in pages) bytes[loc] += size
      }
      END { for (p in bytes) print p "\t" bytes[p] }
    ' "$TMP/pages.txt" "$TMP/nm.tsv" > "$TMP/bin.tsv"
    NOTES="$NOTES bin_bytes from newest dev rlib per crate via nm (inlining approx, bins re-link the same code);"
  fi
fi

# ---- heat: CPU share per source file, sampled under test runs (approx) ----
: > "$TMP/heat.tsv"
if [ "$WANT_HEAT" = 1 ]; then
  if ! command -v eu-stack >/dev/null; then
    NOTES="$NOTES heat unavailable (no eu-stack);"
  elif ! cargo test --workspace --no-run --offline >/dev/null 2>&1; then
    NOTES="$NOTES heat unavailable (test build failed);"
  else
    : > "$TMP/hits.tsv"
    SAMPLED=0
    TOTAL=0
    DEADLINE=$((SECONDS + 1200))
    for RUN in $(seq 1 "$HEAT_RUNS"); do
      for bin in target/debug/deps/*; do
        [ -f "$bin" ] && [ -x "$bin" ] || continue
        case "$bin" in *.d|*.rlib|*.rmeta|*.so|*.dylib) continue ;; esac
        TOTAL=$((TOTAL + 1))
        [ $SECONDS -gt $DEADLINE ] && break 2
        timeout -s KILL 300 "$bin" </dev/null >>"$TMP/run.log" 2>&1 &
        PID=$!
        sleep 0.05
        TPID=$(pgrep -P $PID 2>/dev/null | head -1)
        [ -z "$TPID" ] && { wait $PID 2>/dev/null; continue; }
        SAMPLED=$((SAMPLED + 1))
        EXE=$(readlink -f /proc/$TPID/exe 2>/dev/null)
        TICKS=0
        while kill -0 $TPID 2>/dev/null && [ $TICKS -lt 2400 ]; do
          MAPS=$(awk -v exe="$EXE" '$6 == exe && index($2, "r-x") == 1 {print $1; exit}' /proc/$TPID/maps 2>/dev/null)
          BASE=${MAPS%%-*}; END=${MAPS##*-}
          for P in $TPID $(pgrep -P $TPID 2>/dev/null); do
            eu-stack -p $P 2>/dev/null | grep -oE "#[0-9]+ +0x[0-9a-f]+" | awk '{print $2}' | head -16 >> "$TMP/frames.$SAMPLED"
          done
          echo "$EXE $BASE $END" > "$TMP/base.$SAMPLED"
          TICKS=$((TICKS + 1))
          sleep 0.1
        done
        wait $PID 2>/dev/null
      done
    done
    # Attribute one innermost repo frame per tick batch is overkill; count
    # every frame resolving into the repo (documented in notes).
    : > "$TMP/heatraw.tsv"
    for f in "$TMP"/frames.*; do
      [ -f "$f" ] || continue
      tag=${f##*/frames.}; read -r EXE BASE END < "$TMP/base.$tag" 2>/dev/null || continue
      [ -n "$BASE" ] || continue
      awk -v b="0x$BASE" -v e="0x$END" '{a=strtonum($1); if (a>=strtonum(b) && a<strtonum(e)) printf "%x\n", a-strtonum(b)}' "$f" | sort -u > "$TMP/addrs.$tag"
      [ -s "$TMP/addrs.$tag" ] || continue
      # shellcheck disable=SC2046
      addr2line -e "$EXE" -f -C $(cat "$TMP/addrs.$tag" | tr '\n' ' ') 2>/dev/null | grep -E "^/" | sed "s|^$ROOT/||" >> "$TMP/heatraw.tsv"
    done
    TOTALHITS=$(wc -l < "$TMP/heatraw.tsv")
    if [ "$TOTALHITS" -ge 50 ]; then
      cut -d: -f1 "$TMP/heatraw.tsv" | sort | uniq -c | awk -v t="$TOTALHITS" -v r="$ROOT/" '{p=$2; sub("^"r, "", p); printf "./%s\t%.2f\n", p, 100*$1/t}' > "$TMP/heat.tsv"
      NOTES="$NOTES heat from $TOTALHITS attributed samples over $SAMPLED/$TOTAL binaries at 10Hz incl. one child level; every repo-resolving frame counts;"
    else
      NOTES="$NOTES heat unavailable: only $TOTALHITS attributed samples (need 50; suite too fast — use --heat-runs N to amplify);"
    fi
  fi
fi

# ---- emit ----
python3 - "$TMP" "$OUT" "$NOTES" <<'EOF'
import json, sys
tmp, out, notes = sys.argv[1], sys.argv[2], sys.argv[3]
def tsv(name):
    d = {}
    try:
        with open(f"{tmp}/{name}") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" in line:
                    k, v = line.split("\t", 1)
                    d[k] = v
    except FileNotFoundError:
        pass
    return d
lines, reach, bins, heat = tsv("lines.tsv"), tsv("reach.tsv"), tsv("bin.tsv"), tsv("heat.tsv")
import subprocess
rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--short"], capture_output=True, text=True).stdout.strip())
repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip().rstrip("/").rsplit("/", 1)[-1]
pages = []
with open(f"{tmp}/pages.txt") as f:
    for line in f:
        p = line.rstrip("\n")
        pages.append({
            "path": p,
            "lines": int(lines.get(p, 0)),
            "bin_bytes": int(bins[p]) if p in bins else None,
            "heat_pct": float(heat[p]) if p in heat else None,
            "reach": int(reach.get(p, 0)),
        })
with open(out, "w") as f:
    json.dump({"repo": repo, "rev": rev, "dirty": dirty, "pages": pages, "notes": notes.strip()}, f, indent=1)
print(f"wrote {out}: {len(pages)} pages")
EOF
