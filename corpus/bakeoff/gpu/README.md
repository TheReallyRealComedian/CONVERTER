# GPU-Feld (P3) — eigene Container auf der Mintbox

CONVERTERs Compose-Stack bleibt unberührt; alles hier läuft in eigenen
Containern gegen die A2000 (12 GB). Repo-Clone auf der Mintbox:
`~/CODE/CONVERTER` (Korpus-Binärdateien sind gitignored und werden per
rsync vom Mac provisioniert — Mac bleibt Source-of-Truth, Ergebnisse
wandern per rsync zurück und werden DORT committet).

## Kandidaten

| Kandidat | Container | Aufruf |
|---|---|---|
| `mineru-vlm` | `mineru:latest` (offizielles Dockerfile, vllm-Basis) | `mineru -p … -o … -b vlm-vllm-engine` |
| `marker2` | `bakeoff-marker:latest` (Dockerfile.marker) | `marker_single … --output_format markdown` |
| `vlm-dots` | Server `vllm/vllm-openai` serviert `rednote-hilab/dots.ocr` + Client `bakeoff-dotsclient` (Repo-Parser) | Parser gegen `:8000` |

Fallback, falls dots.ocr die 12 GB sprengt: PaddleOCR-VL (0,9B) — erst
messen, dann wechseln.

## Ablauf auf der Mintbox

```bash
cd ~/CODE/CONVERTER && git pull
bash corpus/bakeoff/gpu/build_gpu.sh          # Container bauen (einmalig, lange)
# Läufe (Beispiel):
python3 corpus/bakeoff/harness/run_one.py --candidate mineru-vlm --class 05
```

Kennzahlen zusätzlich zu P1/P2: `vram_peak_mb` (nvidia-smi-Sampler um jeden
Lauf) und Seiten/Minute (aus runtime + Seitenzahl; für Server-Kandidaten
zusätzlich Steady-State auf 12, weil Modell-Load dort amortisiert).

## Die zwei Messungen, die kein Benchmark liefert

1. **Durchsatz auf der A2000** — für diese Karte existiert keine
   publizierte Dokument-VLM-Messung. Cold (inkl. Load) und Steady-State.
2. **Deutsche Qualität** — kein publizierter Deutsch-Score für dots.ocr /
   PaddleOCR-VL / MinerU-VLM; Scoring läuft auf dem Mac (Gold + Struktur +
   Judge), der Korpus ist die einzige Quelle.

**Loop-Rate**: 06/07 tragen die dokumentierten Auslöser (Punkt-/
Unterstrichlinien) — `max_line_repeat`/`max_ngram_repeat`/`loop_flag` aus
score_struct sind die Messgröße, plus Befund im Judge.
