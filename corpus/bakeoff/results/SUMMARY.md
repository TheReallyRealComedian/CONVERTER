# Bake-off — Rollup (automatisch aus results/)

## Statusmatrix (main-Läufe)

| Kandidat | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09A | 09B | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | ✓ | ✓ | ✓ |
| eigenbau | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | · | · | ✓ | ✓ | ✓ |
| gemini-nativ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | · | · | ✓ | ✓ | ✓ |
| markitdown | · | · | · | · | · | · | · | · | ✓ | ✓ | · | · | · | · | · |
| pandoc | · | · | · | · | · | · | · | ✓ | · | · | · | · | · | · | · |
| tesseract | · | · | · | · | ✓ | ✓ | ✓ | · | · | · | · | · | · | ✓ | ✓ |
| textlayer | ✓ | ✓ | ✓ | ✓ | ✗ RuntimeError | ✗ RuntimeError | ✗ RuntimeError | · | · | · | · | · | ✓ | ✓ | ✓ |
| trafilatura | · | · | · | · | · | · | · | · | · | · | ✓ | · | · | · | · |
| unstructured-neu | · | · | · | · | · | · | · | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · |
| unstructured-pin | · | · | · | · | · | · | · | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · |

## Gegen Gold (Metrik a) — inkl. Kalibrierungs-Kandidaten

| Kandidat | Dok | f1 | CER | ZellenR | ZellenP | Regeln | Eigenheiten |
|---|---|---|---|---|---|---|---|
| docling | 01.gold | 0.9594 | 0.1042 | 0.361 | 0.361 | R2 0/11 | 1/4 erhalten |
| docling | 07.gold | 0.8540 | 0.3032 | 0.234 | 0.429 | R1 2/3 |  |
| docling | 08 | 0.9553 | 0.0264 | 0.963 | 0.897 | R3 ✗/✗/✗/✗ | 3/3 erhalten |
| eigenbau | 01.gold | 0.9149 | 0.0645 | 0.000 | 0.000 | R2 0/11 | 3/4 erhalten |
| eigenbau | 07.gold | 0.9199 | 0.1477 | 0.766 | 0.776 | R1 3/3 |  |
| gemini-cal-high | 01.gold | 0.9682 | 0.0676 | 0.868 | 0.868 | R2 11/11 | 4/4 erhalten |
| gemini-cal-high | 07.gold | 0.9502 | 0.0155 | 0.571 | 0.647 | R1 3/3 |  |
| gemini-cal-low | 01.gold | 0.9662 | 0.0787 | 0.868 | 0.868 | R2 11/11 | 4/4 erhalten |
| gemini-cal-low | 07.gold | 0.9429 | 0.0268 | 0.649 | 0.581 | R1 0/3 |  |
| gemini-cal-medium | 01.gold | 0.9809 | 0.0498 | 0.866 | 0.866 | R2 11/11 | 3/4 erhalten |
| gemini-cal-medium | 07.gold | 0.9792 | 0.0223 | 0.779 | 0.706 | R1 3/3 |  |
| gemini-nativ | 01.gold | 0.9781 | 0.1065 | 0.868 | 0.868 | R2 11/11 | 4/4 erhalten |
| gemini-nativ | 07.gold | 0.9628 | 0.0145 | 0.623 | 0.632 | R1 2/3 |  |
| pandoc | 08 | 0.9413 | 0.0820 | 0.963 | 0.963 | R3 ✓/✓/✓/✓ | 3/3 erhalten |
| tesseract | 07.gold | 0.7796 | 0.1597 | 0.000 | 0.000 | R1 0/3 |  |
| textlayer | 01.gold | 0.9149 | 0.0645 | 0.000 | 0.000 | R2 0/11 | 3/4 erhalten |
| unstructured-neu | 08 | 0.9639 | 0.0517 | 0.963 | 0.839 | R3 ✗/✗/✗/✗ | 3/3 erhalten |
| unstructured-pin | 08 | 0.9639 | 0.0517 | 0.963 | 0.839 | R3 ✗/✗/✗/✗ | 3/3 erhalten |

## Struktur (Metrik b, main-Läufe)

### 01 — Paper zweispaltig (EN, 54 S.)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | 0.892 | 0.953 | — | 0.98 | 5+0 | 0/94/0/0 | 0 | – | 146.78 | 0.000 |
| eigenbau | ✓ | 0.999 | 0.997 | — | 1.00 | 1+0 | 2/0/0/0 | 0 | – | 5.6 | 0.000 |
| gemini-nativ | ✓ | 0.904 | 0.911 | — | 1.05 | 3+0 | 10/9/56/18 | 9 | – | 436.83 | 1.005 |
| textlayer | ✓ | 1.000 | 1.000 | — | 1.00 | 0+0 | 2/0/0/0 | 0 | – | 0.23 | 0.000 |

### 02 — Regulatorische Guideline (DE, 57 S.)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | 0.915 | 0.985 | 0.889 | 0.90 | 42+0 | 0/55/0/0 | 344 | – | 203.33 | 0.000 |
| eigenbau | ✓ | 0.489 | 0.762 | 0.437 | 0.73 | 9+0 | 0/0/0/0 | 460 | ⚠ | 130.14 | 0.000 |
| gemini-nativ | ✓ | 0.975 | 0.986 | 0.927 | 0.96 | 45+2 | 19/9/11/3 | 342 | – | 166.78 | 0.382 |
| textlayer | ✓ | 1.000 | 1.000 | 1.000 | 1.00 | 0+0 | 0/0/0/0 | 346 | – | 0.75 | 0.000 |

### 03 — Tabelle ueber Seitengrenze (DE, 20 S.)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | 0.917 | 0.939 | 0.842 | 0.94 | 17+0 | 0/11/0/0 | 88 | – | 182.5 | 0.000 |
| eigenbau | ✓ | 0.900 | 0.963 | 0.773 | 0.95 | 2+0 | 0/0/0/0 | 60 | – | 21.64 | 0.000 |
| gemini-nativ | ✓ | 1.000 | 0.973 | 0.865 | 0.95 | 10+0 | 10/0/11/31 | 82 | – | 34.73 | 0.067 |
| textlayer | ✓ | 1.000 | 1.000 | 1.000 | 1.00 | 0+0 | 0/0/0/0 | 82 | – | 0.05 | 0.000 |

### 04 — Verbundene Zellen (DE, 12 S.)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | 0.983 | 0.989 | 0.929 | 0.92 | 12+0 | 0/5/0/0 | 64 | – | 642.71 | 0.000 |
| eigenbau | ✓ | 0.956 | 0.981 | 0.951 | 0.89 | 3+0 | 0/0/0/0 | 53 | – | 30.62 | 0.000 |
| gemini-nativ | ✓ | 0.983 | 0.999 | 0.970 | 1.11 | 4+8 | 0/0/0/0 | 64 | – | 178.86 | 0.426 |
| textlayer | ✓ | 1.000 | 1.000 | 1.000 | 1.00 | 0+0 | 0/0/0/0 | 66 | – | 0.47 | 0.000 |

### 05 — Scan sauber, textebenen-frei (DE, 15 S.)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | — | — | — | — | 0+0 | 0/3/0/0 | 562 | – | 301.51 | 0.000 |
| eigenbau | ✓ | — | — | — | — | 0+0 | 2/1/0/0 | 540 | – | 256.51 | 0.388 |
| gemini-nativ | ✓ | — | — | — | — | 0+0 | 1/0/0/0 | 583 | – | 62.16 | 0.098 |
| tesseract | ✓ | — | — | — | — | 0+0 | 1/0/0/0 | 584 | – | 35.18 | 0.000 |
| textlayer | ✗ RuntimeError | | | | | | | | | 0.33 | |

### 06 — Scan degradiert (Kopie der Kopie, schief)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | — | — | — | — | 0+0 | 0/6/0/0 | 31 | – | 96.4 | 0.000 |
| eigenbau | ✓ | — | — | — | — | 6+0 | 4/2/1/0 | 95 | – | 62.74 | 0.040 |
| gemini-nativ | ✓ | — | — | — | — | 4+1 | 1/2/4/0 | 73 | – | 28.58 | 0.035 |
| tesseract | ✓ | — | — | — | — | 0+0 | 0/0/0/0 | 1 | – | 21.49 | 0.000 |
| textlayer | ✗ RuntimeError | | | | | | | | | 0.03 | |

### 07 — Formular Punktlinien (Scan, DE, 2 S.)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | 0.904 | 0.860 | 0.809 | 0.98 | 1+0 | 0/6/0/0 | 76 | – | 67.24 | 0.000 |
| eigenbau | ✓ | 0.936 | 0.705 | 0.919 | 1.28 | 5+0 | 1/2/3/0 | 95 | – | 83.48 | 0.105 |
| gemini-nativ | ✓ | 0.931 | 0.888 | 0.901 | 1.05 | 2+4 | 1/6/0/0 | 75 | – | 25.06 | 0.022 |
| tesseract | ✓ | 0.886 | 0.905 | 0.858 | 0.94 | 0+0 | 0/0/0/0 | 71 | – | 27.79 | 0.000 |
| textlayer | ✗ RuntimeError | | | | | | | | | 0.27 | |

### 08 — DOCX Fussnoten (DE)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | 0.958 | 0.898 | 0.956 | 1.01 | 2+0 | 0/5/8/15 | 164 | – | 8.18 | 0.000 |
| pandoc | ✓ | 0.979 | 0.878 | 0.968 | 1.12 | 0+2 | 6/8/16/0 | 177 | – | 1.02 | 0.000 |
| unstructured-neu | ✓ | 0.913 | 0.996 | 0.913 | 0.93 | 2+0 | 5/8/16/0 | 153 | – | 52.12 | 0.000 |
| unstructured-pin | ✓ | 0.913 | 0.996 | 0.913 | 0.93 | 2+0 | 5/8/16/0 | 153 | – | 17.07 | 0.000 |

### 09A — PPTX mehrspaltig + Notes (DE, 25 Folien)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | 0.871 | 0.770 | 0.615 | 1.13 | 21+0 | 0/0/0/0 | 88 | – | 2.31 | 0.000 |
| markitdown | ✓ | 1.000 | 0.712 | 0.556 | 1.35 | 20+0 | 0/0/23/0 | 108 | – | 4.15 | 0.000 |
| unstructured-neu | ✓ | 0.394 | 1.000 | 0.363 | 0.40 | 9+0 | 0/0/0/0 | 34 | – | 12.52 | 0.000 |
| unstructured-pin | ✓ | 0.383 | 1.000 | 0.353 | 0.39 | 9+0 | 0/0/0/0 | 34 | – | 7.86 | 0.000 |

### 09B — PPTX SmartArt (EN, 98 Folien, 53 MB)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | 0.949 | 0.848 | 0.768 | 1.10 | 8+0 | 36/0/0/0 | 21 | – | 5.18 | 0.000 |
| markitdown | ✓ | 0.988 | 0.689 | 0.771 | 1.34 | 2+0 | 78/0/11/0 | 42 | – | 3.47 | 0.000 |
| unstructured-neu | ✓ | 0.948 | 0.985 | 0.765 | 0.97 | 7+0 | 36/0/0/0 | 21 | – | 16.44 | 0.000 |
| unstructured-pin | ✓ | 0.944 | 0.985 | 0.762 | 0.96 | 7+0 | 36/0/0/0 | 21 | – | 7.39 | 0.000 |

### 10 — HTML Artikel (DE)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trafilatura | ✓ | 0.692 | 1.000 | 0.692 | 0.69 | 0+0 | 0/0/0/3 | 123 | – | 1.47 | 0.000 |
| unstructured-neu | ✓ | 0.983 | 0.990 | 0.981 | 0.99 | 0+1 | 1/2/3/9 | 178 | – | 10.37 | 0.000 |
| unstructured-pin | ✓ | 0.983 | 0.990 | 0.981 | 0.99 | 0+1 | 1/2/3/9 | 178 | – | 6.92 | 0.000 |

### 11 — EML Zitatkette (DE)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unstructured-neu | ✓ | 0.976 | 1.000 | 0.976 | 0.97 | 2+0 | 0/0/0/0 | 62 | – | 8.15 | 0.000 |
| unstructured-pin | ✓ | 0.976 | 1.000 | 0.976 | 0.97 | 2+0 | 0/0/0/0 | 62 | – | 6.53 | 0.000 |

### 12 — Grosses PDF (DE, 280 S.) — Durchsatz/Kosten
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | 0.894 | 0.947 | — | 0.97 | 68+0 | 0/416/0/0 | 15589 | – | 492.19 | 0.000 |
| eigenbau | ✓ | 0.984 | 0.997 | — | 0.98 | 28+0 | 0/0/0/0 | 15366 | ⚠ | 201.04 | 0.000 |
| gemini-nativ | ✓ | 0.897 | 0.946 | — | 0.98 | 65+0 | 24/17/84/103 | 15632 | – | 1564.03 | 3.365 |
| textlayer | ✓ | 1.000 | 1.000 | — | 1.00 | 0+0 | 0/0/0/0 | 15633 | – | 2.96 | 0.000 |

### 13 — Mischdokument (6 mixed / 2 native / 2 scanned von 32 S.)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | 0.599 | 0.178 | 0.353 | 2.84 | 0+0 | 0/46/0/0 | 170 | – | 272.84 | 0.000 |
| eigenbau | ✓ | 0.999 | 0.206 | 0.967 | 4.29 | 2+0 | 17/10/17/1 | 224 | – | 184.24 | 0.267 |
| gemini-nativ | ✓ | 0.956 | 0.250 | 0.704 | 3.35 | 2+0 | 19/11/10/1 | 212 | – | 87.17 | 0.112 |
| tesseract | ✓ | 0.982 | 0.146 | 0.722 | 5.56 | 0+0 | 0/0/0/0 | 302 | – | 135.13 | 0.000 |
| textlayer | ✓ | 1.000 | 1.000 | 1.000 | 1.00 | 0+0 | 0/0/0/0 | 35 | – | 0.07 | 0.000 |

### 14 — OCR-Ebene kaputt (eng-Modell, keine Umlaute; klassifiziert NATIVE)
| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling | ✓ | 0.931 | 0.976 | 0.829 | 0.96 | 0+0 | 0/4/0/0 | 543 | – | 302.93 | 0.000 |
| eigenbau | ✓ | 1.000 | 1.000 | 1.000 | 0.98 | 0+0 | 0/0/0/0 | 549 | – | 10.55 | 0.000 |
| gemini-nativ | ✓ | 0.913 | 0.947 | 0.912 | 0.97 | 0+0 | 2/0/0/0 | 583 | – | 125.86 | 0.224 |
| tesseract | ✓ | 0.960 | 0.964 | 0.955 | 0.98 | 0+0 | 0/0/0/0 | 586 | – | 100.74 | 0.000 |
| textlayer | ✓ | 1.000 | 1.000 | 1.000 | 1.00 | 0+0 | 0/0/0/0 | 549 | – | 0.12 | 0.000 |

## Judge-Rankings (Metrik c, Klassen ohne Gold)

- **02**: gemini-nativ > docling > textlayer > eigenbau
- **03**: eigenbau > gemini-nativ > docling > textlayer
- **04**: gemini-nativ > eigenbau > docling > textlayer
- **05**: gemini-nativ > tesseract > eigenbau > docling
- **06**: eigenbau > gemini-nativ > docling > tesseract
- **09A**: markitdown > docling > unstructured-neu > unstructured-pin
- **09B**: markitdown > docling > unstructured-neu > unstructured-pin
- **10**: trafilatura > unstructured-pin > unstructured-neu
- **11**: unstructured-pin > unstructured-neu
- **12**: gemini-nativ > docling > textlayer > eigenbau
- **13**: eigenbau > gemini-nativ > tesseract > textlayer > docling
- **14**: gemini-nativ > tesseract > textlayer > eigenbau > docling

## Kosten (LEDGER)

Summe: **8.366 USD** von 22.0 USD Deckel

- eigenbau: 0.887 USD
- gemini-cal-high: 0.052 USD
- gemini-cal-low: 0.039 USD
- gemini-cal-medium: 0.087 USD
- gemini-nativ: 7.300 USD
