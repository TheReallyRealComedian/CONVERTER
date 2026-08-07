# Metrik (c) — LLM-Judge-Rubrik für Klassen ohne Gold

**Wann**: Phase 2/3, für die 11 Klassen ohne Gold-Fassung (alle außer 01/07/08).
**Warum ein Judge**: TEDS korreliert nur r=0,684 mit menschlichem Urteil, ein
LLM-Judge 0,93 (Sprint-Prompt). **Pflicht dazu**: eine Stichprobe pro Kandidat
selbst in Augenschein nehmen und Abweichungen zwischen Judge und eigenem
Eindruck im Bericht benennen.

**Bias-Notiz (offenzulegen, nicht wegzuoptimieren)**: Wenn der Judge aus
derselben Modellfamilie stammt wie ein Kandidat (z. B. Gemini bewertet
Gemini-Output), ist Selbst-Präferenz möglich. Deshalb: Judge-Familie ≠
stärkster Kandidat, wo machbar (Claude-Subagenten bewerten Gemini-Output),
und die Familie des Judges steht im Ergebnis-Doc.

## Eingaben pro Urteil

1. Die Original-Seiten als Bilder (gerendert, 150 dpi reicht; bei langen
   Dokumenten eine dokumentierte Seiten-Stichprobe: erste, letzte, plus 3
   strukturreiche Seiten — dieselbe Stichprobe für alle Kandidaten).
2. Das Kandidaten-Markdown (derselbe Ausschnitt).

## Urteil (JSON, pro Dokument × Kandidat)

```json
{
  "klasse": "…", "kandidat": "…",
  "treue": 1-5,            // Wortlaut korrekt? Umlaute? Zahlen? KEINE Reparaturen?
  "vollstaendigkeit": 1-5, // fehlt Inhalt (Folien, Notes, SmartArt, Fußnoten, Zitatkette)?
  "struktur": 1-5,         // Tabellen als Tabellen, Überschriften, Listen, verbundene Zellen
  "lesereihenfolge": 1-5,  // Spalten, Folienreihenfolge, Zitat-Verschachtelung
  "halluzination": 1-5,    // 5 = nichts erfunden; ausgefüllte Blanko-Felder = 1
  "befunde": ["konkret, mit Zitat aus Output und Stelle im Original"],
  "gesamturteil": "brauchbar | brauchbar mit Nacharbeit | unbrauchbar"
}
```

**Skalen-Anker** (damit 3 überall dasselbe heißt): 5 = ein Korrektor findet
nichts Wesentliches · 4 = Kleinigkeiten, kein Inhaltsverlust · 3 = brauchbar,
aber spürbare Nacharbeit · 2 = Inhalts-/Strukturverlust, der den Zweck
gefährdet · 1 = irreführend oder leer.

**Regeln**: Quell-Eigenheiten (Tippfehler des Originals) sind KEINE Fehler des
Kandidaten — Reparaturen sind es. Blanko-Formulare müssen blanko bleiben.
Für Klasse 14 gilt das Klassen-README: durchgereichte kaputte Textebene ohne
Meldung = durchgefallen, egal wie „sauber" der Output aussieht.
