# corpus/bakeoff/harness/budget.py
"""Kostendeckel des Bake-offs — im Harness durchgesetzt, nicht per Disziplin.

Jeder bezahlte Modell-Call laeuft ueber ``Ledger``: ``precheck()`` VOR dem
Call wirft ``BudgetExceeded``, sobald die Summe den Deckel erreicht;
``record()`` danach schreibt die echten Zahlen aus ``usage_metadata`` ins
Ledger-File. Das Ledger ist kumulativ ueber alle Laeufe (dateibasiert), ein
abgebrochener Lauf verliert nichts.

Preise: Stand P1 sind die Gemini-Preise **Platzhalter mit Sicherheitsaufschlag**
(``verified: False``) — deutlich ueber den letzten bekannten 2.5-flash-Preisen.
Vor den breiten P2-Laeufen werden sie gegen die offizielle Preisliste
verifiziert; der Deckel greift bis dahin auf den konservativen Werten.
Die Token-Zaehlung selbst ist exakt (``usage_metadata``), nur der Preis
pro Token ist bis zur Verifikation eine obere Schranke.
"""

import json
import time
from pathlib import Path

# Harte Obergrenze: 20 EUR (Sprint-Vorgabe), konservativ in USD umgerechnet.
CAP_EUR = 20.0
EUR_USD = 1.10  # dokumentierte, bewusst grosszuegige Umrechnung
CAP_USD = CAP_EUR * EUR_USD

# USD pro 1M Tokens. Konservative obere Schranke, bis verifiziert (P2).
PRICES = {
    "gemini-3.6-flash": {"in": 0.60, "out": 3.50, "verified": False},
    "_default": {"in": 1.00, "out": 5.00, "verified": False},
}


class BudgetExceeded(RuntimeError):
    pass


def price_for(model: str) -> dict:
    for key, p in PRICES.items():
        if key != "_default" and key in (model or ""):
            return p
    return PRICES["_default"]


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = price_for(model)
    return tokens_in / 1e6 * p["in"] + tokens_out / 1e6 * p["out"]


class Ledger:
    def __init__(self, path: Path, cap_usd: float = CAP_USD):
        self.path = Path(path)
        self.cap_usd = cap_usd
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"cap_usd": cap_usd, "cap_eur": CAP_EUR, "entries": []}

    def spent(self) -> float:
        return sum(e["cost_usd"] for e in self.data["entries"])

    def precheck(self):
        if self.spent() >= self.cap_usd:
            raise BudgetExceeded(
                f"Kostendeckel erreicht: {self.spent():.2f} USD >= {self.cap_usd:.2f} USD"
                " — Lauf abgebrochen, fehlende Kandidaten/Klassen im Bericht nennen."
            )

    def record(self, candidate: str, class_id: str, model: str,
               tokens_in: int, tokens_out: int, note: str = "") -> float:
        cost = estimate_cost(model, tokens_in, tokens_out)
        self.data["entries"].append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "candidate": candidate,
            "class": class_id,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost, 6),
            "price_verified": price_for(model)["verified"],
            "note": note,
        })
        self._save()
        return cost

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        tmp.replace(self.path)
