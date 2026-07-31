# 11 — EML mit deutscher Zitatkette + Signatur

**Was ist hier schwierig?** Für dieses Format gibt es kein etabliertes Werkzeug im Feld —
der einzige ernsthafte Kandidat ist ein Einzelmaintainer-Projekt. Zu trennen sind hier:
die eigene Antwort oben, die eingebettete Signatur („Vielen Dank und schöne Grüße / Oliver
Gluth"), und darunter die zitierte Vorgeschichte, eingeleitet durch den deutschen
Outlook-Kopfblock:

```
Von: Polar Care Germany <kundenservice@polar.com>
Gesendet: Freitag, 8. Mai 2020 09:42
An: Oliver Gluth <oliver@smallpieces.de>
Betreff: [Polar Care] Re: AW: [Polar Care] Re: …
```

Die Verschachtelung steht zusätzlich im Betreff (`AW: … Re: AW: … Re: …`) — vier Ebenen.
Wer die Kette nicht abtrennt, zählt fremden Text als eigenen Inhalt.

- `polar-care_zitatkette.eml` — konvertiert, für Parser mit EML-Eingang
- `polar-care_zitatkette.msg` — Outlook-Original, falls der MIME-Aufbau relevant ist
- DE, 3 Grußformeln, 1 vollständiger Zitatkopf; Body 6.123 Zeichen im `.msg`,
  5.937 im konvertierten `.eml` (die Konvertierung normalisiert Zeilenenden)

**Zwei Abweichungen:**

1. **Kein Anhang.** Keine Mail in der Nextcloud hat deutsche Zitatkette + Signatur + Anhang
   und ist zugleich cloud-fähig. Die eine Mail, die alle drei hat, ist betrieblich und liegt
   als `../intern/B4_Mailkette_Re_2023-fea-0087.eml` (4 Zitatköpfe, 3 Grußformeln, Anhang).
2. **Zitatstil.** Es ist der Outlook-Kopfblock, nicht das `Am … schrieb …` von
   Thunderbird/Gmail. Beide Muster kommen im deutschen Schriftverkehr vor; ein Konverter
   sollte beide kennen. Das Strato-Postfach ist über den MCP-Connector nur lesbar — ein
   RFC822-Export für ein `Am … schrieb …`-Beispiel ist von hier aus nicht möglich.
