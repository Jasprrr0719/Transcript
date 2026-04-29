# Meeting Transkription — Desktop App

Lokale Live-Transkription von **System-Audio** (Discord, YouTube, Spotify, Browser, …) und/oder **Mikrofon** mit Whisper. Komplett kostenlos, läuft ohne Internet auf deinem PC.

## Was die App kann

- 🔊 Nimmt **System-Sound** auf (alles was dein PC abspielt)
- 🎙️ Nimmt **Mikrofon** auf
- 🎚️ **Beides gleichzeitig** mischen
- 🤖 Transkribiert live mit faster-whisper auf deiner **NVIDIA-GPU**
- 🇩🇪🇬🇧 Deutsch & Englisch umschaltbar
- ⏱️ Mit Zeitstempeln
- 💾 Export als `.txt` oder `.md`

## Erste Installation

### Voraussetzungen
- Windows 10/11
- Python 3.9 oder neuer ([python.org](https://python.org)) — beim Installieren **"Add Python to PATH"** anhaken!
- NVIDIA-Treiber aktuell (für GPU-Beschleunigung)

### Schritte

1. **Doppelklick auf `install.bat`**
   Dauert beim ersten Mal 5–10 Minuten (lädt PyTorch + CUDA-Pakete).

2. **Doppelklick auf `start.bat`**
   Beim allerersten Start lädt die App das Whisper-Modell (~500 MB) — danach geht's direkt.

## Bedienung

1. **Audio-Quelle wählen**: System / Mikrofon / Beides
2. **Sprache wählen**: DE oder EN (vor dem Start)
3. **Aufnahme starten** klicken
4. Sprich oder lass abspielen — Text erscheint alle ~5 Sekunden
5. **Stoppen** wenn fertig
6. **Export** als Textdatei

## Modell anpassen

In `app.py` ganz oben:

```python
MODEL_SIZE = "small"   # "tiny", "base", "small", "medium", "large-v3"
```

- `tiny` / `base`: schnell, weniger genau
- `small`: guter Mittelweg ✅ Standard
- `medium`: genauer, braucht mehr GPU-Speicher
- `large-v3`: beste Qualität, ~3 GB GPU-RAM

## Fehlerbehebung

**"Python wurde nicht gefunden"**
→ Python neu installieren mit "Add to PATH" angehakt.

**"CUDA out of memory"**
→ Modell in `app.py` auf `"base"` oder `"tiny"` umstellen.

**Kein System-Audio aufgenommen**
→ Stelle sicher, dass beim Klick auf "Aufnahme starten" auch wirklich Audio abgespielt wird. Bei Stille wird nichts an Whisper geschickt.

**Mikrofon-Fehler**
→ Windows-Einstellungen → Datenschutz → Mikrofon → Apps erlauben.

## Rechtlicher Hinweis ⚠️

Das **heimliche Aufzeichnen** von Gesprächen ist in Deutschland nach § 201 StGB strafbar. Wenn du Discord-Calls oder Meetings transkribierst, **informiere die anderen Teilnehmer vorher** und hole ihr Einverständnis ein. Für eigene Notizen, YouTube-Videos, Vorträge etc. gibt's natürlich keine Probleme.
