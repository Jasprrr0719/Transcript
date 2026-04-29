"""
Meeting Transkription — Desktop-App
Nimmt System-Audio (Discord, Spotify, Browser, etc.) UND/ODER Mikrofon auf
und transkribiert live mit faster-whisper (lokal, kostenlos, ohne Internet).
"""

import os
import sys
import time
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc
import customtkinter as ctk
from tkinter import filedialog, messagebox

# ============================================================
# KONFIGURATION
# ============================================================
SAMPLE_RATE = 16000           # Whisper erwartet 16kHz
CHUNK_SECONDS = 5             # Alle 5 Sekunden ein Chunk
MIN_AUDIO_LEVEL = 0.005       # Minimaler RMS-Pegel, damit's transkribiert wird (Stille filtern)

# Whisper-Modell-Größe — "small" ist guter Mittelweg auf NVIDIA GPU
# Optionen: "tiny", "base", "small", "medium", "large-v3"
MODEL_SIZE = "small"

# ============================================================
# THEME
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT = "#d4ff3a"
ACCENT_DARK = "#0e0e10"
RED = "#ff4d5e"
DIM = "#8a8a96"


class TranscriptionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Meeting Transkription")
        self.geometry("1000x720")
        self.configure(fg_color="#0e0e10")

        # State
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.entries = []           # list of (timestamp_str, text, elapsed_sec)
        self.recording_start = None
        self.model = None
        self.language = "de"
        self.source_mode = "both"   # "system", "mic", "both"

        # Threads
        self.capture_thread = None
        self.transcribe_thread = None
        self.stop_event = threading.Event()

        self._build_ui()
        self._update_timer()

        # Modell im Hintergrund laden
        self.after(100, self._load_model_async)

    # ============================================================
    # UI AUFBAU
    # ============================================================
    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent", height=80)
        header.pack(fill="x", padx=24, pady=(20, 0))

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left")

        self.indicator = ctk.CTkLabel(
            title_frame, text="●", text_color=ACCENT,
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.indicator.pack(side="left", padx=(0, 10))

        title_text = ctk.CTkLabel(
            title_frame, text="Meeting Transkription",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold")
        )
        title_text.pack(side="left")

        # Sprache rechts
        self.lang_var = ctk.StringVar(value="DE")
        lang_frame = ctk.CTkFrame(header, fg_color="#16161a", corner_radius=20)
        lang_frame.pack(side="right")

        self.lang_de_btn = ctk.CTkButton(
            lang_frame, text="DE", width=50, height=30, corner_radius=15,
            fg_color=ACCENT, text_color=ACCENT_DARK,
            command=lambda: self._set_language("de"),
            font=ctk.CTkFont(weight="bold")
        )
        self.lang_de_btn.pack(side="left", padx=4, pady=4)

        self.lang_en_btn = ctk.CTkButton(
            lang_frame, text="EN", width=50, height=30, corner_radius=15,
            fg_color="transparent", text_color=DIM,
            command=lambda: self._set_language("en"),
            font=ctk.CTkFont(weight="bold")
        )
        self.lang_en_btn.pack(side="left", padx=4, pady=4)

        # Untertitel
        subtitle = ctk.CTkLabel(
            self, text="Lokale Transkription mit Whisper · keine Daten verlassen deinen PC",
            text_color=DIM, font=ctk.CTkFont(size=12)
        )
        subtitle.pack(anchor="w", padx=28, pady=(0, 16))

        # Audio-Quelle Auswahl
        source_frame = ctk.CTkFrame(self, fg_color="#16161a", corner_radius=12)
        source_frame.pack(fill="x", padx=24, pady=(0, 12))

        ctk.CTkLabel(
            source_frame, text="AUDIO-QUELLE",
            text_color=DIM, font=ctk.CTkFont(size=10, weight="bold")
        ).pack(anchor="w", padx=16, pady=(12, 4))

        radio_frame = ctk.CTkFrame(source_frame, fg_color="transparent")
        radio_frame.pack(fill="x", padx=12, pady=(0, 12))

        self.source_var = ctk.StringVar(value="both")
        for value, label in [
            ("system", "🔊  System-Sound (Discord, YouTube, Spotify, …)"),
            ("mic", "🎙️  Mikrofon"),
            ("both", "🔊 + 🎙️  Beides gleichzeitig"),
        ]:
            rb = ctk.CTkRadioButton(
                radio_frame, text=label, variable=self.source_var, value=value,
                fg_color=ACCENT, hover_color=ACCENT, border_color=DIM,
                font=ctk.CTkFont(size=12),
                command=self._update_source
            )
            rb.pack(anchor="w", padx=8, pady=4)

        # Steuerung
        controls = ctk.CTkFrame(self, fg_color="#16161a", corner_radius=12, height=70)
        controls.pack(fill="x", padx=24, pady=(0, 12))

        self.start_btn = ctk.CTkButton(
            controls, text="●  Aufnahme starten", width=180, height=44,
            fg_color=ACCENT, text_color=ACCENT_DARK, hover_color="#b8e030",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.start_recording
        )
        self.start_btn.pack(side="left", padx=12, pady=12)

        self.stop_btn = ctk.CTkButton(
            controls, text="■  Stoppen", width=120, height=44,
            fg_color=RED, text_color="white", hover_color="#e63d4e",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.stop_recording, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=4, pady=12)

        self.clear_btn = ctk.CTkButton(
            controls, text="Löschen", width=90, height=44,
            fg_color="transparent", text_color=DIM, border_width=1, border_color="#2a2a32",
            hover_color="#1c1c21",
            command=self.clear_transcript
        )
        self.clear_btn.pack(side="left", padx=4, pady=12)

        self.export_txt_btn = ctk.CTkButton(
            controls, text="↓ .txt", width=70, height=44,
            fg_color="transparent", text_color=DIM, border_width=1, border_color="#2a2a32",
            hover_color="#1c1c21",
            command=lambda: self.export("txt"), state="disabled"
        )
        self.export_txt_btn.pack(side="left", padx=4, pady=12)

        self.export_md_btn = ctk.CTkButton(
            controls, text="↓ .md", width=70, height=44,
            fg_color="transparent", text_color=DIM, border_width=1, border_color="#2a2a32",
            hover_color="#1c1c21",
            command=lambda: self.export("md"), state="disabled"
        )
        self.export_md_btn.pack(side="left", padx=4, pady=12)

        # Status rechts
        status_frame = ctk.CTkFrame(controls, fg_color="transparent")
        status_frame.pack(side="right", padx=12)

        self.timer_label = ctk.CTkLabel(
            status_frame, text="00:00:00",
            font=ctk.CTkFont(family="Consolas", size=16, weight="bold")
        )
        self.timer_label.pack()

        self.status_label = ctk.CTkLabel(
            status_frame, text="Lade Whisper-Modell...",
            text_color=DIM, font=ctk.CTkFont(size=11)
        )
        self.status_label.pack()

        # Transkript-Bereich
        transcript_container = ctk.CTkFrame(self, fg_color="#16161a", corner_radius=12)
        transcript_container.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        # Header der Transkript-Box
        t_header = ctk.CTkFrame(transcript_container, fg_color="transparent", height=40)
        t_header.pack(fill="x", padx=20, pady=(12, 0))

        ctk.CTkLabel(
            t_header, text="— TRANSKRIPT",
            text_color=DIM, font=ctk.CTkFont(size=10, weight="bold")
        ).pack(side="left")

        self.word_count_label = ctk.CTkLabel(
            t_header, text="0 Wörter",
            text_color=DIM, font=ctk.CTkFont(size=10)
        )
        self.word_count_label.pack(side="right")

        # Scrollbarer Textbereich
        self.transcript_text = ctk.CTkTextbox(
            transcript_container,
            fg_color="#16161a", text_color="#ededf0",
            font=ctk.CTkFont(family="Georgia", size=15),
            wrap="word", border_width=0,
            scrollbar_button_color="#2a2a32"
        )
        self.transcript_text.pack(fill="both", expand=True, padx=20, pady=(8, 20))

        self.transcript_text.tag_config("timestamp", foreground=DIM)
        self.transcript_text.tag_config("text", foreground="#ededf0")
        self.transcript_text.tag_config("empty", foreground=DIM)

        self._show_empty_state()

    def _show_empty_state(self):
        self.transcript_text.configure(state="normal")
        self.transcript_text.delete("1.0", "end")
        self.transcript_text.insert("end", "\n\n        ◌\n\n        Wähle eine Audio-Quelle und drücke „Aufnahme starten“.\n", "empty")
        self.transcript_text.configure(state="disabled")

    # ============================================================
    # MODELL LADEN
    # ============================================================
    def _load_model_async(self):
        thread = threading.Thread(target=self._load_model, daemon=True)
        thread.start()

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel

            # Versuche GPU mit CUDA
            try:
                self.model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
                device_info = "GPU (CUDA)"
            except Exception as e:
                print(f"GPU nicht verfügbar, fallback auf CPU: {e}")
                self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
                device_info = "CPU"

            self.after(0, lambda: self.status_label.configure(
                text=f"Bereit · Modell: {MODEL_SIZE} · {device_info}"
            ))
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self.status_label.configure(
                text=f"Fehler beim Laden: {err_msg[:60]}"
            ))
            self.after(0, lambda: messagebox.showerror(
                "Modell konnte nicht geladen werden",
                f"Fehler:\n\n{err_msg}\n\nBitte prüfe, ob alle Pakete installiert sind."
            ))

    # ============================================================
    # SPRACHE & QUELLE
    # ============================================================
    def _set_language(self, lang):
        if self.is_recording:
            return
        self.language = lang
        if lang == "de":
            self.lang_de_btn.configure(fg_color=ACCENT, text_color=ACCENT_DARK)
            self.lang_en_btn.configure(fg_color="transparent", text_color=DIM)
        else:
            self.lang_en_btn.configure(fg_color=ACCENT, text_color=ACCENT_DARK)
            self.lang_de_btn.configure(fg_color="transparent", text_color=DIM)

    def _update_source(self):
        self.source_mode = self.source_var.get()

    # ============================================================
    # AUFNAHME-LOGIK
    # ============================================================
    def start_recording(self):
        if self.is_recording:
            return
        if self.model is None:
            messagebox.showwarning("Modell wird noch geladen",
                                   "Bitte warte, bis das Whisper-Modell geladen ist.")
            return

        self.is_recording = True
        self.recording_start = time.time()
        self.stop_event.clear()
        self.audio_queue = queue.Queue()

        # Beim ersten Mal Empty-State entfernen
        if not self.entries:
            self.transcript_text.configure(state="normal")
            self.transcript_text.delete("1.0", "end")
            self.transcript_text.configure(state="disabled")

        # Threads starten
        self.capture_thread = threading.Thread(target=self._capture_audio, daemon=True)
        self.transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.capture_thread.start()
        self.transcribe_thread.start()

        # UI aktualisieren
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.indicator.configure(text_color=RED)
        source_name = {"system": "System-Audio", "mic": "Mikrofon", "both": "System + Mikrofon"}[self.source_mode]
        self.status_label.configure(text=f"● Aufnahme läuft · {source_name}")

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.stop_event.set()

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.indicator.configure(text_color=ACCENT)
        self.status_label.configure(text="Verarbeite letzten Chunk...")

        # Warte kurz, dann zeige Bereit
        self.after(1500, lambda: self.status_label.configure(text="Gestoppt"))

    # ============================================================
    # AUDIO CAPTURE (System + Mic)
    # ============================================================
    def _capture_audio(self):
        """Nimmt Audio in CHUNK_SECONDS-Blöcken auf und schiebt sie in die Queue."""
        try:
            chunk_frames = SAMPLE_RATE * CHUNK_SECONDS

            # Quellen vorbereiten
            sources = []
            if self.source_mode in ("system", "both"):
                # Default-Speaker als Loopback (System-Audio)
                default_speaker = sc.default_speaker()
                loopback_mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
                sources.append(("system", loopback_mic))

            if self.source_mode in ("mic", "both"):
                default_mic = sc.default_microphone()
                sources.append(("mic", default_mic))

            # Recorder öffnen
            recorders = []
            for name, src in sources:
                rec = src.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=1024)
                rec.__enter__()
                recorders.append((name, rec, src))

            try:
                while not self.stop_event.is_set():
                    # Pro Quelle einen Chunk lesen, dann mischen
                    chunks = []
                    for name, rec, src in recorders:
                        data = rec.record(numframes=chunk_frames)
                        if data.ndim > 1:
                            data = data.mean(axis=1)  # zu mono
                        chunks.append(data.astype(np.float32))

                    # Quellen mischen (einfach addieren, mit Soft-Limiter)
                    if len(chunks) == 1:
                        mixed = chunks[0]
                    else:
                        # Auf gleiche Länge bringen
                        min_len = min(len(c) for c in chunks)
                        mixed = sum(c[:min_len] for c in chunks)
                        # Soft-Clipping
                        mixed = np.tanh(mixed)

                    # Pegel prüfen
                    rms = np.sqrt(np.mean(mixed ** 2))
                    if rms > MIN_AUDIO_LEVEL:
                        elapsed = time.time() - self.recording_start
                        self.audio_queue.put((elapsed, mixed))

            finally:
                for _, rec, _ in recorders:
                    try:
                        rec.__exit__(None, None, None)
                    except Exception:
                        pass

        except Exception as e:
            err = str(e)
            print(f"Capture-Fehler: {err}")
            self.after(0, lambda: self.status_label.configure(text=f"Audio-Fehler: {err[:60]}"))
            self.after(0, lambda: messagebox.showerror(
                "Audio-Aufnahme fehlgeschlagen",
                f"Fehler bei der Audio-Aufnahme:\n\n{err}\n\n"
                "Mögliche Ursachen:\n"
                "• Kein Mikrofon angeschlossen\n"
                "• Audio-Treiber-Problem\n"
                "• Andere App blockiert das Audio"
            ))
            self.is_recording = False

    # ============================================================
    # TRANSKRIPTION
    # ============================================================
    def _transcribe_loop(self):
        """Holt Audio-Chunks aus der Queue und transkribiert sie."""
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                elapsed, audio = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                segments, info = self.model.transcribe(
                    audio,
                    language=self.language,
                    beam_size=1,             # schneller; 5 wäre genauer
                    vad_filter=True,         # filtert Stille / Atmen
                    vad_parameters=dict(min_silence_duration_ms=500),
                )

                full_text = " ".join(seg.text.strip() for seg in segments).strip()

                if full_text:
                    timestamp = self._format_timestamp(elapsed)
                    self.after(0, lambda t=timestamp, txt=full_text, e=elapsed:
                               self._add_entry(t, txt, e))
            except Exception as e:
                print(f"Transkriptions-Fehler: {e}")

    def _format_timestamp(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _add_entry(self, timestamp, text, elapsed_sec):
        # Großschreibung am Anfang
        if text:
            text = text[0].upper() + text[1:]

        self.entries.append((timestamp, text, elapsed_sec))

        self.transcript_text.configure(state="normal")
        self.transcript_text.insert("end", f"[{timestamp}]  ", "timestamp")
        self.transcript_text.insert("end", f"{text}\n\n", "text")
        self.transcript_text.see("end")
        self.transcript_text.configure(state="disabled")

        # Wortzähler updaten
        total_words = sum(len(e[1].split()) for e in self.entries)
        self.word_count_label.configure(text=f"{total_words} Wörter")

        # Export-Buttons aktivieren
        self.export_txt_btn.configure(state="normal")
        self.export_md_btn.configure(state="normal")

    # ============================================================
    # TIMER
    # ============================================================
    def _update_timer(self):
        if self.is_recording and self.recording_start:
            elapsed = time.time() - self.recording_start
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.timer_label.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.after(250, self._update_timer)

    # ============================================================
    # LÖSCHEN & EXPORT
    # ============================================================
    def clear_transcript(self):
        if self.is_recording:
            if not messagebox.askyesno("Aufnahme läuft", "Aufnahme läuft. Wirklich alles löschen?"):
                return
            self.stop_recording()

        self.entries = []
        self.recording_start = None
        self.timer_label.configure(text="00:00:00")
        self.word_count_label.configure(text="0 Wörter")
        self.export_txt_btn.configure(state="disabled")
        self.export_md_btn.configure(state="disabled")
        self._show_empty_state()
        self.status_label.configure(text="Bereit")

    def export(self, fmt):
        if not self.entries:
            return

        now = datetime.now()
        default_name = f"meeting_{now.strftime('%Y-%m-%d_%H%M')}.{fmt}"

        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            initialfile=default_name,
            filetypes=[(f"{fmt.upper()} Datei", f"*.{fmt}")]
        )
        if not filepath:
            return

        if fmt == "txt":
            content = self._build_txt(now)
        else:
            content = self._build_md(now)

        try:
            Path(filepath).write_text(content, encoding="utf-8")
            self.status_label.configure(text=f"Exportiert: {Path(filepath).name}")
        except Exception as e:
            messagebox.showerror("Export fehlgeschlagen", str(e))

    def _build_txt(self, now):
        lang_name = "Deutsch" if self.language == "de" else "Englisch"
        out = "Meeting Transkription\n"
        out += f"{now.strftime('%d.%m.%Y %H:%M')}\n"
        out += f"Sprache: {lang_name}\n"
        out += "=" * 50 + "\n\n"
        for ts, text, _ in self.entries:
            out += f"[{ts}] {text}\n\n"
        return out

    def _build_md(self, now):
        lang_name = "Deutsch" if self.language == "de" else "Englisch"
        last_ts = self.entries[-1][0] if self.entries else "–"
        out = "# Meeting Transkription\n\n"
        out += f"**Datum:** {now.strftime('%d.%m.%Y %H:%M')}  \n"
        out += f"**Sprache:** {lang_name}  \n"
        out += f"**Dauer:** {last_ts}  \n"
        out += f"**Quelle:** {self.source_mode}\n\n"
        out += "---\n\n"
        for ts, text, _ in self.entries:
            out += f"**[{ts}]** {text}\n\n"
        return out


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    app = TranscriptionApp()

    def on_close():
        if app.is_recording:
            if not messagebox.askyesno("Aufnahme läuft", "Aufnahme läuft. Wirklich beenden?"):
                return
            app.stop_recording()
            time.sleep(0.5)
        app.destroy()
        sys.exit(0)

    app.protocol("WM_DELETE_WINDOW", on_close)
    app.mainloop()
