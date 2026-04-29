"""
Meeting Transkription — Desktop-App v2 "Aurora"
Lokale Live-Transkription mit faster-whisper.
Features: System+Mic Audio, Live-Visualizer, Bookmarks, Auto-Save,
Mini-Modus, Quellen-Tags, Pausen-Erkennung, Dark/Light Toggle.
"""

import os
import sys
import time
import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc
import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas

# ============================================================
# KONFIGURATION
# ============================================================
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
MIN_AUDIO_LEVEL = 0.005
PAUSE_THRESHOLD_SEC = 8     # Längere Lücke zwischen Einträgen = visueller Bruch
AUTOSAVE_INTERVAL_SEC = 30  # Auto-Save alle 30 Sekunden
MODEL_SIZE = "small"
APP_DIR = Path.home() / "MeetingTranskription"
APP_DIR.mkdir(exist_ok=True)
AUTOSAVE_PATH = APP_DIR / "autosave.json"

# ============================================================
# THEMES
# ============================================================
THEMES = {
    "dark": {
        "bg":        "#0a0a0f",
        "bg_elev":   "#13131a",
        "bg_card":   "#1a1a24",
        "bg_hover":  "#22222e",
        "border":    "#2a2a38",
        "text":      "#ededf0",
        "text_dim":  "#7a7a8a",
        "text_mute": "#4a4a58",
        "accent":    "#5eead4",   # Aurora-Türkis
        "accent2":   "#a78bfa",   # Aurora-Violett
        "accent3":   "#f0abfc",   # Aurora-Pink
        "warm":      "#fbbf77",
        "red":       "#fb7185",
        "green":     "#5eead4",
        "viz":       "#5eead4",
        "viz2":      "#a78bfa",
    },
    "light": {
        "bg":        "#faf8f5",   # cremiges Off-White
        "bg_elev":   "#ffffff",
        "bg_card":   "#f5f2ec",
        "bg_hover":  "#ece8e0",
        "border":    "#e5e0d6",
        "text":      "#1a1a24",
        "text_dim":  "#6a6a78",
        "text_mute": "#a0a0ac",
        "accent":    "#0d9488",
        "accent2":   "#7c3aed",
        "accent3":   "#c026d3",
        "warm":      "#d97706",
        "red":       "#e11d48",
        "green":     "#0d9488",
        "viz":       "#0d9488",
        "viz2":      "#7c3aed",
    }
}


class AudioVisualizer(ctk.CTkFrame):
    """Live-Pegel-Visualizer mit Aurora-Style Bars."""
    def __init__(self, parent, theme, num_bars=40, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.theme = theme
        self.num_bars = num_bars
        self.levels = [0.0] * num_bars
        self.target_levels = [0.0] * num_bars

        self.canvas = Canvas(
            self, bg=theme["bg_elev"], highlightthickness=0, height=60
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw())

    def update_theme(self, theme):
        self.theme = theme
        self.canvas.configure(bg=theme["bg_elev"])
        self._draw()

    def push_level(self, level):
        """level: 0.0 - 1.0, neuer Pegelwert"""
        # Shift left, add new
        self.target_levels = self.target_levels[1:] + [min(1.0, max(0.0, level))]

    def animate(self):
        # Smooth interpolation zu Zielwerten + leichter Decay
        for i in range(self.num_bars):
            diff = self.target_levels[i] - self.levels[i]
            self.levels[i] += diff * 0.3
            self.target_levels[i] *= 0.85  # Decay
        self._draw()

    def _draw(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        bar_w = w / self.num_bars
        gap = max(1, bar_w * 0.25)
        actual_w = bar_w - gap
        cy = h / 2

        for i, level in enumerate(self.levels):
            x = i * bar_w + gap / 2
            bar_h = max(2, level * (h * 0.85))

            # Farb-Mix: links Türkis -> rechts Violett
            t = i / max(1, self.num_bars - 1)
            color = self._mix_color(self.theme["viz"], self.theme["viz2"], t)

            # Mit Alpha simulieren über Helligkeit für gestoppte Bars
            if level < 0.05:
                color = self.theme["text_mute"]

            self.canvas.create_rectangle(
                x, cy - bar_h / 2, x + actual_w, cy + bar_h / 2,
                fill=color, outline="", width=0
            )

    @staticmethod
    def _mix_color(c1, c2, t):
        """Linear-Interpolation zwischen zwei Hex-Farben."""
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"


class TranscriptionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # State
        self.theme_name = "dark"
        self.theme = THEMES[self.theme_name]
        self.is_recording = False
        self.is_mini = False
        self.audio_queue = queue.Queue()
        self.entries = []  # list of dicts: {ts, text, source, elapsed, bookmark}
        self.recording_start = None
        self.last_entry_elapsed = 0
        self.model = None
        self.language = "de"
        self.source_mode = "both"
        self.last_autosave = 0

        # Threads
        self.capture_thread = None
        self.transcribe_thread = None
        self.stop_event = threading.Event()

        # Window setup
        ctk.set_appearance_mode("dark")
        self.title("Meeting Transkription — Aurora")
        self.geometry("1100x780")
        self.minsize(900, 600)
        self.configure(fg_color=self.theme["bg"])

        self._build_ui()
        self._update_timer()
        self._animate_visualizer()
        self._autosave_loop()
        self._bind_hotkeys()

        self.after(100, self._load_model_async)
        self.after(500, self._check_autosave_recovery)

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        t = self.theme

        # ============ HEADER ============
        self.header = ctk.CTkFrame(self, fg_color="transparent", height=90)
        self.header.pack(fill="x", padx=28, pady=(20, 0))

        # Brand
        brand = ctk.CTkFrame(self.header, fg_color="transparent")
        brand.pack(side="left")

        # Aurora-Glow Indikator
        self.indicator_canvas = Canvas(brand, width=24, height=24,
                                       bg=t["bg"], highlightthickness=0)
        self.indicator_canvas.pack(side="left", padx=(0, 12), pady=(8, 0))
        self._draw_indicator()

        title_box = ctk.CTkFrame(brand, fg_color="transparent")
        title_box.pack(side="left")

        self.title_label = ctk.CTkLabel(
            title_box, text="Meeting Transkription",
            text_color=t["text"],
            font=ctk.CTkFont(family="Georgia", size=26, weight="bold")
        )
        self.title_label.pack(anchor="w")

        self.subtitle_label = ctk.CTkLabel(
            title_box, text="Aurora · lokal · privat",
            text_color=t["text_dim"],
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.subtitle_label.pack(anchor="w")

        # Header rechts: Theme + Lang + Mini
        header_right = ctk.CTkFrame(self.header, fg_color="transparent")
        header_right.pack(side="right")

        self.mini_btn = ctk.CTkButton(
            header_right, text="◱  Mini", width=70, height=32,
            fg_color="transparent", text_color=t["text_dim"],
            border_width=1, border_color=t["border"],
            hover_color=t["bg_hover"],
            command=self._toggle_mini,
            font=ctk.CTkFont(size=11)
        )
        self.mini_btn.pack(side="right", padx=(8, 0), pady=(8, 0))

        self.theme_btn = ctk.CTkButton(
            header_right, text="☾", width=40, height=32,
            fg_color="transparent", text_color=t["text_dim"],
            border_width=1, border_color=t["border"],
            hover_color=t["bg_hover"],
            command=self._toggle_theme,
            font=ctk.CTkFont(size=14)
        )
        self.theme_btn.pack(side="right", padx=(8, 0), pady=(8, 0))

        self.lang_frame = ctk.CTkFrame(header_right, fg_color=t["bg_elev"], corner_radius=18)
        self.lang_frame.pack(side="right", pady=(8, 0))

        self.lang_de_btn = ctk.CTkButton(
            self.lang_frame, text="DE", width=44, height=26, corner_radius=13,
            fg_color=t["accent"], text_color=t["bg"],
            command=lambda: self._set_language("de"),
            font=ctk.CTkFont(weight="bold", size=11)
        )
        self.lang_de_btn.pack(side="left", padx=3, pady=3)

        self.lang_en_btn = ctk.CTkButton(
            self.lang_frame, text="EN", width=44, height=26, corner_radius=13,
            fg_color="transparent", text_color=t["text_dim"],
            command=lambda: self._set_language("en"),
            font=ctk.CTkFont(weight="bold", size=11)
        )
        self.lang_en_btn.pack(side="left", padx=3, pady=3)

        # ============ VISUALIZER ============
        viz_container = ctk.CTkFrame(self, fg_color=t["bg_elev"], corner_radius=14)
        viz_container.pack(fill="x", padx=28, pady=(16, 0))

        viz_inner = ctk.CTkFrame(viz_container, fg_color="transparent")
        viz_inner.pack(fill="x", padx=16, pady=12)

        self.visualizer = AudioVisualizer(viz_inner, t, num_bars=48, height=56)
        self.visualizer.pack(fill="x")

        # ============ KONTROLLEN ============
        self.controls = ctk.CTkFrame(self, fg_color=t["bg_elev"], corner_radius=14)
        self.controls.pack(fill="x", padx=28, pady=(12, 0))

        # Source Selector (links)
        source_box = ctk.CTkFrame(self.controls, fg_color="transparent")
        source_box.pack(side="left", padx=16, pady=14)

        ctk.CTkLabel(
            source_box, text="QUELLE", text_color=t["text_dim"],
            font=ctk.CTkFont(family="Consolas", size=9, weight="bold")
        ).pack(anchor="w")

        self.source_var = ctk.StringVar(value="both")
        source_menu_frame = ctk.CTkFrame(source_box, fg_color="transparent")
        source_menu_frame.pack(anchor="w", pady=(4, 0))

        self.source_buttons = {}
        for value, label in [("system", "🔊 System"), ("mic", "🎙 Mic"), ("both", "🔊+🎙 Beides")]:
            b = ctk.CTkButton(
                source_menu_frame, text=label, width=80, height=30,
                command=lambda v=value: self._set_source(v),
                font=ctk.CTkFont(size=11),
                corner_radius=8
            )
            b.pack(side="left", padx=(0, 4))
            self.source_buttons[value] = b
        self._update_source_buttons()

        # Buttons (mitte)
        btn_box = ctk.CTkFrame(self.controls, fg_color="transparent")
        btn_box.pack(side="left", padx=12, pady=14)

        self.start_btn = ctk.CTkButton(
            btn_box, text="●  Aufnahme", width=140, height=42,
            fg_color=t["accent"], text_color=t["bg"], hover_color=t["accent2"],
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=10,
            command=self.start_recording
        )
        self.start_btn.pack(side="left", padx=(0, 6))

        self.stop_btn = ctk.CTkButton(
            btn_box, text="■", width=42, height=42,
            fg_color=t["red"], text_color="white", hover_color=t["red"],
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=10,
            command=self.stop_recording, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(0, 6))

        self.bookmark_btn = ctk.CTkButton(
            btn_box, text="✦  Bookmark", width=110, height=42,
            fg_color="transparent", text_color=t["accent3"],
            border_width=1, border_color=t["accent3"],
            hover_color=t["bg_hover"],
            font=ctk.CTkFont(size=11, weight="bold"),
            corner_radius=10,
            command=self.add_bookmark, state="disabled"
        )
        self.bookmark_btn.pack(side="left", padx=(0, 6))

        # Status (rechts)
        status_box = ctk.CTkFrame(self.controls, fg_color="transparent")
        status_box.pack(side="right", padx=16, pady=14)

        self.timer_label = ctk.CTkLabel(
            status_box, text="00:00:00",
            text_color=t["text"],
            font=ctk.CTkFont(family="Consolas", size=20, weight="bold")
        )
        self.timer_label.pack(anchor="e")

        self.status_label = ctk.CTkLabel(
            status_box, text="Lade Whisper-Modell…",
            text_color=t["text_dim"],
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.status_label.pack(anchor="e")

        # ============ AKTIONEN ============
        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=28, pady=(10, 0))

        self.clear_btn = ctk.CTkButton(
            actions, text="Löschen", width=80, height=30,
            fg_color="transparent", text_color=t["text_dim"],
            border_width=1, border_color=t["border"],
            hover_color=t["bg_hover"],
            font=ctk.CTkFont(size=10),
            command=self.clear_transcript
        )
        self.clear_btn.pack(side="left", padx=(0, 6))

        self.export_txt_btn = ctk.CTkButton(
            actions, text="↓ .txt", width=70, height=30,
            fg_color="transparent", text_color=t["text_dim"],
            border_width=1, border_color=t["border"],
            hover_color=t["bg_hover"],
            font=ctk.CTkFont(size=10),
            command=lambda: self.export("txt"), state="disabled"
        )
        self.export_txt_btn.pack(side="left", padx=(0, 6))

        self.export_md_btn = ctk.CTkButton(
            actions, text="↓ .md", width=70, height=30,
            fg_color="transparent", text_color=t["text_dim"],
            border_width=1, border_color=t["border"],
            hover_color=t["bg_hover"],
            font=ctk.CTkFont(size=10),
            command=lambda: self.export("md"), state="disabled"
        )
        self.export_md_btn.pack(side="left", padx=(0, 6))

        self.hotkey_hint = ctk.CTkLabel(
            actions,
            text="⌨  Strg+R Start/Stopp · F2 Bookmark",
            text_color=t["text_mute"],
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.hotkey_hint.pack(side="right")

        # ============ TRANSKRIPT ============
        self.transcript_container = ctk.CTkFrame(self, fg_color=t["bg_elev"], corner_radius=14)
        self.transcript_container.pack(fill="both", expand=True, padx=28, pady=(12, 24))

        t_header = ctk.CTkFrame(self.transcript_container, fg_color="transparent")
        t_header.pack(fill="x", padx=24, pady=(16, 0))

        self.transcript_title = ctk.CTkLabel(
            t_header, text="—  TRANSKRIPT", text_color=t["text_dim"],
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold")
        )
        self.transcript_title.pack(side="left")

        self.word_count_label = ctk.CTkLabel(
            t_header, text="0 Wörter · 0 Bookmarks",
            text_color=t["text_dim"],
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.word_count_label.pack(side="right")

        # Scrollable frame statt Textbox — für reichhaltige Karten
        self.transcript_scroll = ctk.CTkScrollableFrame(
            self.transcript_container,
            fg_color="transparent",
            scrollbar_button_color=t["border"],
            scrollbar_button_hover_color=t["accent"]
        )
        self.transcript_scroll.pack(fill="both", expand=True, padx=20, pady=(8, 20))

        self._show_empty_state()

    def _draw_indicator(self):
        """Zeichnet einen pulsierenden Aurora-Glow-Punkt."""
        c = self.indicator_canvas
        c.delete("all")
        t = self.theme
        c.configure(bg=t["bg"])

        if self.is_recording:
            color = t["red"]
            pulse = (time.time() * 2) % 2
            radius = 6 + (1 if pulse < 1 else 2 - pulse) * 3
            # Glow-Layers
            for r in range(int(radius + 6), int(radius), -1):
                alpha_color = self._with_alpha(color, 0.05 + (radius / r) * 0.1)
                c.create_oval(12-r, 12-r, 12+r, 12+r, fill=alpha_color, outline="")
            c.create_oval(12-radius, 12-radius, 12+radius, 12+radius,
                          fill=color, outline="")
        else:
            color = t["accent"]
            c.create_oval(12-7, 12-7, 12+7, 12+7,
                          fill=color, outline=t["accent2"], width=1)

        if self.is_recording:
            self.after(50, self._draw_indicator)

    def _with_alpha(self, hex_color, alpha):
        """Mischt Farbe Richtung Background um Alpha zu simulieren."""
        bg = self.theme["bg"]
        r1, g1, b1 = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        r2, g2, b2 = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
        r = int(r2 + (r1 - r2) * alpha)
        g = int(g2 + (g1 - g2) * alpha)
        b = int(b2 + (b1 - b2) * alpha)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _show_empty_state(self):
        # Clear children
        for w in self.transcript_scroll.winfo_children():
            w.destroy()

        empty = ctk.CTkFrame(self.transcript_scroll, fg_color="transparent")
        empty.pack(pady=80)

        ctk.CTkLabel(
            empty, text="◌", text_color=self.theme["text_mute"],
            font=ctk.CTkFont(size=48)
        ).pack()

        ctk.CTkLabel(
            empty, text="Bereit, deine Worte einzufangen.",
            text_color=self.theme["text_dim"],
            font=ctk.CTkFont(family="Georgia", size=15, slant="italic")
        ).pack(pady=(12, 4))

        ctk.CTkLabel(
            empty, text="Quelle wählen → Aufnahme starten",
            text_color=self.theme["text_mute"],
            font=ctk.CTkFont(family="Consolas", size=11)
        ).pack()

    def _add_entry_card(self, entry):
        """Fügt eine Transkript-Karte hinzu, mit Quellen-Tag und optionaler Pausen-Trennung."""
        t = self.theme

        # Pausen-Trennung wenn lange Lücke
        if (entry["elapsed"] - self.last_entry_elapsed) > PAUSE_THRESHOLD_SEC and len(self.entries) > 1:
            divider = ctk.CTkFrame(self.transcript_scroll, fg_color="transparent", height=24)
            divider.pack(fill="x", pady=4)
            line = ctk.CTkFrame(divider, fg_color=t["border"], height=1)
            line.pack(fill="x", padx=200, pady=11)
            pause_sec = int(entry["elapsed"] - self.last_entry_elapsed)
            ctk.CTkLabel(
                divider, text=f"  ··· {pause_sec}s Pause ···  ",
                text_color=t["text_mute"], fg_color=t["bg_elev"],
                font=ctk.CTkFont(family="Consolas", size=9)
            ).place(relx=0.5, rely=0.5, anchor="center")

        self.last_entry_elapsed = entry["elapsed"]

        # Card
        card = ctk.CTkFrame(self.transcript_scroll, fg_color="transparent")
        card.pack(fill="x", pady=4)

        # Linke Meta-Spalte
        meta = ctk.CTkFrame(card, fg_color="transparent", width=100)
        meta.pack(side="left", padx=(0, 12), anchor="n")
        meta.pack_propagate(False)

        # Bookmark-Marker
        if entry.get("bookmark"):
            ctk.CTkLabel(
                meta, text="✦", text_color=t["accent3"],
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w")

        # Timestamp
        ctk.CTkLabel(
            meta, text=entry["ts"], text_color=t["text_dim"],
            font=ctk.CTkFont(family="Consolas", size=11)
        ).pack(anchor="w")

        # Source-Badge
        src_color = t["accent"] if entry["source"] == "system" else t["accent2"]
        src_icon = "🔊" if entry["source"] == "system" else "🎙"
        src_label = "SYS" if entry["source"] == "system" else "MIC"
        if entry["source"] == "mix":
            src_color = t["accent3"]
            src_icon = "◐"
            src_label = "MIX"

        badge = ctk.CTkLabel(
            meta, text=f"{src_icon} {src_label}",
            text_color=src_color,
            font=ctk.CTkFont(family="Consolas", size=9, weight="bold")
        )
        badge.pack(anchor="w", pady=(2, 0))

        # Text
        text_frame = ctk.CTkFrame(card, fg_color="transparent")
        text_frame.pack(side="left", fill="both", expand=True)

        # Bookmark-Hintergrund-Akzent
        if entry.get("bookmark"):
            text_frame.configure(fg_color=t["bg_card"])
            inner = ctk.CTkFrame(text_frame, fg_color="transparent")
            inner.pack(fill="both", expand=True, padx=12, pady=8)
            text_label = ctk.CTkLabel(
                inner, text=entry["text"], text_color=t["text"],
                font=ctk.CTkFont(family="Georgia", size=15),
                wraplength=700, justify="left", anchor="w"
            )
            text_label.pack(anchor="w", fill="x")
        else:
            text_label = ctk.CTkLabel(
                text_frame, text=entry["text"], text_color=t["text"],
                font=ctk.CTkFont(family="Georgia", size=15),
                wraplength=700, justify="left", anchor="w"
            )
            text_label.pack(anchor="w", fill="x")

        # Auto-scroll nach unten
        self.after(50, lambda: self.transcript_scroll._parent_canvas.yview_moveto(1.0))

    # ============================================================
    # THEME TOGGLE
    # ============================================================
    def _toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.theme = THEMES[self.theme_name]
        ctk.set_appearance_mode(self.theme_name)
        self.configure(fg_color=self.theme["bg"])
        self.theme_btn.configure(text="☀" if self.theme_name == "light" else "☾")
        self._refresh_theme()

    def _refresh_theme(self):
        """Baut das Transkript neu mit aktuellem Theme."""
        t = self.theme
        self.visualizer.update_theme(t)
        self._draw_indicator()

        # Texte / Borders
        self.title_label.configure(text_color=t["text"])
        self.subtitle_label.configure(text_color=t["text_dim"])
        self.timer_label.configure(text_color=t["text"])
        self.status_label.configure(text_color=t["text_dim"])
        self.transcript_title.configure(text_color=t["text_dim"])
        self.word_count_label.configure(text_color=t["text_dim"])
        self.hotkey_hint.configure(text_color=t["text_mute"])

        self.controls.configure(fg_color=t["bg_elev"])
        self.transcript_container.configure(fg_color=t["bg_elev"])

        # Buttons-Farben grob auffrischen
        self.start_btn.configure(fg_color=t["accent"], text_color=t["bg"], hover_color=t["accent2"])
        self.stop_btn.configure(fg_color=t["red"])
        self.bookmark_btn.configure(text_color=t["accent3"], border_color=t["accent3"], hover_color=t["bg_hover"])

        for btn in (self.theme_btn, self.mini_btn, self.clear_btn,
                    self.export_txt_btn, self.export_md_btn):
            btn.configure(text_color=t["text_dim"], border_color=t["border"], hover_color=t["bg_hover"])

        self.lang_frame.configure(fg_color=t["bg_elev"])
        self._set_language(self.language)
        self._update_source_buttons()

        # Transkript neu rendern
        for w in self.transcript_scroll.winfo_children():
            w.destroy()

        if not self.entries:
            self._show_empty_state()
        else:
            self.last_entry_elapsed = 0
            for entry in self.entries:
                self._add_entry_card(entry)

    # ============================================================
    # MINI-MODUS
    # ============================================================
    def _toggle_mini(self):
        if not self.is_mini:
            # In Mini-Modus
            self.is_mini = True
            self._saved_geometry = self.geometry()
            self.geometry("420x180")
            self.attributes("-topmost", True)
            self.minsize(380, 140)
            # Verstecke alles außer Header + Visualizer
            self.transcript_container.pack_forget()
            for child in self.controls.winfo_children():
                pass  # behalte controls
            self.mini_btn.configure(text="◰  Voll")
        else:
            # Zurück
            self.is_mini = False
            self.geometry(self._saved_geometry if hasattr(self, "_saved_geometry") else "1100x780")
            self.attributes("-topmost", False)
            self.minsize(900, 600)
            self.transcript_container.pack(fill="both", expand=True, padx=28, pady=(12, 24))
            self.mini_btn.configure(text="◱  Mini")

    # ============================================================
    # MODELL LADEN
    # ============================================================
    def _load_model_async(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
            try:
                self.model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
                device_info = "GPU"
            except Exception as e:
                print(f"GPU nicht verfügbar: {e}")
                self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
                device_info = "CPU"
            self.after(0, lambda: self.status_label.configure(
                text=f"Bereit · {MODEL_SIZE} · {device_info}"
            ))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: self.status_label.configure(text=f"Fehler: {err[:50]}"))
            self.after(0, lambda: messagebox.showerror(
                "Modell konnte nicht geladen werden", err
            ))

    # ============================================================
    # SPRACHE & QUELLE
    # ============================================================
    def _set_language(self, lang):
        if self.is_recording:
            return
        self.language = lang
        t = self.theme
        if lang == "de":
            self.lang_de_btn.configure(fg_color=t["accent"], text_color=t["bg"])
            self.lang_en_btn.configure(fg_color="transparent", text_color=t["text_dim"])
        else:
            self.lang_en_btn.configure(fg_color=t["accent"], text_color=t["bg"])
            self.lang_de_btn.configure(fg_color="transparent", text_color=t["text_dim"])

    def _set_source(self, value):
        if self.is_recording:
            return
        self.source_mode = value
        self.source_var.set(value)
        self._update_source_buttons()

    def _update_source_buttons(self):
        t = self.theme
        for value, btn in self.source_buttons.items():
            if value == self.source_mode:
                btn.configure(fg_color=t["accent2"], text_color=t["bg"], hover_color=t["accent"])
            else:
                btn.configure(fg_color=t["bg_card"], text_color=t["text_dim"], hover_color=t["bg_hover"])

    # ============================================================
    # AUFNAHME
    # ============================================================
    def start_recording(self):
        if self.is_recording or self.model is None:
            if self.model is None:
                messagebox.showwarning("Bitte warten", "Whisper-Modell lädt noch.")
            return

        self.is_recording = True
        self.recording_start = time.time()
        self.last_entry_elapsed = 0
        self.stop_event.clear()
        self.audio_queue = queue.Queue()

        if not self.entries:
            for w in self.transcript_scroll.winfo_children():
                w.destroy()

        self.capture_thread = threading.Thread(target=self._capture_audio, daemon=True)
        self.transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.capture_thread.start()
        self.transcribe_thread.start()

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.bookmark_btn.configure(state="normal")
        src = {"system": "System", "mic": "Mic", "both": "System+Mic"}[self.source_mode]
        self.status_label.configure(text=f"● Aufnahme · {src}")
        self._draw_indicator()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.stop_event.set()

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.bookmark_btn.configure(state="disabled")
        self.status_label.configure(text="Verarbeite letzten Chunk…")
        self.after(1500, lambda: self.status_label.configure(text="Gestoppt"))
        self._draw_indicator()
        self._save_autosave()

    def add_bookmark(self):
        """Markiert den NÄCHSTEN Eintrag als Bookmark."""
        self._pending_bookmark = True
        # Visuelles Feedback
        original_text = self.bookmark_btn.cget("text")
        self.bookmark_btn.configure(text="✦  Markiert!")
        self.after(1200, lambda: self.bookmark_btn.configure(text=original_text))

    # ============================================================
    # AUDIO CAPTURE
    # ============================================================
    def _capture_audio(self):
        try:
            chunk_frames = SAMPLE_RATE * CHUNK_SECONDS
            sources = []

            if self.source_mode in ("system", "both"):
                default_speaker = sc.default_speaker()
                loopback_mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
                sources.append(("system", loopback_mic))

            if self.source_mode in ("mic", "both"):
                default_mic = sc.default_microphone()
                sources.append(("mic", default_mic))

            recorders = []
            for name, src in sources:
                rec = src.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=1024)
                rec.__enter__()
                recorders.append((name, rec))

            try:
                # Sub-chunking für Visualizer (alle ~100ms ein Pegel)
                viz_frames = SAMPLE_RATE // 10  # 100ms
                accumulated = {name: [] for name, _ in recorders}
                accumulated_frames = 0

                while not self.stop_event.is_set():
                    sub_chunks = {}
                    for name, rec in recorders:
                        data = rec.record(numframes=viz_frames)
                        if data.ndim > 1:
                            data = data.mean(axis=1)
                        sub_chunks[name] = data.astype(np.float32)
                        accumulated[name].append(data.astype(np.float32))

                    # Visualizer füttern
                    if sub_chunks:
                        if len(sub_chunks) == 1:
                            viz_audio = list(sub_chunks.values())[0]
                        else:
                            viz_audio = sum(sub_chunks.values()) / len(sub_chunks)
                        rms = float(np.sqrt(np.mean(viz_audio ** 2)))
                        # Auf 0..1 mappen mit etwas Boost
                        level = min(1.0, rms * 8)
                        self.visualizer.push_level(level)

                    accumulated_frames += viz_frames

                    # Wenn 5 Sekunden voll → an Whisper schicken
                    if accumulated_frames >= chunk_frames:
                        chunks_per_source = {}
                        for name in accumulated:
                            chunks_per_source[name] = np.concatenate(accumulated[name])
                            accumulated[name] = []
                        accumulated_frames = 0

                        # Mix für Transkription
                        if len(chunks_per_source) == 1:
                            mixed = list(chunks_per_source.values())[0]
                            source_tag = list(chunks_per_source.keys())[0]
                        else:
                            min_len = min(len(c) for c in chunks_per_source.values())
                            mixed = sum(c[:min_len] for c in chunks_per_source.values())
                            mixed = np.tanh(mixed)
                            source_tag = "mix"

                        rms_full = np.sqrt(np.mean(mixed ** 2))
                        if rms_full > MIN_AUDIO_LEVEL:
                            elapsed = time.time() - self.recording_start
                            self.audio_queue.put((elapsed, mixed, source_tag))

            finally:
                for _, rec in recorders:
                    try: rec.__exit__(None, None, None)
                    except Exception: pass

        except Exception as e:
            err = str(e)
            print(f"Capture error: {err}")
            self.after(0, lambda: self.status_label.configure(text=f"Audio-Fehler: {err[:50]}"))
            self.is_recording = False

    # ============================================================
    # TRANSKRIPTION
    # ============================================================
    def _transcribe_loop(self):
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                elapsed, audio, source = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                segments, _ = self.model.transcribe(
                    audio, language=self.language, beam_size=1,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                full_text = " ".join(seg.text.strip() for seg in segments).strip()

                if full_text:
                    ts = self._format_timestamp(elapsed)
                    self.after(0, lambda t=ts, txt=full_text, e=elapsed, s=source:
                               self._add_entry(t, txt, e, s))
            except Exception as e:
                print(f"Transcribe error: {e}")

    def _format_timestamp(self, sec):
        m = int(sec // 60); s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

    def _add_entry(self, ts, text, elapsed, source):
        if text:
            text = text[0].upper() + text[1:]

        is_bookmark = getattr(self, "_pending_bookmark", False)
        self._pending_bookmark = False

        entry = {
            "ts": ts, "text": text, "elapsed": elapsed,
            "source": source, "bookmark": is_bookmark
        }
        self.entries.append(entry)
        self._add_entry_card(entry)

        words = sum(len(e["text"].split()) for e in self.entries)
        bookmarks = sum(1 for e in self.entries if e.get("bookmark"))
        self.word_count_label.configure(text=f"{words} Wörter · {bookmarks} Bookmarks")

        self.export_txt_btn.configure(state="normal")
        self.export_md_btn.configure(state="normal")

    # ============================================================
    # TIMER & ANIMATION
    # ============================================================
    def _update_timer(self):
        if self.is_recording and self.recording_start:
            el = time.time() - self.recording_start
            h = int(el // 3600); m = int((el % 3600) // 60); s = int(el % 60)
            self.timer_label.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.after(250, self._update_timer)

    def _animate_visualizer(self):
        self.visualizer.animate()
        self.after(40, self._animate_visualizer)

    # ============================================================
    # AUTOSAVE
    # ============================================================
    def _autosave_loop(self):
        if self.is_recording and time.time() - self.last_autosave > AUTOSAVE_INTERVAL_SEC:
            self._save_autosave()
        self.after(5000, self._autosave_loop)

    def _save_autosave(self):
        try:
            data = {
                "saved_at": datetime.now().isoformat(),
                "language": self.language,
                "entries": self.entries,
            }
            AUTOSAVE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.last_autosave = time.time()
        except Exception as e:
            print(f"Autosave failed: {e}")

    def _check_autosave_recovery(self):
        if not AUTOSAVE_PATH.exists():
            return
        try:
            data = json.loads(AUTOSAVE_PATH.read_text(encoding="utf-8"))
            if data.get("entries"):
                ans = messagebox.askyesno(
                    "Wiederherstellung",
                    f"Es wurde ein nicht-gespeichertes Transkript gefunden vom\n"
                    f"{data.get('saved_at', '?')[:19]}\n\n"
                    f"Wiederherstellen? ({len(data['entries'])} Einträge)"
                )
                if ans:
                    self.entries = data["entries"]
                    self.language = data.get("language", "de")
                    self._set_language(self.language)
                    for w in self.transcript_scroll.winfo_children():
                        w.destroy()
                    self.last_entry_elapsed = 0
                    for entry in self.entries:
                        self._add_entry_card(entry)
                    words = sum(len(e["text"].split()) for e in self.entries)
                    bookmarks = sum(1 for e in self.entries if e.get("bookmark"))
                    self.word_count_label.configure(text=f"{words} Wörter · {bookmarks} Bookmarks")
                    self.export_txt_btn.configure(state="normal")
                    self.export_md_btn.configure(state="normal")
                else:
                    AUTOSAVE_PATH.unlink(missing_ok=True)
        except Exception as e:
            print(f"Recovery failed: {e}")

    # ============================================================
    # HOTKEYS
    # ============================================================
    def _bind_hotkeys(self):
        self.bind("<Control-r>", lambda e: self._toggle_recording())
        self.bind("<Control-R>", lambda e: self._toggle_recording())
        self.bind("<F2>", lambda e: self.add_bookmark() if self.is_recording else None)
        self.bind("<Escape>", lambda e: self.stop_recording() if self.is_recording else None)

    def _toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    # ============================================================
    # CLEAR & EXPORT
    # ============================================================
    def clear_transcript(self):
        if self.is_recording:
            if not messagebox.askyesno("Aufnahme läuft", "Wirklich alles löschen?"):
                return
            self.stop_recording()

        self.entries = []
        self.recording_start = None
        self.last_entry_elapsed = 0
        self.timer_label.configure(text="00:00:00")
        self.word_count_label.configure(text="0 Wörter · 0 Bookmarks")
        self.export_txt_btn.configure(state="disabled")
        self.export_md_btn.configure(state="disabled")
        self._show_empty_state()
        self.status_label.configure(text="Bereit")
        AUTOSAVE_PATH.unlink(missing_ok=True)

    def export(self, fmt):
        if not self.entries:
            return
        now = datetime.now()
        default = f"meeting_{now.strftime('%Y-%m-%d_%H%M')}.{fmt}"
        path = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}", initialfile=default,
            filetypes=[(f"{fmt.upper()}", f"*.{fmt}")]
        )
        if not path:
            return

        if fmt == "txt":
            content = self._build_txt(now)
        else:
            content = self._build_md(now)

        try:
            Path(path).write_text(content, encoding="utf-8")
            self.status_label.configure(text=f"Exportiert: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Export fehlgeschlagen", str(e))

    def _build_txt(self, now):
        lang = "Deutsch" if self.language == "de" else "Englisch"
        out = f"Meeting Transkription\n{now.strftime('%d.%m.%Y %H:%M')}\nSprache: {lang}\n"
        out += "=" * 50 + "\n\n"
        for e in self.entries:
            mark = "✦ " if e.get("bookmark") else ""
            src = {"system": "[SYS]", "mic": "[MIC]", "mix": "[MIX]"}.get(e["source"], "")
            out += f"{mark}[{e['ts']}] {src} {e['text']}\n\n"
        return out

    def _build_md(self, now):
        lang = "Deutsch" if self.language == "de" else "Englisch"
        last = self.entries[-1]["ts"] if self.entries else "–"
        out = f"# Meeting Transkription\n\n"
        out += f"**Datum:** {now.strftime('%d.%m.%Y %H:%M')}  \n"
        out += f"**Sprache:** {lang}  \n**Dauer:** {last}  \n"
        out += f"**Quelle:** {self.source_mode}\n\n---\n\n"

        bookmarks = [e for e in self.entries if e.get("bookmark")]
        if bookmarks:
            out += "## ✦ Bookmarks\n\n"
            for b in bookmarks:
                out += f"- **[{b['ts']}]** {b['text'][:80]}…\n"
            out += "\n---\n\n## Transkript\n\n"

        for e in self.entries:
            mark = "✦ " if e.get("bookmark") else ""
            src_tag = {"system": "🔊", "mic": "🎙", "mix": "◐"}.get(e["source"], "")
            out += f"{mark}**[{e['ts']}]** {src_tag} {e['text']}\n\n"
        return out


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    app = TranscriptionApp()

    def on_close():
        if app.is_recording:
            if not messagebox.askyesno("Aufnahme läuft", "Wirklich beenden?"):
                return
            app.stop_recording()
            time.sleep(0.5)
        app._save_autosave()
        app.destroy()
        sys.exit(0)

    app.protocol("WM_DELETE_WINDOW", on_close)
    app.mainloop()
