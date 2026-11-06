"""
Spacoust Desktop — NASA Earth image → AI analysis → music.

التشغيل:
    python -m pip install -r requirements.txt
    python desktop_app.py

مفتاح OpenAI: الصقه داخل ملف api_key.txt بجانب هذا الملف (سطر واحد يبدأ بـ sk-).
كل صورة تُحلَّل بالذكاء الاصطناعي قبل توليد الصوت.
"""

import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image

import spacoust as sc

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
ENV_FILE = APP_DIR / ".env"
KEY_FILE = APP_DIR / "api_key.txt"   # الصق مفتاحك داخل هذا الملف
LOGO_FILE = Path(getattr(sys, "_MEIPASS", APP_DIR)) / "logo.png"   # الشعار بجانب هذا الملف

ACCENT = "#3B82F6"
SCALE_LABELS = {
    "major_pentatonic": "Major pentatonic 🌿",
    "minor_pentatonic": "Minor pentatonic 🌊",
    "dorian": "Dorian ❄️",
    "phrygian": "Phrygian 🔥",
}


# ──────────────────────────── API key storage ────────────────────────────

def _key_from_file(path):
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            value = line.strip()
            if value.startswith("OPENAI_API_KEY="):
                value = value.split("=", 1)[1]
            value = value.strip().strip("\"'")
            if value.startswith("sk-") and value.isascii() and len(value) > 20 and "PASTE" not in value:
                return value
    except OSError:
        pass
    return None


def load_key():
    return (_key_from_file(KEY_FILE)
            or os.getenv("OPENAI_API_KEY")
            or _key_from_file(ENV_FILE)
            or _key_from_file(APP_DIR / ".env.example")
            # key bundled inside Spacoust.exe (so the exe works alone when shared)
            or _key_from_file(Path(getattr(sys, "_MEIPASS", APP_DIR)) / "api_key.txt"))



# ──────────────────────────── audio playback ────────────────────────────

class Player:
    """Plays WAV files with the OS's built-in tools (no extra libraries)."""

    def __init__(self):
        self.proc = None

    def play(self, path):
        self.stop()
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif sys.platform == "darwin":
            self.proc = subprocess.Popen(["afplay", str(path)])
        else:
            self.proc = subprocess.Popen(["aplay", "-q", str(path)])

    def stop(self):
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        elif self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None


# ──────────────────────────── main window ────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("Spacoust — Earth Information Jukebox")
        self.geometry("1040x700")
        self._load_logo()
        self._set_window_icon()
        self.minsize(920, 640)

        self.api_key = load_key()
        self.image_path = None
        self.wav_path = None
        self.melody = []
        self.run_id = 0
        self.player = Player()

        self.grid_columnconfigure((0, 1), weight=1, uniform="col")
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_left()
        self._build_right()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── layout ──
    def _build_header(self):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(16, 10))
        if self.icon_img:
            icon = self.icon_img
            self._header_icon = ctk.CTkImage(light_image=icon, dark_image=icon, size=(40, 40))
            ctk.CTkLabel(bar, image=self._header_icon, text="").pack(side="left", padx=(0, 10))
        ctk.CTkLabel(bar, text="Spacoust", font=ctk.CTkFont(size=26, weight="bold")).pack(side="left")
        ctk.CTkLabel(bar, text="   Listen to our planet — NASA Earth data sonification",
                     text_color="gray60").pack(side="left", pady=(6, 0))

    def _build_left(self):
        left = ctk.CTkFrame(self)
        left.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(0, 20))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)

        self.preview = ctk.CTkLabel(left, text="No image yet\n\nOpen a NASA Earth image to begin",
                                    fg_color=("gray85", "gray14"), corner_radius=10, text_color="gray55")
        self.preview.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        self.preview.bind("<Configure>", lambda _e: self._show_preview())

        ctk.CTkButton(left, text="📂  Open image", height=38, fg_color="gray25",
                      command=self.open_image).grid(row=1, column=0, sticky="ew", padx=16)

        row = ctk.CTkFrame(left, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew", padx=16, pady=(14, 0))
        ctk.CTkLabel(row, text="Notes").pack(side="left")
        self.steps_label = ctk.CTkLabel(row, text="64", width=36)
        self.steps_label.pack(side="right")
        self.steps = ctk.CTkSlider(row, from_=16, to=160, number_of_steps=18,
                                   command=lambda v: self.steps_label.configure(text=str(int(v))))
        self.steps.set(64)
        self.steps.pack(side="left", fill="x", expand=True, padx=10)

        self.sonify_btn = ctk.CTkButton(left, text="🎶  Analyze & Sonify", height=44,
                                        font=ctk.CTkFont(size=15, weight="bold"),
                                        fg_color=ACCENT, state="disabled", command=self.sonify)
        self.sonify_btn.grid(row=3, column=0, sticky="ew", padx=16, pady=(14, 8))

        self.progress = ctk.CTkProgressBar(left, mode="determinate", progress_color=ACCENT)
        self.progress.grid(row=4, column=0, sticky="ew", padx=16)
        self.progress.set(0)
        self.status = ctk.CTkLabel(left, text="", text_color="gray60")
        self.status.grid(row=5, column=0, sticky="w", padx=16, pady=(4, 12))

    def _build_right(self):
        right = ctk.CTkFrame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(0, 20))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(right, text="🤖  What the AI sees", font=ctk.CTkFont(size=15, weight="bold"),
                     anchor="w").grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 6))
        self.desc = ctk.CTkTextbox(right, height=110, wrap="word")
        self.desc.grid(row=1, column=0, sticky="ew", padx=16)
        self._set_desc("The AI description of your image will appear here.")

        cards = ctk.CTkFrame(right, fg_color="transparent")
        cards.grid(row=2, column=0, sticky="ew", padx=16, pady=12)
        cards.grid_columnconfigure((0, 1, 2), weight=1, uniform="card")
        self.metrics = {}
        for i, (key, title) in enumerate([("phenomenon", "Phenomenon"), ("mood", "Mood"),
                                          ("tempo", "Tempo"), ("scale", "Scale"),
                                          ("duration", "Duration"), ("key", "Key")]):
            card = ctk.CTkFrame(cards, fg_color=("gray85", "gray17"), corner_radius=8)
            card.grid(row=i // 3, column=i % 3, sticky="ew", padx=4, pady=4)
            ctk.CTkLabel(card, text=title, text_color="gray55", font=ctk.CTkFont(size=11)).pack(pady=(6, 0))
            lbl = ctk.CTkLabel(card, text="—", font=ctk.CTkFont(size=14, weight="bold"), wraplength=140)
            lbl.pack(pady=(0, 6))
            self.metrics[key] = lbl

        mel = ctk.CTkFrame(right, fg_color=("gray85", "gray14"), corner_radius=10)
        mel.grid(row=3, column=0, sticky="nsew", padx=16)
        ctk.CTkLabel(mel, text="Melody line  (gaps = darkness = silence)", text_color="gray55",
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10, pady=(6, 0))
        self.canvas = ctk.CTkCanvas(mel, bg="#1d1d1d", highlightthickness=0, height=120)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        self.canvas.bind("<Configure>", lambda _e: self._draw_melody())

        btns = ctk.CTkFrame(right, fg_color="transparent")
        btns.grid(row=4, column=0, sticky="ew", padx=16, pady=16)
        btns.grid_columnconfigure((0, 1, 2), weight=1)
        self.play_btn = ctk.CTkButton(btns, text="▶  Play", fg_color=ACCENT, state="disabled",
                                      command=lambda: self.player.play(self.wav_path))
        self.play_btn.grid(row=0, column=0, sticky="ew", padx=4)
        self.stop_btn = ctk.CTkButton(btns, text="■  Stop", fg_color="gray25", state="disabled",
                                      command=self.player.stop)
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=4)
        self.save_btn = ctk.CTkButton(btns, text="💾  Save WAV", fg_color="gray25", state="disabled",
                                      command=self.save_wav)
        self.save_btn.grid(row=0, column=2, sticky="ew", padx=4)

    # ── helpers ──
    def _set_desc(self, text):
        self.desc.configure(state="normal")
        self.desc.delete("1.0", "end")
        self.desc.insert("1.0", text)
        self.desc.configure(state="disabled")

    def _set_status(self, text):
        self.after(0, lambda: self.status.configure(text=text))

    def _load_logo(self):
        """Builds the round globe icon and the white logo card from logo.png."""
        self.logo_img = self.icon_img = self.card_img = None
        if not LOGO_FILE.exists():
            return
        from PIL import ImageDraw
        logo = Image.open(LOGO_FILE).convert("RGBA")
        self.logo_img = logo
        w, h = logo.size
        cx, cy, r = int(w * 0.502), int(h * 0.472), int(w * 0.206)   # the globe
        icon = logo.crop((cx - r, cy - r, cx + r, cy + r))
        mask = Image.new("L", icon.size, 0)
        ImageDraw.Draw(mask).ellipse((1, 1, 2 * r - 1, 2 * r - 1), fill=255)
        alpha = Image.fromarray(np.minimum(np.asarray(icon)[..., 3], np.asarray(mask)))
        icon.putalpha(alpha)
        self.icon_img = icon.resize((256, 256), Image.LANCZOS)

        pad = 28
        card = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        m = Image.new("L", card.size, 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, card.width - 1, card.height - 1), 48, fill=255)
        card.paste(Image.new("RGBA", card.size, (255, 255, 255, 255)), (0, 0), m)
        card.alpha_composite(logo, (pad, pad))
        self.card_img = card

    def _set_window_icon(self):
        """Taskbar / title-bar icon (CustomTkinter sets its own first, so set ours after it)."""
        if not self.icon_img:
            return
        from PIL import ImageTk

        def apply():
            try:
                self._icon_photo = ImageTk.PhotoImage(self.icon_img.resize((64, 64), Image.LANCZOS))
                self.iconphoto(True, self._icon_photo)
            except Exception:
                pass

        apply()
        self.after(300, apply)

    def _show_logo_placeholder(self):
        if not self.card_img:
            return
        side = min(self.preview.winfo_width(), self.preview.winfo_height()) - 60
        side = max(min(side, 300), 80)
        img = self.card_img.copy()
        img.thumbnail((side, side))
        self._ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.preview.configure(image=self._ctk_img, text="Open a NASA Earth image to begin",
                               compound="top")

    def _show_preview(self):
        if not self.image_path:
            self._show_logo_placeholder()
            return
        w, h = max(self.preview.winfo_width() - 8, 50), max(self.preview.winfo_height() - 8, 50)
        img = Image.open(self.image_path).convert("RGB")
        img.thumbnail((w, h))
        self._ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.preview.configure(image=self._ctk_img, text="", compound="center")

    def _draw_melody(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if not self.melody or w < 10:
            return
        values = [m for m in self.melody if m is not None]
        if not values:
            return
        lo, hi = min(values) - 1, max(values) + 1
        n = len(self.melody)
        pad = 8
        prev = None
        for i, m in enumerate(self.melody):
            if m is None:
                prev = None
                continue
            x = pad + i * (w - 2 * pad) / max(n - 1, 1)
            y = h - pad - (m - lo) / (hi - lo) * (h - 2 * pad)
            if prev:
                c.create_line(*prev, x, y, fill=ACCENT, width=2)
            c.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, fill="#93C5FD", outline="")
            prev = (x, y)

    # ── actions ──
    def reload_key(self):
        """Reads the key again from api_key.txt (no key UI inside the app)."""
        self.api_key = load_key()
        if not self.api_key:
            messagebox.showwarning(
                "OpenAI key missing",
                f"Paste your OpenAI key into this file, save it, then try again:\n\n{KEY_FILE}")
        return bool(self.api_key)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose a NASA Earth image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.tif *.tiff *.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            Image.open(path).verify()
        except Exception:
            messagebox.showerror("Not an image", "This file could not be opened as an image.")
            return
        self.image_path = path
        self._show_preview()
        self.sonify_btn.configure(state="normal")
        self.status.configure(text=Path(path).name)

    def sonify(self):
        if not self.image_path:
            return
        if not self.reload_key():
            return
        self.player.stop()
        self._busy(True)
        self.run_id += 1
        args = (self.image_path, int(self.steps.get()), self.api_key, self.run_id)
        threading.Thread(target=self._work, args=args, daemon=True).start()

    def _work(self, path, steps, key, run_id):
        try:
            self._set_status("🤖 AI is analyzing the image…")
            ai = sc.ai_analyze(path, api_key=key)

            self._set_status("🎼 Composing the music…")
            img = Image.open(path).convert("RGB")
            img.thumbnail((1024, 1024))
            features, columns = sc.extract_features(np.asarray(img), steps)
            music = sc.map_to_music(features, columns, ai)
            audio = sc.render(music)

            wav = Path(tempfile.gettempdir()) / f"spacoust_{os.getpid()}_{run_id}.wav"
            sc.save_wav(str(wav), audio)
            self.after(0, self._done, ai, music, len(audio) / sc.SR, wav)
        except Exception as e:  # noqa: BLE001
            self.after(0, self._failed, e)

    def _done(self, ai, music, duration, wav):
        self._busy(False)
        self.wav_path = wav
        self._set_desc(ai.get("description", "—"))
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        values = {
            "phenomenon": str(ai.get("phenomenon", "—")).title(),
            "mood": str(ai.get("mood", "—")).title(),
            "tempo": f"{music['tempo_bpm']:.0f} BPM",
            "scale": SCALE_LABELS.get(music["scale"], music["scale"]),
            "duration": f"{duration:.0f} s",
            "key": names[music["root_midi"] % 12],
        }
        for k, v in values.items():
            self.metrics[k].configure(text=v)
        self.melody = [n["midi"] if n else None for n in music["notes"]]
        self._draw_melody()
        for b in (self.play_btn, self.stop_btn, self.save_btn):
            b.configure(state="normal")
        self.status.configure(text="✅ Done — playing")
        self.player.play(wav)

    def _failed(self, err):
        self._busy(False)
        name = type(err).__name__
        if name == "AuthenticationError":
            self.status.configure(text="❌ Invalid API key")
            messagebox.showerror("Invalid API key",
                                 f"OpenAI rejected the key. Check the key in:\n\n{KEY_FILE}")
        elif name in ("APIConnectionError", "APITimeoutError"):
            self.status.configure(text="❌ No connection to OpenAI")
            messagebox.showerror("Connection problem", "Could not reach OpenAI. Check your internet and try again.")
        elif name == "RateLimitError":
            self.status.configure(text="❌ OpenAI limit / no credit")
            messagebox.showerror("OpenAI limit", "Rate limit or no credit on this key.\n\n" + str(err)[:300])
        else:
            self.status.configure(text=f"❌ {name}")
            messagebox.showerror("Error", f"{name}: {str(err)[:400]}")

    def _busy(self, busy):
        self.sonify_btn.configure(state="disabled" if busy else "normal")
        if busy:
            self.progress.configure(mode="indeterminate")
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress.set(1)

    def save_wav(self):
        if not self.wav_path:
            return
        default = f"{Path(self.image_path).stem}_spacoust.wav"
        dest = filedialog.asksaveasfilename(defaultextension=".wav", initialfile=default,
                                            filetypes=[("WAV audio", "*.wav")])
        if dest:
            Path(dest).write_bytes(Path(self.wav_path).read_bytes())
            self.status.configure(text=f"💾 Saved: {Path(dest).name}")

    def _on_close(self):
        self.player.stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
