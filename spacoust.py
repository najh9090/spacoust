#!/usr/bin/env python3
"""
Spacoust — Earth Data Sonification
NASA Space Apps Challenge · "The Earth Information Jukebox"

يحوّل صورة أو إطاراً مرئياً لبيانات الأرض إلى مقطوعة صوتية (.wav).

الاستخدام:
    python spacoust.py                      # يولّد صورة تجريبية ويحوّلها
    python spacoust.py earth.jpg            # يحوّل صورتك
    python spacoust.py earth.jpg -o out.wav --steps 96
    python spacoust.py earth.jpg --ai       # تحليل الصورة بـ OpenAI أولاً

المكتبات:
    pip install numpy scipy pillow opencv-python
    pip install openai        # اختياري، فقط مع --ai
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100  # معدل العينات (Hz)

# السلالم الموسيقية: كل رقم = عدد أنصاف النغمات من الجذر
SCALES = {
    "major_pentatonic": [0, 2, 4, 7, 9],      # الغطاء النباتي / الحياة
    "minor_pentatonic": [0, 3, 5, 7, 10],     # المحيطات / المياه
    "dorian": [0, 2, 3, 5, 7, 9, 10],         # الجليد / الغيوم / الرمادي
    "phrygian": [0, 1, 3, 5, 7, 8, 10],       # الحرارة / الحرائق / الصحاري
}


# ═══════════════════════════ 1) قراءة الصورة ═══════════════════════════

def load_image(path, max_side=1024):
    """يقرأ الصورة كمصفوفة RGB ويصغّرها لتسريع المعالجة."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return np.asarray(img)


def make_demo_image(path, w=800, h=400, seed=7):
    """يولّد صورة تجريبية تشبه الأرض: محيط، يابسة، صحراء، غيوم."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 25)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    land = noise + 0.35 * np.sin(x * 6.0) > 0.85
    desert = land & (np.abs(y - 0.5) < 0.18)
    clouds = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 12)
    clouds = np.clip((clouds - clouds.mean()) * 18, 0, 1)

    img = np.zeros((h, w, 3), np.float32)
    img[:] = [0.05, 0.18, 0.45]                       # محيط
    img[land] = [0.15, 0.45, 0.15]                    # نبات
    img[desert] = [0.80, 0.55, 0.25]                  # صحراء
    img[y < 0.08] = img[y > 0.92] = [0.92, 0.95, 1.0]  # جليد قطبي
    img = img * (1 - clouds[..., None]) + clouds[..., None]
    img *= 0.25 + 0.75 * np.clip(1.6 - x * 1.3, 0, 1)[..., None]  # خط الليل
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)
    return path


# ═══════════════════════ 2) استخراج الخصائص البصرية ═══════════════════════

def circular_mean_deg(h_deg, weights):
    """متوسط دائري لدرجة اللون (لأن 359° و 1° متجاوران)."""
    rad = np.deg2rad(h_deg)
    s, c = np.sum(weights * np.sin(rad)), np.sum(weights * np.cos(rad))
    return float(np.rad2deg(np.arctan2(s, c)) % 360)


def extract_features(rgb, n_steps):
    """خصائص عامة للصورة + خصائص لكل عمود (كل عمود = نغمة زمنية)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[..., 0] * 2.0          # OpenCV يخزن 0..179
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0
    r = rgb[..., 0].astype(np.float32) / 255.0
    b = rgb[..., 2].astype(np.float32) / 255.0

    global_f = {
        "brightness": float(val.mean()),
        "saturation": float(sat.mean()),
        "hue_deg": circular_mean_deg(hue, sat + 1e-6),
        "contrast": float(val.std()),
        "edge_density": float(edges.mean()),
        "warm_ratio": float(np.mean(r > b)),
    }

    H, W = val.shape
    bounds = np.linspace(0, W, n_steps + 1).astype(int)
    rows = np.arange(H, dtype=np.float32)[:, None]
    columns = []
    for i in range(n_steps):
        sl = slice(bounds[i], max(bounds[i + 1], bounds[i] + 1))
        v, s = val[:, sl], sat[:, sl]
        centroid = float((rows * v).sum() / (v.sum() + 1e-9) / max(H - 1, 1))
        columns.append({
            "brightness": float(v.mean()),
            "saturation": float(s.mean()),
            "hue": circular_mean_deg(hue[:, sl], s + 1e-6),
            "edges": float(edges[:, sl].mean()),
            "height": 1.0 - centroid,  # أين يتركز الضوء عمودياً
            "pan": float(np.clip((r[:, sl].mean() - b[:, sl].mean()) * 2, -1, 1)),
        })
    return global_f, columns


# ═══════════════════ 3) تحويل الخصائص إلى معلمات موسيقية ═══════════════════

def choose_scale(hue, sat):
    if sat < 0.12:
        return "dorian"
    if 75 <= hue < 165:
        return "major_pentatonic"
    if 165 <= hue < 270:
        return "minor_pentatonic"
    return "phrygian"


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def map_to_music(g, cols, ai=None):
    tempo = 70 + g["brightness"] * 70                 # صورة أسطع = إيقاع أسرع
    scale_name = choose_scale(g["hue_deg"], g["saturation"])
    if ai:  # اقتراحات الذكاء الاصطناعي تتقدّم إن وُجدت
        try:
            tempo = float(np.clip(float(ai.get("tempo_bpm", tempo)), 50, 160))
        except (TypeError, ValueError):
            pass
        if ai.get("scale") in SCALES:
            scale_name = ai["scale"]
    scale = SCALES[scale_name]
    root = 48 + int(g["hue_deg"] / 360 * 12)          # اللون يحدد مفتاح المقطوعة

    bright = np.array([c["brightness"] for c in cols])
    height = np.array([c["height"] for c in cols])
    contour = 0.75 * bright + 0.25 * height
    contour = (contour - contour.min()) / (np.ptp(contour) + 1e-9)

    n_deg = len(scale) * 2                            # نطاق أوكتافين
    notes, prev = [], None
    for c, val in zip(cols, contour):
        if c["brightness"] < 0.06:                    # الظلام = صمت
            notes.append(None)
            continue
        idx = int(round(val * (n_deg - 1)))
        if prev is not None:                          # منع القفزات الحادة
            idx = prev + int(np.clip(idx - prev, -3, 3))
        prev = idx
        midi = root + 12 + scale[idx % len(scale)] + 12 * (idx // len(scale))
        notes.append({
            "freq": midi_to_hz(midi),
            "midi": midi,
            "velocity": 0.35 + 0.65 * c["brightness"],
            "timbre": 0.2 + 0.6 * c["saturation"],    # تشبّع أعلى = صوت أغنى
            "pan": c["pan"],
            "hat": c["edges"],
        })

    # خط الباص: كل 8 خطوات (مازورة) جذر مختلف حسب سطوع المقطع
    bass = []
    for start in range(0, len(cols), 8):
        seg = bright[start:start + 8].mean()
        degree = [0, 3, 4, 2][min(int(seg * 4), 3)] % len(scale)
        bass.append(root - 12 + scale[degree])

    return {
        "tempo_bpm": round(tempo, 1),
        "scale": scale_name,
        "root_midi": root,
        "step_sec": 60 / tempo / 2,                   # كل خطوة = ثُمن
        "notes": notes,
        "bass": bass,
        "kick_gain": 0.3 + 0.7 * min(g["contrast"] * 3, 1),
        "reverb_mix": 0.15 + 0.3 * (1 - g["edge_density"] * 5),
    }


# ═══════════════════════════ 4) توليد الصوت ═══════════════════════════

def envelope(n, a, d, s, r):
    a_n = min(int(a * SR), n)
    d_n = min(int(d * SR), n - a_n)
    r_n = min(int(r * SR), n - a_n - d_n)
    s_n = n - a_n - d_n - r_n
    return np.concatenate([
        np.linspace(0, 1, a_n, endpoint=False),
        np.linspace(1, s, d_n, endpoint=False),
        np.full(s_n, s),
        np.linspace(s, 0, r_n),
    ])


def tone(freq, dur, timbre=0.4, adsr=(0.01, 0.1, 0.6, 0.25), harmonics=6):
    """نغمة جمعية (additive): أساسية + توافقيات يتحكم بها 'timbre'."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    wave = np.zeros(n)
    for k in range(1, harmonics + 1):
        wave += (timbre ** (k - 1)) / k * np.sin(2 * np.pi * freq * k * t * (1 + 0.0007 * k))
    return wave / np.max(np.abs(wave) + 1e-9) * envelope(n, *adsr)


def hat(dur=0.06):
    n = int(dur * SR)
    noise = np.random.default_rng(1).uniform(-1, 1, n)
    noise = sosfilt(butter(4, 7000, "hp", fs=SR, output="sos"), noise)
    return noise * np.exp(-np.arange(n) / SR * 60)


def kick(dur=0.3):
    t = np.arange(int(dur * SR)) / SR
    f = 45 + 75 * np.exp(-t * 25)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def place(buf, sig, start, pan=0.0, gain=1.0):
    """يضع إشارة أحادية في الخلفية الاستريو مع توزيع متساوي القدرة."""
    i = int(start * SR)
    j = min(i + len(sig), len(buf))
    if i >= len(buf):
        return
    angle = (pan + 1) * np.pi / 4
    buf[i:j, 0] += sig[: j - i] * gain * np.cos(angle)
    buf[i:j, 1] += sig[: j - i] * gain * np.sin(angle)


def reverb(x, mix, seconds=2.2):
    n = int(seconds * SR)
    rng = np.random.default_rng(3)
    ir = rng.uniform(-1, 1, n) * np.exp(-np.arange(n) / SR * 3.2)
    ir = sosfilt(butter(2, 5000, "lp", fs=SR, output="sos"), ir)
    ir /= np.sum(np.abs(ir))
    wet = np.stack([fftconvolve(x[:, ch], ir)[: len(x)] for ch in range(2)], 1)
    wet *= np.max(np.abs(x)) / (np.max(np.abs(wet)) + 1e-9)
    return (1 - mix) * x + mix * wet


def render(music, tail=3.0):
    step = music["step_sec"]
    total = len(music["notes"]) * step + tail
    buf = np.zeros((int(total * SR), 2))
    root = music["root_midi"]

    # الطبقة 1: وسادة صوتية (Pad) هادئة — الجذر + الخامسة + الأوكتاف
    pad_dur = total - 0.5
    lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.12 * np.arange(int(pad_dur * SR)) / SR)
    for interval, pan in [(0, -0.4), (7, 0.4), (12, 0.0)]:
        p = tone(midi_to_hz(root - 12 + interval), pad_dur, 0.25, (2.5, 0.5, 0.8, 2.5)) * lfo
        place(buf, p, 0, pan, 0.07)

    # الطبقة 2: الباص وضربة الإيقاع مع بداية كل مازورة
    for b, midi in enumerate(music["bass"]):
        t0 = b * 8 * step
        place(buf, tone(midi_to_hz(midi), 8 * step, 0.3, (0.02, 0.3, 0.5, 0.6)), t0, 0, 0.22)
        place(buf, kick(), t0, 0, 0.5 * music["kick_gain"])
        place(buf, kick(), t0 + 4 * step, 0, 0.3 * music["kick_gain"])

    # الطبقة 3: اللحن الرئيسي + صنج خفيف حسب كثافة الحواف
    for i, note in enumerate(music["notes"]):
        if note is None:
            continue
        t0 = i * step
        place(buf, tone(note["freq"], step * 1.8, note["timbre"]), t0, note["pan"] * 0.7,
              0.28 * note["velocity"])
        if note["hat"] > 0.02:
            place(buf, hat(), t0, -note["pan"] * 0.5, min(note["hat"] * 4, 0.25))

    buf = reverb(buf, float(np.clip(music["reverb_mix"], 0.1, 0.45)))
    fade = int(0.05 * SR)
    buf[:fade] *= np.linspace(0, 1, fade)[:, None]
    buf[-int(1.5 * SR):] *= np.linspace(1, 0, int(1.5 * SR))[:, None]
    return buf / (np.max(np.abs(buf)) + 1e-9) * 0.9


def save_wav(path, audio):
    wavfile.write(path, SR, (audio * 32767).astype(np.int16))


# ═══════════════════ اختياري: تحليل الصورة بـ OpenAI ═══════════════════

AI_PROMPT = (
    "You are an Earth-science sonification assistant. Look at this NASA Earth "
    "image and reply with JSON only, with keys: description (2 sentences), "
    "phenomenon (e.g. wildfire, hurricane, vegetation, sea ice), mood (one word), "
    "tempo_bpm (50-160), scale (one of: " + ", ".join(SCALES) + ")."
)


def ai_analyze(image_path, model=None, api_key=None):
    """يرسل الصورة إلى نموذج رؤية من OpenAI ويعيد وصفاً ومقترحات موسيقية."""
    from openai import OpenAI  # استيراد متأخر كي لا تُطلب المكتبة بدون --ai

    # إن لم يُمرَّر مفتاح، يُقرأ تلقائياً من متغير البيئة OPENAI_API_KEY
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((1024, 1024))
    bio = io.BytesIO()
    img.save(bio, "JPEG", quality=85)
    b64 = base64.b64encode(bio.getvalue()).decode()

    resp = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": AI_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    )
    return json.loads(resp.choices[0].message.content)


# ═══════════════════════════════ التشغيل ═══════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Spacoust: NASA Earth image → music")
    ap.add_argument("image", nargs="?", help="مسار الصورة (يُولَّد مثال إن تُرك فارغاً)")
    ap.add_argument("-o", "--output", help="ملف الإخراج .wav")
    ap.add_argument("--steps", type=int, default=64, help="عدد النغمات (أعمدة المسح)")
    ap.add_argument("--ai", action="store_true", help="تحليل الصورة بـ OpenAI أولاً")
    args = ap.parse_args()

    image = args.image or make_demo_image("demo_earth.png")
    out = args.output or f"{Path(image).stem}_spacoust.wav"

    ai = None
    if args.ai:
        try:
            ai = ai_analyze(image)
            print(f"🤖 AI: {ai.get('description', '')}")
        except Exception as e:  # نكمل بدون الذكاء الاصطناعي إن فشل الاتصال
            print(f"⚠️  تعذّر تحليل OpenAI ({e}) — المتابعة بالتحليل البصري فقط")

    g, cols = extract_features(load_image(image), args.steps)
    music = map_to_music(g, cols, ai)
    audio = render(music)
    save_wav(out, audio)

    report = {
        "image": str(image), "output": out, "features": g, "ai": ai,
        "music": {k: music[k] for k in ("tempo_bpm", "scale", "root_midi", "bass")},
        "melody_midi": [n["midi"] if n else None for n in music["notes"]],
        "duration_sec": round(len(audio) / SR, 2),
    }
    Path(out).with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"🎵 {out}  |  {music['tempo_bpm']} BPM  |  {music['scale']}  |  "
          f"{report['duration_sec']}s")


if __name__ == "__main__":
    main()
