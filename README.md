# Spacoust 🌍🎵

**Hear the planet.** Spacoust turns NASA Earth images into music.
Built for the NASA Space Apps Challenge — *The Earth Information Jukebox*.

<p align="center"><img src="logo.png" width="320" alt="Spacoust logo"></p>

## ⬇️ Download

**[Download Spacoust for Windows](https://github.com/najh9090/spacoust/releases/latest/download/Spacoust.exe)** · [Website](https://najh9090.github.io/spacoust/)

Windows 10/11, no installation. If Windows shows "Windows protected your PC", click **More info → Run anyway**.

## How it works

1. **Open an image**: any satellite image or Earth data map.
2. **AI reads the scene**: OpenAI vision names the phenomenon and sets mood and tempo.
3. **Listen**: the image is scanned left to right and turned into music you can save as WAV.

| Image | Sound |
|---|---|
| Brightness | Pitch (and overall tempo) |
| Darkness | Silence |
| Saturation | Tone color |
| Edges | Percussion |
| Dominant color | Scale: green → major pentatonic, blue → minor pentatonic, red → phrygian, grey → dorian |

The same image always produces the same music: it is computed from the data, not random.

## Run from source

```
pip install -r requirements.txt
python desktop_app.py
```

Put your OpenAI key in a file named `api_key.txt` next to the app (one line starting with `sk-`). Never commit it.

Build the Windows app: double-click `build_exe.bat` → `dist\Spacoust.exe`.

## Files

| File | |
|---|---|
| `desktop_app.py` | Desktop app |
| `spacoust.py` | Sonification engine (also a command-line tool) |
| `build_exe.bat` | Builds `Spacoust.exe` |
| `docs/` | Website (GitHub Pages) |
