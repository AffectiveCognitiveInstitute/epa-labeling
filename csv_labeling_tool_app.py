
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV Labeling Web UI (Flask)
- Upload a CSV (expects: item_id,condition,participant_id,chat_id,turn_id,text,coder_1,coder_2,coder_3,coder_4,_coder_5)
- Choose coder (1..5)
- Label each item with one of 10 labels via large selectable cards
- Auto-saves label immediately back to the CSV stored on the server
- Resume: opens at the first unlabeled row for that coder; if complete, starts at row 1
- Navigation: previous/next buttons + direct jump
- Download current CSV anytime

Run:
  python app.py
Then open:
  http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import io
import csv
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import markdown

import pandas as pd
from flask import Flask, request, redirect, url_for, send_file, render_template_string, flash

APP_TITLE = "CSV Labeling Tool"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CURRENT_CSV_PATH = os.path.join(DATA_DIR, "current.csv")

import json

SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

DEFAULT_CODER_NAMES = {
    "1": "Coder 1",
    "2": "Coder 2",
    "3": "Coder 3",
    "4": "Coder 4",
    "5": "Coder 5",
}

def load_settings() -> dict:
    ensure_data_dir()
    if not os.path.exists(SETTINGS_PATH):
        return {"coder_names": DEFAULT_CODER_NAMES.copy()}
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        if "coder_names" not in s or not isinstance(s["coder_names"], dict):
            s["coder_names"] = DEFAULT_CODER_NAMES.copy()
        # Ensure all keys exist
        for k, v in DEFAULT_CODER_NAMES.items():
            s["coder_names"].setdefault(k, v)
        return s
    except Exception:
        return {"coder_names": DEFAULT_CODER_NAMES.copy()}

def save_settings(s: dict) -> None:
    ensure_data_dir()
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

def get_coder_name(coder: int) -> str:
    s = load_settings()
    return (s.get("coder_names", {}).get(str(coder)) or f"Coder {coder}").strip()

# Thread-safety for file read/write
CSV_LOCK = threading.Lock()

# Label taxonomy (key -> long description)
LABELS: List[Tuple[str, str]] = [
    ("help",       "Praktisch helfen; konkrete Unterstützung oder Umsetzung anbieten"),
    ("listen",     "Zuhören; nachfragen, klären oder Raum geben, damit die Person mehr schildert"),
    ("comfort",    "Trösten oder validieren; Gefühle anerkennen und normalisieren"),
    ("encourage",  "Ermutigen oder bestärken; Motivation, Zuversicht oder Selbstwirksamkeit fördern"),
    ("explain",    "Erklären oder informieren; sachliche Zusammenhänge oder Hintergründe erläutern"),
    ("advise",     "Raten; Strategien, Empfehlungen oder Optionen vorschlagen"),
    ("cooperate",  "Gemeinsam planen oder kooperieren; explizites gemeinsames Vorgehen anbieten"),
    ("warn",       "Warnen; Risiken, Druck oder mögliche negative Konsequenzen betonen"),
    ("ignore",     "Ignorieren; auf die geäußerte Sorge nicht eingehen oder am Thema vorbeireden"),
    ("denigrate",  "Abwerten; beschämen, herabsetzen oder die Person negativ beurteilen"),
]

# Canonical expected columns (we will tolerate small variations and auto-add missing coder columns)
EXPECTED_BASE_COLS = ["item_id", "condition", "participant_id", "chat_id", "turn_id", "text"]
CODER_COLS_CANON = ["coder_1", "coder_2", "coder_3", "coder_4", "_coder_5"]  # as provided

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes common column naming mistakes:
    - Strips whitespace
    - Converts 'coder5' / 'coder_5' to '_coder_5' if needed
    - Ensures coder columns exist
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # If user provided "coder_5" but not "_coder_5", rename to "_coder_5"
    if "coder_5" in df.columns and "_coder_5" not in df.columns:
        df = df.rename(columns={"coder_5": "_coder_5"})

    # If user provided "_coder5" etc.
    for alt in ["_coder5", "coder5", "Coder_5", "Coder5"]:
        if alt in df.columns and "_coder_5" not in df.columns:
            df = df.rename(columns={alt: "_coder_5"})

    # Ensure all coder columns exist
    for col in CODER_COLS_CANON:
        if col not in df.columns:
            df[col] = ""

    # Ensure base cols exist; if not, we keep but warn in UI; minimal requirement is text
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")

    # Keep as strings (avoid NaN)
    for col in CODER_COLS_CANON:
        df[col] = df[col].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    return df

def read_current_df() -> Optional[pd.DataFrame]:
    if not os.path.exists(CURRENT_CSV_PATH):
        return None
    with CSV_LOCK:
        df = pd.read_csv(CURRENT_CSV_PATH, dtype=str, keep_default_na=False)
    df = normalize_columns(df)
    return df

def write_current_df(df: pd.DataFrame) -> None:
    ensure_data_dir()
    with CSV_LOCK:
        df.to_csv(CURRENT_CSV_PATH, index=False)

def coder_to_col(coder: int) -> str:
    if coder in [1,2,3,4]:
        return f"coder_{coder}"
    if coder == 5:
        return "_coder_5"
    raise ValueError("Coder must be in 1..5")

def first_unlabeled_index(df: pd.DataFrame, coder_col: str) -> int:
    vals = df[coder_col].fillna("").astype(str)
    for i, v in enumerate(vals.tolist()):
        if v.strip() == "":
            return i
    return 0  # if complete, go back to start

def labeled_count(df: pd.DataFrame, coder_col: str) -> int:
    return int((df[coder_col].fillna("").astype(str).str.strip() != "").sum())

def wrap_index(idx: int, n: int) -> int:
    if n <= 0:
        return 0
    return idx % n

# --- Flask app ---

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

BASE_HTML = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{title}}</title>
  <style>
    :root{
      --bg:#0b0f17;
      --card:#121a2a;
      --card2:#0f1524;
      --text:#e6e9f2;
      --muted:#a9b1c7;
      --border:#24304b;
      --accent:#6ea8fe;
      --good:#8be9a8;
      --warn:#ffd27d;
      --danger:#ff7d7d;
    }
    body{ margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", "Liberation Sans", sans-serif;
          background:var(--bg); color:var(--text); }
    a{ color: var(--accent); text-decoration:none; }
    .container{ max-width:1100px; margin:0 auto; padding:24px; }
    .header{ display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom:18px;}
    .title{ font-size:20px; font-weight:700; }
    .pill{ display:inline-block; padding:6px 10px; border:1px solid var(--border); border-radius:999px; color:var(--muted); background: rgba(255,255,255,0.02); }
    .panel{ background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border:1px solid var(--border);
            border-radius:14px; padding:18px; }
    .row{ display:flex; gap:16px; flex-wrap:wrap;}
    .col{ flex:1 1 340px; }
    label{ display:block; font-size:13px; color:var(--muted); margin-bottom:6px;}
    input[type="file"], select, input[type="number"]{
      width:100%; padding:10px 12px; border-radius:10px; border:1px solid var(--border); background:var(--card2); color:var(--text);
      outline:none;
    }
    button, .btn{
      border:1px solid var(--border); background:var(--card); color:var(--text);
      padding:10px 14px; border-radius:10px; cursor:pointer; font-weight:600;
    }
    button:hover, .btn:hover{ border-color: rgba(110,168,254,0.6); }
    .btn-primary{ border-color: rgba(110,168,254,0.7); }
    .btn-danger{ border-color: rgba(255,125,125,0.7); }
    .btn-row{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    .muted{ color:var(--muted); }
    .flash{ margin: 0 0 14px 0; padding: 10px 12px; border-radius: 10px; border: 1px solid rgba(255,210,125,0.6); background: rgba(255,210,125,0.08); color: var(--text); }
    .textbox{
      white-space: pre-wrap;
      line-height: 1.5;
      font-size: 15px;
      padding: 14px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
      max-height: 320px;
      overflow: auto;
    }
    .meta{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
    .grid{
      display:grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    @media (max-width: 850px){
      .grid{ grid-template-columns: 1fr; }
    }
    .labelcard{
      border:1px solid var(--border);
      background: rgba(255,255,255,0.02);
      padding: 12px 12px;
      border-radius: 12px;
      text-align:left;
      cursor:pointer;
      transition: transform 0.02s ease-in-out, border-color 0.08s ease-in-out, background 0.08s ease-in-out;
    }
    .labelcard:hover{ border-color: rgba(110,168,254,0.65); }
    .labelcard.selected{
      border-color: rgba(139,233,168,0.85);
      background: rgba(139,233,168,0.08);
      box-shadow: 0 0 0 1px rgba(139,233,168,0.12) inset;
    }
    .labelkey{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }
    .labeldesc{ font-size: 14px; }
    .progresswrap{ display:flex; align-items:center; gap: 12px; flex-wrap:wrap; margin-top: 16px; }
    .progressbar{
      flex: 1 1 360px;
      height: 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
      overflow:hidden;
    }
    .progressfill{
      height:100%;
      width: {{progress_pct}}%;
      background: rgba(110,168,254,0.75);
    }
    .footerrow{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-top: 14px;}
    .small{ font-size: 12px; }
    .center{ text-align:center; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title">{{title}}</div>
      <div class="btn-row">
        {% if has_csv %}
          <a class="btn" href="{{ url_for('download_csv') }}">CSV herunterladen</a>
        {% endif %}
        <a class="btn" href="{{ url_for('index') }}">Start</a>
      </div>
    </div>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for msg in messages %}
          <div class="flash">{{msg}}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    {{ body|safe }}
  </div>
</body>
</html>
"""

INDEX_BODY = r"""
<div class="panel">
  {% if not has_csv %}
    <div class="muted">Noch keine CSV geladen. Bitte CSV hochladen, um zu starten.</div>
  {% else %}
    <div class="meta">
      <span class="pill">Datensätze: <b>{{n_rows}}</b></span>
      <span class="pill">Spalten: <b>{{n_cols}}</b></span>
      <span class="pill">Pflichtspalte: <b>text</b></span>
    </div>
    <div class="muted small">Hinweis: Labels werden serverseitig direkt in die aktuell geladene CSV geschrieben (Spalte des gewählten Coders).</div>
  {% endif %}
</div>

<div style="height:14px;"></div>

<div class="row">
  <div class="col panel">
    <div style="font-weight:700; margin-bottom:10px;">1) CSV auswählen</div>
    <form action="{{ url_for('upload_csv') }}" method="post" enctype="multipart/form-data">
      <label>CSV hochladen</label>
      <input type="file" name="file" accept=".csv,text/csv" required />
      <div style="height:10px;"></div>
      <button class="btn-primary" type="submit">CSV laden / ersetzen</button>
    </form>
    <div style="height:12px;"></div>
    <div class="muted small">
      Beim Upload wird die App zurückgesetzt (neue Datenbasis). Fehlende Coder-Spalten werden automatisch ergänzt.
    </div>
  </div>

  <div class="col panel" style="opacity: {{ 1.0 if has_csv else 0.5 }};">
    <div style="font-weight:700; margin-bottom:10px;">2) Coder auswählen</div>

    <form action="{{ url_for('start_labeling') }}" method="get">
      <label>Coder</label>
      <select name="coder" {{ "disabled" if not has_csv else "" }}>
        <option value="1">{{ coder_names['1'] }} (coder_1)</option>
        <option value="2">{{ coder_names['2'] }} (coder_1)</option>
        <option value="3">{{ coder_names['3'] }} (coder_1)</option>
        <option value="4">{{ coder_names['4'] }} (coder_1)</option>
        <option value="5">{{ coder_names['5'] }} (coder_1)</option>
      </select>
      <div style="height:12px;"></div>

      {% if has_csv %}
        <div class="muted small">Fortschritt pro Coder (bereits gelabelt):</div>
        <div style="height:8px;"></div>
        <div class="meta">
          {% for c in coder_stats %}
            <span class="pill">{{c.name}}: <b>{{c.done}}</b>/{{c.total}}</span>
          {% endfor %}
        </div>
      {% endif %}

      <div style="height:10px;"></div>
      <button class="btn-primary" type="submit" {{ "disabled" if not has_csv else "" }}>Labeling starten / fortsetzen</button>
    </form>
  </div>
</div>

<div style="height:14px;"></div>
<div style="font-weight:700; margin-bottom:10px;">Coder-Namen (Anzeige)</div>

<form action="{{ url_for('save_coder_names') }}" method="post">
  <div class="row">
    <div class="col">
      <label>Coder 1 Name</label>
      <input name="name1" value="{{ coder_names['1'] }}" />
      <div style="height:10px;"></div>

      <label>Coder 2 Name</label>
      <input name="name2" value="{{ coder_names['2'] }}" />
      <div style="height:10px;"></div>

      <label>Coder 3 Name</label>
      <input name="name3" value="{{ coder_names['3'] }}" />
    </div>

    <div class="col">
      <label>Coder 4 Name</label>
      <input name="name4" value="{{ coder_names['4'] }}" />
      <div style="height:10px;"></div>

      <label>Coder 5 Name</label>
      <input name="name5" value="{{ coder_names['5'] }}" />
    </div>
  </div>

  <div style="height:12px;"></div>
  <button class="btn" type="submit" {{ "disabled" if not has_csv else "" }}>Namen speichern</button>
  <div class="muted small" style="margin-top:8px;">
    Diese Namen ändern nur die Anzeige im UI. CSV-Spalten bleiben unverändert.
  </div>
</form>
"""

LABEL_BODY = r"""
<div class="panel">
  <div class="meta">
    <span class="pill">Coder: <b>{{coder_name}}</b> → <span class="muted">{{coder_col}}</span></span>
    <span class="pill">Datensatz: <b>{{idx_display}}</b>/{{n}}</span>
    <span class="pill">Gelabelt: <b>{{done}}</b>/{{n}}</span>
  </div>

  <div class="textbox">{{ text|safe }}</div>

  <div class="grid">
    {% for key, desc in labels %}
      <form method="post" action="{{ url_for('set_label') }}">
        <input type="hidden" name="coder" value="{{coder}}" />
        <input type="hidden" name="idx" value="{{idx}}" />
        <input type="hidden" name="label" value="{{key}}" />
        <button class="labelcard {{ 'selected' if current_label == key else '' }}" type="submit">
          <div class="labelkey">{{key}}</div>
          <div class="labeldesc">{{desc}}</div>
        </button>
      </form>
    {% endfor %}
  </div>

  <div class="progresswrap">
    <div class="progressbar" aria-label="Progress">
      <div class="progressfill"></div>
    </div>

    <div class="btn-row">
      <a class="btn" href="{{ url_for('label_view', coder=coder, idx=prev_idx) }}">← Zurück</a>
      <a class="btn" href="{{ url_for('label_view', coder=coder, idx=next_idx) }}">Weiter →</a>
    </div>

    <form method="get" action="{{ url_for('label_view') }}" style="display:flex; gap:10px; align-items:center;">
      <input type="hidden" name="coder" value="{{coder}}" />
      <input type="number" name="idx" min="1" max="{{n}}" value="{{idx_display}}" style="width:110px;" />
      <button class="btn" type="submit">Springen</button>
    </form>
  </div>

  <div class="footerrow">
    <div class="muted small">
      Aktuelles Label: <b>{{ current_label if current_label else "—" }}</b>
      {% if current_label %}
        <span class="muted"> (klicken Sie einfach ein anderes Label, um zu überschreiben)</span>
      {% endif %}
    </div>

    <form method="post" action="{{ url_for('clear_label') }}">
      <input type="hidden" name="coder" value="{{coder}}" />
      <input type="hidden" name="idx" value="{{idx}}" />
      <button class="btn-danger" type="submit">Label entfernen</button>
    </form>
  </div>
</div>
"""

DONE_BODY = r"""
<div class="panel center">
  <div style="font-size:20px; font-weight:800; margin-bottom:8px;">Danke.</div>
  <div class="muted" style="margin-bottom:14px;">
    Alle Datensätze für <b>{{coder_name}}</b> sind gelabelt.
  </div>

  <div class="btn-row" style="justify-content:center;">
    <a class="btn-primary btn" href="{{ url_for('download_csv') }}">CSV herunterladen</a>
    <a class="btn" href="{{ url_for('index') }}">Zurück zum Start</a>
  </div>

  <div style="height:10px;"></div>
  <div class="muted small">
    Wenn Sie erneut starten, beginnt die Ansicht wieder bei Datensatz 1.
  </div>
</div>
"""

@dataclass
class CoderStat:
    name: str
    done: int
    total: int

def render_page(body_html: str, **ctx):
    return render_template_string(
        BASE_HTML,
        title=ctx.get("title", APP_TITLE),
        body=render_template_string(body_html, **ctx),
        has_csv=ctx.get("has_csv", False),
        progress_pct=ctx.get("progress_pct", 0),
    )

@app.get("/")
def index():
    settings = load_settings()
    coder_names = settings["coder_names"]

    df = read_current_df()
    has_csv = df is not None
    coder_stats = []
    if has_csv:
        total = len(df)
        for c in [1,2,3,4,5]:
            col = coder_to_col(c)
            coder_stats.append(CoderStat(
                name=get_coder_name(c),
                done=labeled_count(df, col),
                total=total
            ))
    return render_page(
        INDEX_BODY,
        title=APP_TITLE,
        has_csv=has_csv,
        n_rows=(len(df) if has_csv else 0),
        n_cols=(len(df.columns) if has_csv else 0),
        coder_stats=coder_stats,
        coder_names=coder_names
    )

@app.post("/upload")
def upload_csv():
    ensure_data_dir()
    if "file" not in request.files:
        flash("Kein Upload gefunden.")
        return redirect(url_for("index"))
    f = request.files["file"]
    if not f or f.filename.strip() == "":
        flash("Bitte eine CSV-Datei auswählen.")
        return redirect(url_for("index"))

    try:
        content = f.read()
        # Try utf-8-sig first, then latin-1 as fallback
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        df = normalize_columns(df)
        write_current_df(df)
    except Exception as e:
        flash(f"CSV konnte nicht geladen werden: {e}")
        return redirect(url_for("index"))

    flash("CSV erfolgreich geladen. Sie können jetzt einen Coder auswählen.")
    return redirect(url_for("index"))

@app.get("/start")
def start_labeling():
    df = read_current_df()
    if df is None:
        flash("Bitte zuerst eine CSV hochladen.")
        return redirect(url_for("index"))

    try:
        coder = int(request.args.get("coder", "1"))
        coder_col = coder_to_col(coder)
    except Exception:
        flash("Ungültiger Coder.")
        return redirect(url_for("index"))

    idx = first_unlabeled_index(df, coder_col)
    # If complete, show done page (but start over at 0 if user continues)
    if labeled_count(df, coder_col) >= len(df) and len(df) > 0:
        return redirect(url_for("done_view", coder=coder))
    return redirect(url_for("label_view", coder=coder, idx=idx+1))  # 1-based in URL

@app.get("/label")
def label_view():
    df = read_current_df()
    if df is None:
        flash("Bitte zuerst eine CSV hochladen.")
        return redirect(url_for("index"))

    try:
        coder = int(request.args.get("coder", "1"))
        coder_col = coder_to_col(coder)
    except Exception:
        flash("Ungültiger Coder.")
        return redirect(url_for("index"))

    n = len(df)
    if n == 0:
        flash("CSV enthält keine Datensätze.")
        return redirect(url_for("index"))

    # idx is 1-based in URL to make jumping easier for humans
    try:
        idx_1based = int(request.args.get("idx", "1"))
    except Exception:
        idx_1based = 1
    idx = wrap_index(idx_1based - 1, n)

    done = labeled_count(df, coder_col)
    # If complete and user opened without explicit idx, show done
    if done >= n and "idx" not in request.args:
        return redirect(url_for("done_view", coder=coder))

    row = df.iloc[idx].to_dict()
    #text = row.get("text", "")
    raw_text = row.get("text", "")

    text_html = markdown.markdown(
        raw_text,
        extensions=[
            "extra",        # Tabellen, Definition Lists, etc.
            "sane_lists",   # saubere Listen
            "nl2br"         # Zeilenumbrüche → <br>
        ]
    )

    current_label = (row.get(coder_col, "") or "").strip()
    # Derive progress percentage as done / total
    progress_pct = 0 if n == 0 else int(round((done / n) * 100))

    prev_idx = wrap_index(idx - 1, n) + 1
    next_idx = wrap_index(idx + 1, n) + 1

    return render_page(
        LABEL_BODY,
        title=f"{APP_TITLE} – Coder {coder}",
        has_csv=True,
        coder=coder,
        coder_name=get_coder_name(coder),
        coder_col=coder_col,
        idx=idx,
        idx_display=idx+1,
        n=n,
        done=done,
        text=text_html,
        labels=LABELS,
        current_label=current_label,
        prev_idx=prev_idx,
        next_idx=next_idx,
        progress_pct=progress_pct
    )

@app.post("/set_label")
def set_label():
    df = read_current_df()
    if df is None:
        flash("Bitte zuerst eine CSV hochladen.")
        return redirect(url_for("index"))

    try:
        coder = int(request.form.get("coder", "1"))
        coder_col = coder_to_col(coder)
        idx = int(request.form.get("idx", "0"))
    except Exception:
        flash("Ungültige Anfrage.")
        return redirect(url_for("index"))

    label = (request.form.get("label", "") or "").strip()
    valid_keys = {k for k, _ in LABELS}
    if label not in valid_keys:
        flash("Ungültiges Label.")
        return redirect(url_for("label_view", coder=coder, idx=idx+1))

    n = len(df)
    if n == 0:
        flash("CSV enthält keine Datensätze.")
        return redirect(url_for("index"))
    idx = wrap_index(idx, n)

    df.at[idx, coder_col] = label
    write_current_df(df)

    # Auto-advance to next item; if completed now, show done page
    done = labeled_count(df, coder_col)
    if done >= n:
        return redirect(url_for("done_view", coder=coder))

    next_idx = wrap_index(idx + 1, n) + 1
    return redirect(url_for("label_view", coder=coder, idx=next_idx))

@app.post("/clear_label")
def clear_label():
    df = read_current_df()
    if df is None:
        flash("Bitte zuerst eine CSV hochladen.")
        return redirect(url_for("index"))

    try:
        coder = int(request.form.get("coder", "1"))
        coder_col = coder_to_col(coder)
        idx = int(request.form.get("idx", "0"))
    except Exception:
        flash("Ungültige Anfrage.")
        return redirect(url_for("index"))

    n = len(df)
    if n == 0:
        flash("CSV enthält keine Datensätze.")
        return redirect(url_for("index"))
    idx = wrap_index(idx, n)

    df.at[idx, coder_col] = ""
    write_current_df(df)
    return redirect(url_for("label_view", coder=coder, idx=idx+1))

@app.get("/done")
def done_view():
    df = read_current_df()
    if df is None:
        flash("Bitte zuerst eine CSV hochladen.")
        return redirect(url_for("index"))

    try:
        coder = int(request.args.get("coder", "1"))
        coder_to_col(coder)
    except Exception:
        flash("Ungültiger Coder.")
        return redirect(url_for("index"))

    return render_page(
        DONE_BODY,
        title=f"{APP_TITLE} – Fertig",
        has_csv=True,
        coder=coder,
        coder_name=get_coder_name(coder),
        progress_pct=100
    )

@app.get("/download")
def download_csv():
    df = read_current_df()
    if df is None or len(df) == 0:
        flash("Keine CSV geladen.")
        return redirect(url_for("index"))
    # Always write out normalized copy to ensure consistent schema
    out = io.StringIO()
    df.to_csv(out, index=False)
    mem = io.BytesIO(out.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="labeled.csv")

@app.post("/settings/coder-names")
def save_coder_names():
    s = load_settings()
    cn = s.get("coder_names", DEFAULT_CODER_NAMES.copy())

    def clean(x: str, fallback: str) -> str:
        x = (x or "").strip()
        return x if x else fallback

    cn["1"] = clean(request.form.get("name1"), "Coder 1")
    cn["2"] = clean(request.form.get("name2"), "Coder 2")
    cn["3"] = clean(request.form.get("name3"), "Coder 3")
    cn["4"] = clean(request.form.get("name4"), "Coder 4")
    cn["5"] = clean(request.form.get("name5"), "Coder 5")

    s["coder_names"] = cn
    save_settings(s)
    flash("Coder-Namen gespeichert.")
    return redirect(url_for("index"))

if __name__ == "__main__":
    ensure_data_dir()

    port = int(os.environ.get("PORT", "5000"))

    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "on")
    host = "127.0.0.1" if debug else "0.0.0.0"

    app.run(host=host, port=port, debug=debug)
