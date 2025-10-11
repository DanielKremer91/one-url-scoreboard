# app.py
# ONE URL Scoreboard — Streamlit App
# Idea: Daniel Kremer (ONE Beyond Search) — Implementation by ChatGPT
# Master-URL modes: Union (default), Own upload, Merge from up to two files, Pick one file, Intersection (all uploads)
# Criteria grouped by clusters; main_kw_exp supports direct value or SV×CTR(position)

import io
import json
import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from collections import Counter
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import numpy as np
import pandas as pd
import streamlit as st

# ============= Page & CSS =============
st.set_page_config(page_title="ONE URL Scoreboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.main .block-container { max-width: 100% !important; padding: 1rem 1rem 2rem !important; }
[data-testid="stSidebar"] { min-width: 260px !important; max-width: 260px !important; }
</style>
""", unsafe_allow_html=True)

# ============= Branding =============
try:
    st.image("https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png", width=250)
except Exception:
    pass

st.title("ONE URL Scoreboard")
st.markdown("""
<div style="background-color:#f2f2f2;color:#000;padding:15px 20px;border-radius:6px;font-size:.9em;max-width:850px;margin-bottom:1.5em;line-height:1.5;">
  Entwickelt von <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">Daniel Kremer</a> von
  <a href="https://onebeyondsearch.com/" target="_blank">ONE Beyond Search</a> &nbsp;|&nbsp;
  Folge mir auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a> für mehr SEO-Insights und Tool-Updates
</div>
<hr>
""", unsafe_allow_html=True)

# ============= Hilfe =============
with st.expander("❓ Hilfe / Tool-Dokumentation", expanded=False):
    st.markdown("""
**ONE URL Scoreboard** priorisiert URLs über wählbare Kriterien (globaler Scoring-Modus: *Rank linear* oder *Perzentil-Buckets*).  
Fehlende Daten im **aktiven** Kriterium ⇒ Score = 0 (kein Reweighting pro URL).

**Wichtig**
- **Master-URL-Liste**: Modi **Union** (Voreinstellung), **Eigene Masterliste**, **Merge aus bis zu zwei Dateien**, **Basis „eine Datei“**, **Schnittmenge (alle Uploads)**.
- **All-Inlinks-Datei** (Crawl/Kantenliste) wird **in die Union aufgenommen** (empfangende Ziel-URLs).
- **Search Console Query-Level** wird **automatisch** pro URL aggregiert.
- **Embeddings robust**: Fehlende Embeddings ⇒ Outlier (unter τ). Uneinheitliche Längen ⇒ Padding/Trunc auf dominante Dimension.
- **CSV-Fallback**: Falls eine Datei nur **eine** Spalte hat, wird automatisch nach `;` `,` `|` oder Tab gesplittet; **erste Zeile = Header**.
- **Kein globaler Prioritätsfaktor**. **Strategische Priorität** per URL bleibt optional.
""")

# ============= Cache leeren Button (gegen stale Session-State) =============
def _reset_all_state():
    for k in [
        "uploads","column_maps","schema_index",
        "llm_bot_detected","llm_bot_include","llm_bot_exclude",
        "llm_bot_custom_include","llm_bot_custom_exclude",
        "llm_crawl_mode","llm_bot_column_choices",
        "offtopic_tau"
    ]:
        if k in st.session_state:
            del st.session_state[k]

with st.sidebar:
    if st.button("🧹 Cache leeren (Uploads & Mappings)"):
        _reset_all_state()
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

# ============= Session & Helpers =============
if "uploads" not in st.session_state:
    st.session_state.uploads = {}  # key -> (df, name)
if "column_maps" not in st.session_state:
    st.session_state.column_maps = {}
if "schema_index" not in st.session_state:
    st.session_state.schema_index = {}
if "llm_bot_detected" not in st.session_state:
    st.session_state.llm_bot_detected = {}  # {bot_name: count}
if "llm_bot_include" not in st.session_state:
    st.session_state.llm_bot_include = []
if "llm_bot_exclude" not in st.session_state:
    st.session_state.llm_bot_exclude = []
if "llm_bot_custom_include" not in st.session_state:
    st.session_state.llm_bot_custom_include = ""
if "llm_bot_custom_exclude" not in st.session_state:
    st.session_state.llm_bot_custom_exclude = ""
st.session_state.setdefault("llm_crawl_mode", None)
st.session_state.setdefault("llm_bot_column_choices", [])

ALIASES = {
    "url": ["url","page","page_url","seite","address","adresse","target","ziel","ziel_url","landing_page","current_url"],
    "clicks": ["clicks","klicks","traffic","besuche","sc_clicks"],
    "impressions": ["impressions","impr","impressionen","search_impressions"],
    "position": ["position","avg_position","ranking","ranking_position","average_position","durchschnittliche_position","durchschn._position","rank","avg_rank","current_position"],
    "search_volume": ["search_volume","sv","volume","suchvolumen","sv_monat"],
    "cpc": ["cpc","cost_per_click"],
    "traffic_value": ["traffic_value","otv","organic_traffic_value","value","potential_value","trafficwert","traffic_wert"],
    "potential_traffic_url": ["potential_traffic_url","potential_traffic","pot_traffic","potentielle_klicks","erwartete_klicks","erwarteter_traffic","estimated_traffic","estimated_clicks"],
    "backlinks": ["backlinks","links_total","bl","inbound_links","links_to_target"],
    "ref_domains": ["ref_domains","referring_domains","rd","domains_ref","verweisende_domains","referring_domains"],
    "unique_inlinks": ["unique_inlinks","internal_inlinks","inlinks_unique","eingehenden_links","inlinks","eingehende_links","inlinks_unique_count","incoming_links_unique"],
    "llm_ref_traffic": ["llm_ref_traffic","llm_referrals","ai_referrals","llm_popularity","llm_traffic","sessions","sitzungen","visits","hits","traffic"],
    "llm_crawl_freq": ["llm_crawl_freq","ai_crawls","llm_crawls","llm_crawler_visits","crawls","visits","hits","requests"],
    "user_agent": ["user_agent","ua","agent","crawler","bot","useragent"],
    "embedding": ["embedding","embeddings","vector","vec","embedding_json"],
    "revenue": ["revenue","umsatz","organic_revenue","organic_umsatz","organic_sales"],
    "priority_factor": ["priority_factor","prio","priority","priorität","gewicht","gewichtung","boost","override","weight_override","wichtigkeit","boost_faktor","faktor","manual_weight"],
    "keyword": ["keyword","query","suchbegriff","suchanfrage"],
    "main_keyword": ["main_keyword","hauptkeyword","primary_keyword","focus_keyword","focus_kw","haupt_kw","haupt-keyword"],
    "overall_traffic": ["overall_traffic","traffic","sessions","overall_sessions","sessions_total","besuche","gesamt_traffic","gesamt_sessions","visits","overall_clicks","clicks_total","klicks_gesamt"],
    "expected_clicks": ["expected_clicks","exp_clicks","expected_clicks_main","expected_clicks_kw","erwartete_klicks","erw_klicks"],
    "llm_citations": ["llm_citations","citations","llm_mentions","llm_refs","llm_links","citations_total"],
}

TRACKING_PARAMS_PREFIXES = ["utm_", "icid_"]
TRACKING_PARAMS_EXACT = {"gclid","fbclid","msclkid","mc_eid","yclid","twclid","igshid"}  # erweitert

EXCLUDE_CLASSIC_BOTS = ["googlebot", "googlebot smartphone", "bingbot", "yandex", "baidu"]
INCLUDE_AI_BOTS = [
    "gptbot", "openai", "anthropic", "claudebot", "perplexitybot",
    "perplexity", "cohere", "ccbot", "bytespider", "google-extended",
    "meta-ai", "facebook ai", "chatgpt", "duckassist", "quorabots", "youbot", "kagi"
]
GENERIC_BOT_TOKENS = [
    "bot", "ai", "gpt", "claude", "perplexity", "cohere", "bytespider", "ccbot",
    "facebook ai", "google-extended", "openai", "anthropic", "metabot", "meta-ai", "youbot", "duckassist","oai","oai-searchbot"
]

# ---- Normalisierung & Utilities ----
def _alias_norm(x: str) -> str:
    return re.sub(r"[^\w]+", "", str(x).lower())

def _clean_invisibles(x: str) -> str:
    return "".join(ch for ch in str(x) if unicodedata.category(ch) != "Cf")

def normalize_header(col: str) -> str:
    c = _clean_invisibles(str(col)).strip().lower()
    c = re.sub(r"[^\w]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_header(c) for c in df.columns]
    return df

def to_numeric_smart(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    # Wenn alles NaN: evtl. deutsches Dezimalformat
    if out.notna().sum() == 0 and s.astype(str).str.contains(r"\d+,\d+", regex=True).any():
        s2 = (
            s.astype(str)
             .str.replace(".", "", regex=False)  # Tausenderpunkt raus
             .str.replace(",", ".", regex=False)  # Komma -> Punkt
        )
        out = pd.to_numeric(s2, errors="coerce")
    return out

# ---- Tracking-Query säubern & URL robust normalisieren ----
def normalize_url(u: str) -> Optional[str]:
    """
    Robust gegen schemalose Eingaben wie 'example.com/foo':
    - Wenn kein Schema erkennbar ist, mit '//' präfixen und dann netloc/path korrekt ermitteln.
    - Standard-Schema: https
    - Entfernt Tracking-Parameter gem. Whitelist
    """
    if not isinstance(u, str) or not u.strip():
        return None
    s = u.strip()
    if "#" in s:
        s = s.split("#", 1)[0]
    try:
        has_scheme = re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", s) is not None
        sp = urlsplit(s if has_scheme else f"//{s}", allow_fragments=True)
    except Exception:
        return None

    scheme = (sp.scheme or "https").lower()
    # Wenn wir ohne Schema geparst haben, landet der Host in sp.netloc, Pfad in sp.path
    netloc = (sp.netloc or "").lower()
    path = sp.path if sp.netloc else ""  # wenn netloc aus path kam, keinen zusätzlichen path setzen

    # Query filtern
    keep_pairs = []
    for k, v in parse_qsl(sp.query, keep_blank_values=True):
        kl = k.lower()
        if any(kl.startswith(pref) for pref in TRACKING_PARAMS_PREFIXES): 
            continue
        if kl in TRACKING_PARAMS_EXACT: 
            continue
        keep_pairs.append((k, v))
    query = urlencode(keep_pairs, doseq=True)

    # Port-Defaults entfernen
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Wenn netloc fehlt, kann der Host versehentlich in path stehen (extreme Edgecases)
    if not netloc and has_scheme:
        # Fallback: so wie es ist zurück
        return urlunsplit((scheme, sp.netloc, sp.path, query, "")) or None
    elif not netloc and not has_scheme:
        # z. B. Input "example.com" → hier ist sp.netloc gesetzt (durch //), also sollte dieser Zweig selten sein
        return None

    return urlunsplit((scheme, netloc, path, query, "")) or None
def read_table(uploaded) -> pd.DataFrame:
    """
    CSV/Excel robust:
    - CSV: Erst normaler Read (mehrspaltig? -> fertig). Wenn single-column: Encoder+Trenner Heuristik, dann Fallback manuelles Splitten.
    - Excel: normal lesen.
    - Erste Zeile = Header (bei manuellem Split).
    """
    data = uploaded.read()
    name = (uploaded.name or "").lower()

    def _as_bytesio():
        return io.BytesIO(data)

    # --- CSV ---
    if name.endswith(".csv"):
        # 1) Schnellpfad: Standard-Read (z. B. , als Trenner) – wenn bereits mehrspaltig, passt es
        try:
            df = pd.read_csv(_as_bytesio())
        except Exception:
            df = pd.DataFrame()
        if df.shape[1] > 1:
            return normalize_headers(df)

        # 2) Heuristiken: encodings × seps (python engine, ohne low_memory)
        for enc, sep in [
            ("utf-8-sig", ";"), ("utf-8-sig", ","), ("utf-8-sig", "\t"),
            ("utf-8", ";"), ("utf-8", ","), ("cp1252", ";"), ("latin-1", ";")
        ]:
            try:
                df_try = pd.read_csv(_as_bytesio(), encoding=enc, sep=sep, engine="python")
                if df_try.shape[1] > 1:
                    return normalize_headers(df_try)
            except Exception:
                continue

        # 3) Auto-Sniff mit python engine (ohne low_memory)
        try:
            df_auto = pd.read_csv(_as_bytesio(), sep=None, engine="python")
            if df_auto.shape[1] > 1:
                return normalize_headers(df_auto)
        except Exception:
            pass

        # 4) Manuelles Splitten, wenn immer noch eine Spalte
        if df.shape[1] == 1 or df.empty:
            if df.empty:
                # letzter Versuch: lesbar machen
                try:
                    df = pd.read_csv(_as_bytesio(), encoding="utf-8-sig", header=None)
                except Exception:
                    df = pd.DataFrame({ "col_1": [] })
            col = df.columns[0] if len(df.columns) else "col_1"
            s = df[col].astype(str)
            counts = {
                ";": s.str.count(";").sum(),
                ",": s.str.count(",").sum(),
                "\t": s.str.count("\t").sum(),
                "|": s.str.count("|").sum()
            }
            delim = max(counts, key=counts.get) if any(v > 0 for v in counts.values()) else ";"
            parts = s.str.split(delim, expand=True)

            # Erste Zeile als Header interpretieren, wenn sie „headerhaft“ aussieht (keine Zahlenkolonne)
            header_row = parts.iloc[0].astype(str).tolist() if parts.shape[0] > 0 else []
            looks_like_header = any(not re.fullmatch(r"\s*[\d\.,]+\s*", h) for h in header_row)
            if looks_like_header and parts.shape[0] > 1:
                parts = parts.iloc[1:].reset_index(drop=True)
                parts.columns = [normalize_header(h) for h in header_row]
            else:
                parts.columns = [f"col_{i+1}" for i in range(parts.shape[1])]

            return normalize_headers(parts)

        return normalize_headers(df)

    # --- Excel ---
    try:
        df = pd.read_excel(io.BytesIO(data))
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception:
            df = pd.DataFrame()
    return normalize_headers(df)

# ---- CSV-Compound-Fallback ("URL;RD;BL") — nur wenn keine URL-Spalte erkannt wird ----
def try_split_compound_url_metrics(df: pd.DataFrame) -> pd.DataFrame:
    url_aliases = set(ALIASES["url"])
    if any(c in df.columns for c in url_aliases):
        return df
    cand_cols = [c for c in df.columns if ("url" in c and ("ref" in c or "link" in c))]
    if len(df.columns) == 1:
        cand_cols = [df.columns[0]]
    if not cand_cols:
        return df
    col = cand_cols[0]
    s = df[col].astype(str)
    if not s.str.contains(";").any():
        return df
    parts = s.str.split(";", n=2, expand=True)
    df2 = df.copy()
    df2["url"] = parts[0].astype(str)
    if parts.shape[1] >= 2: df2["ref_domains"] = to_numeric_smart(parts[1])
    if parts.shape[1] >= 3: df2["backlinks"] = to_numeric_smart(parts[2])
    return df2

def store_upload(key: str, file):
    if file is None: return
    df = read_table(file)
    df = try_split_compound_url_metrics(df)
    st.session_state.uploads[key] = (df, file.name)

def find_first_alias(df: pd.DataFrame, target: str) -> Optional[str]:
    candidates = ALIASES.get(target, [])
    for c in candidates:
        if c in df.columns: return c
    cols_norm = { _alias_norm(c): c for c in df.columns }
    for c in candidates:
        cn = _alias_norm(c)
        if cn in cols_norm: return cols_norm[cn]
    if target == "url":
        for col in df.columns:
            cl = col.lower()
            if cl == "address" or cl.startswith("address_") or cl.startswith("page_url"):
                return col
    return None

def require_columns_ui(key: str, df: pd.DataFrame, targets: List[str], label: str) -> Dict[str, str]:
    colmap = {}
    for t in targets:
        hit = find_first_alias(df, t)
        if hit: colmap[t] = hit
    missing = [t for t in targets if t not in colmap]
    if missing:
        st.warning(f"**{label}**: Spalten nicht eindeutig erkannt. Bitte zuordnen.")
        with st.expander(f"Spalten-Mapping für {label}", expanded=True):
            for t in targets:
                options = [None] + list(df.columns)
                default_idx = options.index(colmap[t]) if t in colmap else 0
                sel = st.selectbox(f"{t} →", options, index=default_idx, key=f"map_{key}_{t}")
                if sel: colmap[t] = sel
    st.session_state.column_maps[key] = colmap
    return colmap

def ensure_url_column(df: pd.DataFrame, url_col: str) -> pd.DataFrame:
    df = df.copy()
    df[url_col] = df[url_col].map(normalize_url)
    return df[df[url_col].notna()]

# ---------- Schema-Index Cache ----------
def build_schema_index():
    idx = {}
    for key, (df, name) in st.session_state.uploads.items():
        role_map = {t: find_first_alias(df, t) for t in ALIASES.keys()}
        idx[key] = {"cols": list(df.columns), "roles": role_map, "name": name}
    st.session_state.schema_index = idx

def find_df_with_targets(targets: List[str], prefer_keys: Optional[List[str]] = None, use_autodiscovery: bool = True
                         ) -> Optional[Tuple[str, pd.DataFrame, Dict[str,str]]]:
    uploads = st.session_state.uploads
    idx = st.session_state.get("schema_index", {})

    def try_key(k: str) -> Optional[Tuple[str, pd.DataFrame, Dict[str,str]]]:
        if k not in uploads: return None
        df, _ = uploads[k]
        roles = idx.get(k, {}).get("roles")
        if roles:
            if all(roles.get(t) for t in targets):
                return k, df, {t: roles[t] for t in targets}
            return None
        mapping = {}
        for t in targets:
            col = find_first_alias(df, t)
            if not col: return None
            mapping[t] = col
        return k, df, mapping

    for k in (prefer_keys or []):
        hit = try_key(k)
        if hit: return hit

    if use_autodiscovery:
        for k in uploads.keys():
            if prefer_keys and k in prefer_keys: continue
            hit = try_key(k)
            if hit: return hit

    return None

# ============= Scoring =============
def rank_scores(series: pd.Series, min_score: float = 0.2) -> pd.Series:
    """
    Neu: Nur Werte > 0 werden gerankt.
    - Werte == 0 und fehlende Werte (NaN) bekommen IMMER 0.0
    - Unter den positiven Werten bekommt die schlechteste mind. 'min_score'
    """
    s = to_numeric_smart(series)
    s = s.clip(lower=0)  # negative -> 0
    mask_pos = s > 0     # NUR positive Werte werden gerankt

    out = pd.Series(0.0, index=s.index)
    if mask_pos.sum() <= 1:
        # genau ein positiver Wert -> 1.0; 0/NaN bleiben 0.0
        out[mask_pos] = 1.0
        return out

    ranks = s[mask_pos].rank(method="average", ascending=False)  # 1 = bester Wert
    n = len(ranks)
    out.loc[mask_pos] = 1.0 - (ranks - 1) / (n - 1) * (1.0 - min_score)
    return out

def bucket_scores(series: pd.Series,
                  quantiles: List[float] = [0.0, 0.5, 0.75, 0.9, 0.97, 1.0],
                  bucket_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]) -> pd.Series:
    s = to_numeric_smart(series).clip(lower=0)
    mask = s.notna()
    res = pd.Series(0.0, index=s.index)
    if mask.sum() == 0: return res
    try:
        qvals = s[mask].quantile(quantiles).values
        bins = np.unique(qvals)
        if len(bins) < 3:
            return rank_scores(series, min_score=0.2)
        # Falls Anzahl der Bins nicht zu den bucket_values passt, sicherheitshalber auf die passende Länge mappen
        labels_count = len(bins) - 1
        values = bucket_values[:labels_count] if len(bucket_values) >= labels_count else (
            bucket_values + [bucket_values[-1]] * (labels_count - len(bucket_values))
        )
        cats = pd.cut(s[mask], bins=bins, include_lowest=True, labels=False)
        bv = dict(zip(range(len(values)), values))
        res[mask] = cats.map(lambda i: bv.get(int(i), 0.0)).astype(float)
        return res
    except Exception:
        return rank_scores(series, min_score=0.2)

def score_series(series: pd.Series, mode: str, min_score: float,
                 quantiles: List[float], bucket_values: List[float]) -> pd.Series:
    return rank_scores(series, min_score) if mode == "Rank (linear)" else bucket_scores(series, quantiles, bucket_values)

def default_ctr_curve() -> pd.DataFrame:
    return pd.DataFrame({"position": list(range(1, 20+1)),
                         "ctr": [0.30,0.15,0.10,0.07,0.05,0.04,0.035,0.03,0.025,0.02,0.018,0.016,0.014,0.012,0.010,0.009,0.008,0.007,0.006,0.005]})

def get_ctr_for_pos(pos: float, ctr_df: pd.DataFrame) -> float:
    try: p = int(np.ceil(float(pos)))
    except Exception: return 0.0
    p = max(1, min(p, int(ctr_df["position"].max())))
    row = ctr_df.loc[ctr_df["position"] == p]
    return float(row["ctr"].values[0]) if not row.empty else 0.0

# ============= Sidebar settings =============
st.sidebar.header("⚙️ Einstellungen")
scoring_mode = st.sidebar.radio(
    "Scoring-Modus (global)",
    ["Rank (linear)", "Perzentil-Buckets"],
    index=0,
    help="Bestimmt, wie rohe Metriken zu Teil-Scores werden: Entweder linear nach Rang (1.0 → min_score) oder in Quantil-Buckets."
)
min_score = (
    st.sidebar.slider(
        "Min-Score schlechteste vorhandene URL",
        0.0, 0.5, 0.2, 0.05,
        help="Nur für Rank (linear): Unter den URLs mit > 0 im aktiven Kriterium erhält die schlechteste mindestens diesen Score. URLs ohne Wert oder mit 0 bekommen immer 0.0."
    )
    if scoring_mode == "Rank (linear)" else 0.2
)

with st.sidebar.expander("Bucket-Setup (Info)", expanded=False):
    st.markdown(
        "- **Quantile**: `[0, .5, .75, .9, .97, 1.0]`\n"
        "- **Scores**: `[0, .25, .5, .75, 1.0]`\n\n"
        "URLs ohne Wert im aktiven Kriterium bekommen **0.0**. "
        "Die Buckets ordnen Werte entsprechend ihrer Verteilung ein (robust bei Ausreißern)."
    )

use_autodiscovery = st.sidebar.toggle(
    "Autodiscovery über alle Uploads",
    value=True,
    help="Wenn aktiv: Fehlen die vorgesehenen Dateien/Spalten, sucht das Tool automatisch in anderen Uploads nach passenden Spalten (per Alias)."
)

# ============= Kriterienauswahl – Karten mit Toggle (schön) =============
st.subheader("Kriterien auswählen")
st.caption("Wähle unten die gewünschten Kriterien. Danach erscheinen die passenden Upload-Masken.")

# CSS für Karten-Grid
st.markdown("""
<style>
.module-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 12px;
    margin: 10px 0 18px;
}
.module-card {
    border: 1px solid #e5e7eb;
    background: #f9fafb;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    transition: all .2s ease;
}
.module-card:hover { background: #eef2ff; box-shadow: 0 6px 12px rgba(0,0,0,.08); }
.module-title { font-weight: 600; color: #111827; font-size: 1.02rem; margin: 0 0 4px 0; }
.module-desc { color: #4b5563; font-size: .92rem; line-height: 1.35; margin-bottom: 8px; }
.module-row { display:flex; align-items:center; justify-content:space-between; gap:10px; }
</style>
""", unsafe_allow_html=True)

# Wichtig: CRITERIA_GROUPS muss vorhanden sein (wie bisher definiert)
active: Dict[str, bool] = {}

for group, crits in CRITERIA_GROUPS.items():
    st.markdown(f"### {group}")
    # Start Grid
    st.markdown('<div class="module-grid">', unsafe_allow_html=True)

    # Für jedes Kriterium eine Karte mit Toggle
    for code, label, helptext in crits:
        # Toggle zuerst rendern (Widget), dann die Karte (HTML). Beides in einem Container halten:
        # Wir nutzen eine kleine leere Spalte, damit Toggle und Karte optisch zusammengehören.
        toggle_key = f"toggle_{code}"

        # Toggle-Widget (klein, ohne Label – das Label steht als Titel in der Karte)
        toggle_state = st.toggle(
            label, key=toggle_key, value=False, help=helptext, label_visibility="collapsed"
        )

        # Karte rendern (Titel + Beschreibung)
        st.markdown(f"""
            <div class="module-card">
                <div class="module-row">
                    <div class="module-title">{label}</div>
                    <div>{'✅' if toggle_state else '⬜️'}</div>
                </div>
                <div class="module-desc">{helptext}</div>
            </div>
        """, unsafe_allow_html=True)

        # Status im bekannten Dict speichern (abwärtskompatibel zum restlichen Code)
        active[code] = toggle_state

    # Ende Grid
    st.markdown('</div>', unsafe_allow_html=True)


# ============= Upload-Masken (nach Auswahl) =============
st.markdown("---")
st.subheader("Basierend auf den gewählten Kriterien benötigen wir folgende Dateien")

if active.get("sc_clicks") or active.get("sc_impr"):
    st.markdown("**Search Console — erwartet:** `URL` (Alias: url/page/page_url/address), `Clicks/Klicks`, `Impressions/Impressionen`. **Query-Ebene ist ok** – wird pro URL aggregiert.")
    store_upload("sc", st.file_uploader("Search Console Datei (CSV/XLSX)", type=["csv","xlsx"], key="upl_sc"))
# Search Console Performance-Klassifizierung Setup (nur Schwellenwerte)
if active.get("sc_perf_class"):
    st.markdown("### 🧩 Search Console Performance-Klassifizierung – Schwellen")
    st.markdown("""
    Definiere die **Klick-Schwellen** für die Kategorien sowie den **Impressionen-Schwellenwert**
    für *Opportunity* (0 Klicks, Impressions ≥ X) vs. *Dead* (0 Klicks, Impressions < X).
    Die **Kategorie-Scores sind fix** und werden nicht angepasst.
    """)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.number_input("Performance ≥ Klicks", min_value=0, value=1000, step=10, key="th_perf")
    with c2:
        st.number_input("Good ≥ Klicks", min_value=0, value=101, step=5, key="th_good")
    with c3:
        st.number_input("Fair ≥ Klicks", min_value=0, value=11, step=1, key="th_fair")
    with c4:
        st.number_input("Weak ≥ Klicks", min_value=0, value=1, step=1, key="th_weak")
    with c5:
        st.number_input("Opportunity: 0 Klicks & Impr ≥", min_value=0, value=100, step=10, key="th_opp_impr")

    st.caption("""
    Fixe Kategorie-Scores: Performance=1.00 · Good=0.75 · Fair=0.50 · Weak=0.25 · Opportunity=0.15 · Dead=0.00
    """)
    # --- Optionaler Upload für SC Performance-Datei ---
    st.markdown("**Search Console Performance-Datei — erwartet:** `keyword/query/suchanfrage`, `URL`, `Clicks/Klicks`, `Impressions/Impressionen` (Query-Ebene möglich; wird pro URL aggregiert).")
    store_upload("sc_perf", st.file_uploader("SC Performance Datei (CSV/XLSX) – optional, sonst wird die normale SC-Datei verwendet", type=["csv","xlsx"], key="upl_sc_perf"))
    st.caption("Wenn hier **keine Datei** hochgeladen wird, verwendet das Tool automatisch die oben hochgeladene **Search Console Datei**.")

if active.get("overall_traffic"):
    st.markdown("**Overall Traffic — erwartet:** `URL` + eine Traffic-Spalte (Aliases erkannt: sessions/visits/overall_clicks/…)")
    store_upload("overall", st.file_uploader("Overall Traffic Datei (CSV/XLSX)", type=["csv","xlsx"], key="upl_overall"))

# AI Overviews Popularität
if active.get("ai_overview"):
    st.markdown("**AI Overviews Popularität — erwartet:** `keyword`, `url` (aktuell rankende URL), `current_url_inside` (ob die aktuelle URL in der AI Overview als Quelle auftaucht).")
    st.caption("Hinweis: `current_url_inside` kann 1/0, true/false, ja/nein sein ODER die URL enthalten, wenn sie enthalten ist. Wir zählen je Zeile 1, wenn die URL tatsächlich in der AI Overview vorkommt.")
    store_upload("aiov", st.file_uploader("AI Overviews Datei (CSV/XLSX)", type=["csv","xlsx"], key="upl_aiov"))

if active.get("otv"):
    st.markdown("**Organic Traffic Value — erwartet:**")
    st.markdown("- **Variante A (URL-Value):** `URL`, `traffic_value` **oder** `potential_traffic_url` (+ optional `cpc`).")
    st.markdown("- **Variante B (Keyword-basiert):** `keyword`, `URL`, `position`, `search_volume` (+ optional `cpc`).")
    c1, c2 = st.columns(2)
    with c1:
        store_upload("otv_url", st.file_uploader("OTV: URL-Value (optional)", type=["csv","xlsx"], key="upl_otv_url"))
    with c2:
        store_upload("otv_kw",  st.file_uploader("OTV: Keyword-Datei (optional)", type=["csv","xlsx"], key="upl_otv_kw"))

# CTR-Kurve separat anzeigen, sobald sie irgendwo gebraucht werden könnte
if active.get("otv") or active.get("main_kw_exp"):
    store_upload("ctr_curve", st.file_uploader(
        "CTR-Kurve (optional, für Expected Clicks & OTV-Keyword)",
        type=["csv","xlsx"], key="upl_ctr"
    ))
    st.caption("Format: Spalten **position** (1..n) und **ctr** (0..1). Fehlende Datei ⇒ Standard-CTR.")

if active.get("ext_pop"):
    st.markdown("**URL-Popularität extern — erwartet:** `URL`, `backlinks`, `ref_domains`.")
    store_upload("ext", st.file_uploader("Extern-Popularität (CSV/XLSX)", type=["csv","xlsx"], key="upl_ext"))

if active.get("int_pop"):
    st.markdown("**URL-Popularität intern — erwartet:**")
    st.markdown("- **Variante A:** `URL`, `unique_inlinks`.")
    st.markdown("- **Variante B (Kantenliste):** `URL` (= Ziel) **und** eine Quellspalte (z. B. `source`, `source_url`, `referrer`).")
    store_upload("int", st.file_uploader("Intern-Popularität (Crawl/Inlinks)", type=["csv","xlsx"], key="upl_int"))

if active.get("llm_ref"):
    st.markdown("**LLM Referrals — erwartet:** **genau zwei Spalten**: `URL`, `Sitzungen / LLM-Traffic`.")
    store_upload("llmref", st.file_uploader("LLM-Referrals (CSV/XLSX)", type=["csv","xlsx"], key="upl_llmref"))

if active.get("llm_crawl"):
    st.markdown("**LLM Crawler Frequenz — erwartet:**")
    st.markdown("- **Variante A (aggregiert):** `URL` + mehrere Bot-Spalten (`GPTBot`, `ClaudeBot`, `PerplexityBot`, `OAI-SearchBot`, …).")
    st.markdown("- **Variante B (Logfile):** `URL`, `user_agent` (+ optional `sessions/visits/hits/requests`). Klassische Bots werden exkludiert.")
    store_upload("llmcrawl", st.file_uploader("LLM-Crawl (CSV/XLSX)", type=["csv", "xlsx"], key="upl_llmcrawl"))

    # (3) NEU: UI für Aggregated/Logfile-Modus & Spaltenauswahl
    with st.expander("LLM-Crawl – Modus & Spaltenauswahl", expanded=False):
        mode = st.radio("Modus", ["Logfile", "Aggregiert (pro Bot-Spalte)"], index=0, key="llm_crawl_mode_radio")
        st.session_state.llm_crawl_mode = "aggregated" if mode.startswith("Aggregiert") else "log"
        if "llmcrawl" in st.session_state.uploads:
            df_aggr, _ = st.session_state.uploads["llmcrawl"]
            urlc = find_first_alias(df_aggr, "url")
            if urlc:
                bot_cols = [c for c in df_aggr.columns if c != urlc]
                st.session_state.llm_bot_column_choices = st.multiselect(
                    "Bot-Spalten wählen (Aggregiert):",
                    bot_cols,
                    default=bot_cols[:3],
                    key="llm_bot_column_choices_ui"
                )

if active.get("llm_citations"):
    st.markdown("**LLM Citations — akzeptierte Formate:**")
    st.markdown("- **A (aggregiert):** `URL`, `llm_citations` (oder Alias wie `citations`).")
    st.markdown("- **B (pro Prompt/Keyword):** `keyword/prompt`, `URL` **und** je LLM eine Spalte (z. B. `gpt4`, `claude`, `perplexity`) mit 0/1, Anzahl oder der verlinkten URL.")
    st.markdown("- **C (pro Prompt/Keyword, eine Spalte):** `keyword/prompt`, `URL`, `cited_url` (wenn `cited_url == URL` ⇒ 1 Citation), optional `llm`.")
    store_upload("llmcite", st.file_uploader("LLM-Citations Datei (CSV/XLSX)", type=["csv","xlsx"], key="upl_llmcite"))

# Embeddings
if active.get("offtopic"):
    st.markdown("**Embeddings — erwartet:** `URL`, `embedding` (JSON-Liste oder Zahlen-Sequenz). Fehlende Embeddings ⇒ Outlier (< τ).")
    store_upload("emb", st.file_uploader("Embeddings-Datei (CSV/XLSX)", type=["csv","xlsx"], key="upl_emb"))

# Revenue
if active.get("revenue"):
    st.markdown("**Umsatz — erwartet:** `URL`, `revenue`.")
    store_upload("rev", st.file_uploader("Umsatz-Datei (CSV/XLSX)", type=["csv","xlsx"], key="upl_rev"))

# SEO Efficiency
if active.get("seo_eff"):
    st.markdown("""
    **URL-SEO-Effizienz — erwartet:**  
    `keyword/query/suchanfrage`, `URL`, `position`  
    (Top-5-Anteil je URL wird berechnet – Anteil der Keywords mit Position ≤ 5).  
    """)

    c1, c2 = st.columns(2)
    with c1:
        store_upload(
            "eff_kw",
            st.file_uploader(
                "Keyword-Datei (SEO-Effizienz, z. B. Sistrix/Ahrefs)",
                type=["csv","xlsx"],
                key="upl_eff_kw"
            )
        )
    with c2:
        store_upload(
            "sc_eff",
            st.file_uploader(
                "Search Console Datei (optional, falls noch nicht oben geladen)",
                type=["csv","xlsx"],
                key="upl_sc_eff"
            )
        )

    st.caption("""
    Wenn keine eigene Keyword-Datei vorhanden ist, verwendet das Tool automatisch die **Search Console Daten**
    (egal ob oben bei *Search Console*, bei *SC Performance-Klassifizierung* oder hier hochgeladen).
    """)

# Priority
if active.get("priority"):
    st.markdown("**Strategische Priorität — erwartet (optional):** `URL`, `priority_factor` (0.5–2.0).")
    store_upload("prio", st.file_uploader("Priorität (optional)", type=["csv","xlsx"], key="upl_prio"))

# Hauptkeyword
if active.get("main_kw_exp") or active.get("main_kw_sv"):
    st.markdown("**Hauptkeyword-Potenzial — erwartet:** `URL`, `main_keyword` + je nach Kriterium weitere Spalten.")
    store_upload("main_kw", st.file_uploader("Hauptkeyword-Mapping (CSV/XLSX)", type=["csv","xlsx"], key="upl_main_kw"))

# ---------- (Re)build schema index ----------
build_schema_index()

# ============= Master URL list builder =============
st.subheader("Master-URL-Liste")
st.markdown("""
Die **Master-URL-Liste** ist die zentrale Ausgangsliste aller URLs, die bewertet werden. 
Alle Kriterien werden an diese Liste gejoined. Nur dort enthaltene URLs bekommen am Ende einen Score.
""")

master_mode = st.radio(
    "Masterlisten-Modus",
    [
        "Union (alle Uploads) [Default]",
        "Eigene Masterliste hochladen",
        "Merge aus bis zu zwei Dateien",
        "Aus einer bestimmten Datei wählen",
        "Schnittmenge (alle Uploads)"
    ],
    index=0,
)

master_urls: Optional[pd.DataFrame] = None

def collect_urls_from_key(key: str) -> Optional[pd.DataFrame]:
    if key not in st.session_state.uploads:
        return None
    df, _ = st.session_state.uploads[key]
    c = find_first_alias(df, "url")
    if not c: return None
    d = ensure_url_column(df[[c]], c).rename(columns={c: "url_norm"})
    return d.drop_duplicates()

def collect_urls_union(include_keys: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    urls = []
    for key, (df, _) in st.session_state.uploads.items():
        if include_keys and key not in include_keys:
            continue
        c = find_first_alias(df, "url")
        if c:
            d = ensure_url_column(df[[c]], c).rename(columns={c: "url_norm"})
            urls.append(d)
    if urls:
        return pd.concat(urls, axis=0, ignore_index=True).drop_duplicates()
    return None

def collect_urls_intersection_all_uploads() -> Optional[pd.DataFrame]:
    """
    (4) Präziser & explizit normalisiert:
    Wir normalisieren jede URL-Spalte mit ensure_url_column und bilden dann die Schnittmenge.
    """
    url_sets = []
    for key, (df, _) in st.session_state.uploads.items():
        c = find_first_alias(df, "url")
        if not c:
            continue
        d = ensure_url_column(df[[c]].copy(), c)  # c ist jetzt normalisiert
        s = set(d[c].dropna().unique().tolist())
        if s:
            url_sets.append(s)
    if not url_sets:
        return None
    inter = set.intersection(*url_sets)
    return pd.DataFrame({"url_norm": sorted(inter)}) if inter else pd.DataFrame({"url_norm": []})

if master_mode == "Union (alle Uploads) [Default]":
    master_urls = collect_urls_union()
elif master_mode == "Eigene Masterliste hochladen":
    mf = st.file_uploader("Eigene Masterliste (Spalte: url/page/page_url/address/...)", type=["csv","xlsx"], key="upl_master")
    if mf:
        dfm = read_table(mf)
        dfm = try_split_compound_url_metrics(dfm)
        url_col = find_first_alias(dfm, "url") or st.selectbox("URL-Spalte in Masterliste wählen", dfm.columns, key="map_master_url")
        dfm = ensure_url_column(dfm, url_col)
        master_urls = dfm[[url_col]].rename(columns={url_col: "url_norm"}).drop_duplicates()
elif master_mode == "Merge aus bis zu zwei Dateien":
    available = list(st.session_state.uploads.keys())
    pick1 = st.selectbox("Quelle 1", options=[None] + available, index=0, key="master_src1")
    pick2 = st.selectbox("Quelle 2 (optional)", options=[None] + available, index=0, key="master_src2")
    if pick1:
        keys = [pick1] + ([pick2] if pick2 else [])
        master_urls = collect_urls_union(include_keys=keys)
elif master_mode == "Aus einer bestimmten Datei wählen":
    available = list(st.session_state.uploads.keys())
    pick = st.selectbox("Datei auswählen", options=[None] + available, index=0, key="master_src_single")
    if pick:
        master_urls = collect_urls_from_key(pick)
elif master_mode == "Schnittmenge (alle Uploads)":
    master_urls = collect_urls_intersection_all_uploads()

if master_urls is None or master_urls.empty:
    if master_mode == "Schnittmenge (alle Uploads)":
        st.warning("Schnittmenge leer – keine URL kommt in **allen** Uploads vor. Prüfe Uploads & URL-Spalten.")
    else:
        st.info("Noch keine Master-URLs erkannt. Lade mindestens eine Datei mit URL-Spalte hoch **oder** wähle eine der Masterlisten-Optionen.")
else:
    st.success(f"Master-URLs: {len(master_urls):,}")
# ============= Compute per-criterion scores =============
def mode_score(series: pd.Series) -> pd.Series:
    return score_series(series, scoring_mode, min_score, [0.0, 0.5, 0.75, 0.9, 0.97, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0])

def join_on_master(df: pd.DataFrame, url_col: str, val_cols: List[str]) -> pd.DataFrame:
    d = ensure_url_column(df[[url_col] + val_cols], url_col).rename(columns={url_col: "url_norm"})
    return master_urls.merge(d, on="url_norm", how="left") if master_urls is not None else d

results: Dict[str, pd.Series] = {}
debug_cols: Dict[str, Dict[str, pd.Series]] = {}

# Search Console
if active.get("sc_clicks") or active.get("sc_impr"):
    need_cols = ["url"] + (["clicks"] if active.get("sc_clicks") else []) + (["impressions"] if active.get("sc_impr") else [])
    found = find_df_with_targets(need_cols, prefer_keys=["sc","sc_perf"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_sc, colmap = found
        urlc = colmap["url"]
        df_sc = ensure_url_column(df_sc, urlc).copy()
        metrics = []
        if active.get("sc_clicks"): metrics.append(colmap["clicks"])
        if active.get("sc_impr"):  metrics.append(colmap["impressions"])
        for m in metrics: df_sc[m] = to_numeric_smart(df_sc[m]).fillna(0)
        agg_sc = df_sc.groupby(urlc, as_index=False)[metrics].sum()
        scj = join_on_master(agg_sc, urlc, metrics)
        if active.get("sc_clicks"):
            results["sc_clicks"] = mode_score(scj[colmap["clicks"]]).fillna(0.0)
            debug_cols.setdefault("sc", {})["clicks_raw"] = scj[colmap["clicks"]]
        if active.get("sc_impr"):
            results["sc_impr"] = mode_score(scj[colmap["impressions"]]).fillna(0.0)
            debug_cols.setdefault("sc", {})["impressions_raw"] = scj[colmap["impressions"]]
    elif master_urls is not None:
        if active.get("sc_clicks"): results["sc_clicks"] = pd.Series(0.0, index=master_urls.index)
        if active.get("sc_impr"):  results["sc_impr"]  = pd.Series(0.0, index=master_urls.index)

# Overall Traffic
if active.get("overall_traffic"):
    found = find_df_with_targets(["url","overall_traffic"], prefer_keys=["overall"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_o, cm = found
        d = join_on_master(df_o, cm["url"], [cm["overall_traffic"]])
        vals = to_numeric_smart(d[cm["overall_traffic"]]).clip(lower=0)
        results["overall_traffic"] = mode_score(vals).fillna(0.0)
        debug_cols["overall_traffic"] = {"overall_traffic_raw": vals}
    elif master_urls is not None:
        results["overall_traffic"] = pd.Series(0.0, index=master_urls.index)

# SC Performance-Klassifizierung (diskrete Kategorien → feste Scores)
if active.get("sc_perf_class"):
    need_cols = ["url","clicks","impressions"]
    found = find_df_with_targets(need_cols, prefer_keys=["sc"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_sc2, cm2 = found
        urlc = cm2["url"]; cc = cm2["clicks"]; ic = cm2["impressions"]
        df_sc2 = ensure_url_column(df_sc2, urlc).copy()
        df_sc2[cc] = to_numeric_smart(df_sc2[cc]).fillna(0)
        df_sc2[ic] = to_numeric_smart(df_sc2[ic]).fillna(0)
        agg = df_sc2.groupby(urlc, as_index=False)[[cc, ic]].sum()

        d = master_urls.merge(agg, left_on="url_norm", right_on=urlc, how="left")
        clicks = to_numeric_smart(d[cc]).fillna(0)
        impr   = to_numeric_smart(d[ic]).fillna(0)

        perf_min = st.session_state.get("th_perf", 1000)
        good_min = st.session_state.get("th_good", 101)
        fair_min = st.session_state.get("th_fair", 11)
        weak_min = st.session_state.get("th_weak", 1)
        opp_impr_min = st.session_state.get("th_opp_impr", 100)

        score_map = {
            "Performance": 1.00,
            "Good":        0.75,
            "Fair":        0.50,
            "Weak":        0.25,
            "Opportunity": 0.15,
            "Dead":        0.00,
        }

        # Klassifizierungslogik
        cats = []
        for c, i in zip(clicks.values, impr.values):
            if c == 0:
                cats.append("Opportunity" if i >= opp_impr_min else "Dead")
            elif c >= perf_min:
                cats.append("Performance")
            elif c >= good_min:
                cats.append("Good")
            elif c >= fair_min:
                cats.append("Fair")
            elif c >= weak_min:
                cats.append("Weak")
            else:
                cats.append("Dead")
        cat_series = pd.Series(cats, index=d.index)
        score_series_disc = cat_series.map(score_map).astype(float).fillna(0.0)

        results["sc_perf_class"] = score_series_disc

        with st.expander("Vorschau: SC Performance-Klassifizierung (Top 10)", expanded=False):
            prev = pd.DataFrame({
                "url": d["url_norm"],
                "sc_clicks": clicks,
                "sc_impressions": impr,
                "sc_category": cat_series,
                "sc_category_score": score_series_disc,
            })
            st.dataframe(prev.head(10), use_container_width=True)

        debug_cols["sc_perf_class"] = {
            "category": cat_series,
            "category_score": score_series_disc,
            "clicks_raw": clicks,
            "impressions_raw": impr,
        }
    elif master_urls is not None:
        results["sc_perf_class"] = pd.Series(0.0, index=master_urls.index)

# AI Overviews Popularität
if active.get("ai_overview"):
    found = find_df_with_targets(["url"], prefer_keys=["aiov"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_aio, cm = found
        urlc = cm["url"]
        df_aio = ensure_url_column(df_aio, urlc).copy()

        inside_candidates = [
            "current_url_inside", "inside_ai_overview", "ai_overview_inside", "in_ai_overview",
            "in_aio", "aio_inside", "ai_overview_flag", "inside"
        ]
        inside_col = next((c for c in inside_candidates if c in df_aio.columns), None)

        if inside_col is None:
            for c in df_aio.columns:
                cl = c.lower()
                if ("current" in cl or "inside" in cl or "ai" in cl) and c != urlc:
                    inside_col = c
                    break

        # (2) Aufgeräumte, eindeutige Heuristik ohne doppelten, unerreichbaren Block
        def _to_indicator(row) -> int:
            if inside_col is None:
                return 0
            val = row.get(inside_col, None) if hasattr(row, "get") else (row[inside_col] if inside_col in row.index else None)
            s = (str(val) if val is not None else "").strip().lower()
            if not s:
                return 0

            # 1) Zahlen/Booleans
            if isinstance(val, (int, float)):
                return 1 if float(val) > 0 else 0
            if s in {"1","true","wahr","yes","ja","y"}: return 1
            if s in {"0","false","falsch","no","nein","n"}: return 0

            # 2) Term-Heuristik
            ai_terms = {"ai overview","ai overviews","ai-overview","ai_overview","aio","aiov"}
            if any(term in s for term in ai_terms):
                if all(neg not in s for neg in {"not in","kein","nicht in","absent"}):
                    return 1

            # 3) Feld enthält evtl. die URL selbst → normalize & vergleichen
            cur = row.get(urlc, None) if hasattr(row, "get") else (row[urlc] if urlc in row.index else None)
            u1 = normalize_url(s) if s else None
            u2 = normalize_url(cur if cur else "")
            if u1 and u2 and u1 == u2:
                return 1

            return 0

        if inside_col is None:
            agg = df_aio[[urlc]].copy()
            agg["_aiov_cnt"] = 0
            agg = agg.groupby(urlc, as_index=False)["_aiov_cnt"].sum()
        else:
            df_aio["_aiov_ind"] = df_aio.apply(_to_indicator, axis=1)
            agg = df_aio.groupby(urlc, as_index=False)["_aiov_ind"].sum().rename(columns={"_aiov_ind":"_aiov_cnt"})

        d = master_urls.merge(agg, left_on="url_norm", right_on=urlc, how="left")
        cnts = to_numeric_smart(d["_aiov_cnt"]).fillna(0)
        results["ai_overview"] = mode_score(cnts).fillna(0.0)
        debug_cols["ai_overview"] = {"ai_overview_count": cnts}
    elif master_urls is not None:
        results["ai_overview"] = pd.Series(0.0, index=master_urls.index)

# OTV
if active.get("otv"):
    raw_val = None
    found_url = find_df_with_targets(["url"], prefer_keys=["otv_url"], use_autodiscovery=use_autodiscovery)
    if found_url:
        _, df_u, cmap_u = found_url
        urlc = cmap_u["url"]
        val_col = find_first_alias(df_u, "traffic_value")
        pot_col = find_first_alias(df_u, "potential_traffic_url")
        cpc_col = find_first_alias(df_u, "cpc")
        keep_cols = [c for c in [val_col, pot_col, cpc_col] if c]
        if keep_cols:
            d = join_on_master(df_u, urlc, keep_cols)
            if val_col is not None:
                raw_val = to_numeric_smart(d[val_col])
            elif pot_col is not None and cpc_col is not None:
                raw_val = to_numeric_smart(d[pot_col]) * to_numeric_smart(d[cpc_col])
            elif pot_col is not None:
                raw_val = to_numeric_smart(d[pot_col])
    if raw_val is None:
        found_kw = find_df_with_targets(["keyword","url","position","search_volume"], prefer_keys=["otv_kw"], use_autodiscovery=use_autodiscovery)
        if found_kw and master_urls is not None:
            _, df_k, cmap_k = found_kw
            urlc, posc, svc = cmap_k["url"], cmap_k["position"], cmap_k["search_volume"]
            cpcc = find_first_alias(df_k, "cpc")
            found_ctr = find_df_with_targets(["position","ctr"], prefer_keys=["ctr_curve"], use_autodiscovery=use_autodiscovery)
            if found_ctr:
                _, ctr_df, ctr_map = found_ctr
                ctr_df = ctr_df[[ctr_map["position"], ctr_map["ctr"]]].rename(columns={ctr_map["position"]:"position", ctr_map["ctr"]:"ctr"})
            else:
                ctr_df = default_ctr_curve()
            df_k = ensure_url_column(df_k, urlc)
            ctrs = df_k[posc].map(lambda p: get_ctr_for_pos(p, ctr_df))
            pot_traffic = to_numeric_smart(df_k[svc]).fillna(0) * ctrs
            raw_row_val = pot_traffic * to_numeric_smart(df_k[cpcc]).fillna(0) if cpcc else pot_traffic
            agg = df_k.assign(_val=raw_row_val).groupby(urlc, as_index=False)["_val"].sum()
            d = master_urls.merge(agg, left_on="url_norm", right_on=urlc, how="left")
            raw_val = d["_val"]
    if master_urls is not None:
        results["otv"] = pd.Series(0.0, index=master_urls.index) if raw_val is None else mode_score(raw_val).fillna(0.0)
        if raw_val is not None: debug_cols["otv"] = {"otv_raw": raw_val}

# External popularity
if active.get("ext_pop"):
    found = find_df_with_targets(["url","backlinks","ref_domains"], prefer_keys=["ext"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_e, cm = found
        d = join_on_master(df_e, cm["url"], [cm["backlinks"], cm["ref_domains"]])
        ext = 0.3 * mode_score(d[cm["backlinks"]]) + 0.7 * mode_score(d[cm["ref_domains"]])
        results["ext_pop"] = ext.fillna(0.0)
        debug_cols["ext_pop"] = {"backlinks_raw": d[cm["backlinks"]], "ref_domains_raw": d[cm["ref_domains"]]}
    elif master_urls is not None:
        results["ext_pop"] = pd.Series(0.0, index=master_urls.index)

# Internal popularity
if active.get("int_pop"):
    found = find_df_with_targets(["url","unique_inlinks"], prefer_keys=["int"], use_autodiscovery=use_autodiscovery)
    if not found:
        found_edges = find_df_with_targets(["url"], prefer_keys=["int"], use_autodiscovery=use_autodiscovery)
        if found_edges:
            _, df_edges, cm_edges = found_edges
            urlc = cm_edges["url"]
            src_candidates = ["source","from","src","source_url","referrer","referrer_url","origin","origin_url","quelle","von","inlink_source","inlink_from"]
            src = next((c for c in src_candidates if c in df_edges.columns and c != urlc), None)
            if src:
                df_tmp = ensure_url_column(df_edges[[urlc, src]].copy(), urlc)
                df_tmp[src] = df_tmp[src].map(normalize_url)
                agg = (
                    df_tmp
                    .dropna(subset=[src])
                    .groupby(urlc, as_index=False)[src]
                    .nunique()
                    .rename(columns={src: "unique_inlinks"})
                )
                found = ("int_edges", agg, {"url": urlc, "unique_inlinks": "unique_inlinks"})
    if found and master_urls is not None:
        _, df_i, cm = found
        d = join_on_master(df_i, cm["url"], [cm["unique_inlinks"]])
        vals = to_numeric_smart(d[cm["unique_inlinks"]]).clip(lower=0)
        results["int_pop"] = mode_score(vals).fillna(0.0)
        debug_cols["int_pop"] = {"unique_inlinks_raw": vals}
    elif master_urls is not None:
        results["int_pop"] = pd.Series(0.0, index=master_urls.index)

# LLM Referral
if active.get("llm_ref"):
    found = find_df_with_targets(["url","llm_ref_traffic"], prefer_keys=["llmref"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_l, cm = found
        d = join_on_master(df_l, cm["url"], [cm["llm_ref_traffic"]])
        results["llm_ref"] = mode_score(d[cm["llm_ref_traffic"]]).fillna(0.0)
        debug_cols["llm_ref"] = {"llm_ref_traffic_raw": d[cm["llm_ref_traffic"]]}
    elif master_urls is not None:
        results["llm_ref"] = pd.Series(0.0, index=master_urls.index)

# LLM Crawl
def _contains_any(s: str, needles: List[str]) -> bool:
    s = s or ""
    l = s.lower()
    return any(n.lower() in l for n in needles if n)

if active.get("llm_crawl"):
    found = None

    if "llmcrawl" in st.session_state.uploads and st.session_state.get("llm_crawl_mode") == "aggregated":
        df_aggr, _ = st.session_state.uploads["llmcrawl"]
        urlc = find_first_alias(df_aggr, "url")
        sel_cols = st.session_state.get("llm_bot_column_choices", [])
        if urlc and sel_cols and master_urls is not None:
            df_aggr = ensure_url_column(df_aggr, urlc).copy()
            for c in sel_cols:
                df_aggr[c] = to_numeric_smart(df_aggr[c]).fillna(0)
            df_aggr["_llm_crawl_freq"] = df_aggr[sel_cols].sum(axis=1)
            agg = df_aggr.groupby(urlc, as_index=False)["_llm_crawl_freq"].sum().rename(columns={"_llm_crawl_freq": "llm_crawl_freq"})
            found = ("llmcrawl_aggr", agg, {"url": urlc, "llm_crawl_freq": "llm_crawl_freq"})
        else:
            found = None

    if found is None:
        found = find_df_with_targets(["url","llm_crawl_freq"], prefer_keys=["llmcrawl"], use_autodiscovery=use_autodiscovery)
        if not found:
            found_log = find_df_with_targets(["url","user_agent"], prefer_keys=["llmcrawl"], use_autodiscovery=use_autodiscovery)
            if found_log:
                _, df_log, cm = found_log
                urlc, uac = cm["url"], cm["user_agent"]
                df_log = ensure_url_column(df_log, urlc).copy()

                sess_col = find_first_alias(df_log, "llm_crawl_freq")
                has_amount = sess_col is not None and sess_col in df_log.columns

                include_from_ui = st.session_state.llm_bot_include or []
                exclude_from_ui = st.session_state.llm_bot_exclude or []
                custom_inc = [s.strip() for s in (st.session_state.llm_bot_custom_include or "").split(",") if s.strip()]
                custom_exc = [s.strip() for s in (st.session_state.llm_bot_custom_exclude or "").split(",") if s.strip()]

                include_needles = include_from_ui + custom_inc or INCLUDE_AI_BOTS[:]
                exclude_needles = list(set((exclude_from_ui + custom_exc) + EXCLUDE_CLASSIC_BOTS))

                mask_inc = df_log[uac].astype(str).map(lambda s: _contains_any(s, include_needles))
                mask_exc = df_log[uac].astype(str).map(lambda s: _contains_any(s, exclude_needles))
                mask = mask_inc & (~mask_exc)

                df_sel = df_log.loc[mask, [urlc] + ([sess_col] if has_amount else [])].copy()
                if not df_sel.empty:
                    if has_amount:
                        df_sel[sess_col] = to_numeric_smart(df_sel[sess_col]).fillna(0)
                        agg = df_sel.groupby(urlc, as_index=False)[sess_col].sum().rename(columns={sess_col: "llm_crawl_freq"})
                    else:
                        agg = df_sel.groupby(urlc, as_index=False).size().rename(columns={"size": "llm_crawl_freq"})
                    found = ("llmcrawl_log", agg, {"url": urlc, "llm_crawl_freq": "llm_crawl_freq"})
                else:
                    found = None

    if found and master_urls is not None:
        _, df_lc, cm = found
        d = join_on_master(df_lc, cm["url"], [cm["llm_crawl_freq"]])
        results["llm_crawl"] = mode_score(d[cm["llm_crawl_freq"]]).fillna(0.0)
        debug_cols["llm_crawl"] = {"llm_crawl_freq_raw": to_numeric_smart(d[cm["llm_crawl_freq"]])}
    elif master_urls is not None:
        results["llm_crawl"] = pd.Series(0.0, index=master_urls.index)

# LLM Citations
if active.get("llm_citations"):
    citations_series = None

    found_aggr = find_df_with_targets(["url","llm_citations"], prefer_keys=["llmcite"], use_autodiscovery=use_autodiscovery)
    if found_aggr and master_urls is not None:
        _, df_c, cm = found_aggr
        d = join_on_master(df_c, cm["url"], [cm["llm_citations"]])
        citations_series = to_numeric_smart(d[cm["llm_citations"]]).fillna(0)

    if citations_series is None:
        found_any = find_df_with_targets(["url"], prefer_keys=["llmcite"], use_autodiscovery=use_autodiscovery)
        if found_any and master_urls is not None:
            _, df_c, cm = found_any
            urlc = cm["url"]
            df_c = ensure_url_column(df_c, urlc).copy()

            cols = set(df_c.columns)
            kw_col = next((c for c in ["keyword","query","prompt","suchanfrage"] if c in cols), None)
            cited_url_col = next((c for c in ["cited_url","referenced_url","linked_url","quelle_url","source_url"] if c in cols), None)
            llm_col = next((c for c in ["llm","model","provider"] if c in cols), None)

            known_llm_tokens = ["gpt","openai","oai","chatgpt","gpt4","gpt-4","gpt-4o","claude","anthropic",
                                "perplexity","perplexitybot","sonar","llama","meta","gemini","copilot","bing","kagi"]
            def looks_like_llm_col(c: str) -> bool:
                cl = c.lower()
                if c == urlc or c == kw_col or c == cited_url_col or c == llm_col:
                    return False
                return any(tok in cl for tok in known_llm_tokens)

            llm_cols = [c for c in df_c.columns if looks_like_llm_col(c)]

            agg = None
            if cited_url_col:
                tmp = df_c[[urlc, cited_url_col]].copy()
                tmp[cited_url_col] = tmp[cited_url_col].map(normalize_url)
                tmp["_hit"] = (tmp[cited_url_col].notna() & (tmp[cited_url_col] == tmp[urlc])).astype(int)
                agg = tmp.groupby(urlc, as_index=False)["_hit"].sum().rename(columns={"_hit":"_cit"})
            elif llm_cols:
                tmp = df_c[[urlc] + llm_cols].copy()

                def col_to_numeric_hits(s: pd.Series) -> pd.Series:
                    num = to_numeric_smart(s)
                    if num.notna().any():
                        return num.fillna(0)
                    return s.astype(str).str.strip().replace({"": np.nan}).notna().astype(int)

                for c in llm_cols:
                    tmp[c] = col_to_numeric_hits(tmp[c])

                tmp["_row_cit"] = tmp[llm_cols].sum(axis=1)
                agg = tmp.groupby(urlc, as_index=False)["_row_cit"].sum().rename(columns={"_row_cit":"_cit"})

            if agg is not None:
                dj = master_urls.merge(agg, left_on="url_norm", right_on=urlc, how="left")
                citations_series = to_numeric_smart(dj["_cit"]).fillna(0)

    if master_urls is not None:
        if citations_series is None:
            results["llm_citations"] = pd.Series(0.0, index=master_urls.index)
        else:
            results["llm_citations"] = mode_score(citations_series).fillna(0.0)
            debug_cols["llm_citations"] = {"llm_citations_raw": citations_series}

# Offtopic
if active.get("offtopic"):
    found = find_df_with_targets(["url","embedding"], prefer_keys=["emb"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_emb, cm = found
        urlc, ec = cm["url"], cm["embedding"]

        def parse_vec(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                try: return np.array(x, dtype=float)
                except Exception: return None
            if isinstance(x, str):
                xs = x.strip()
                if xs.startswith("[") and xs.endswith("]"):
                    try: return np.array(json.loads(xs), dtype=float)
                    except Exception: pass
                parts = re.split(r"[,\s;|]+", xs.strip("[]() "))
                try:
                    vals = [float(p) for p in parts if p!=""]
                    return np.array(vals, dtype=float) if len(vals)>0 else None
                except Exception:
                    return None
            return None

        tmp = df_emb[[urlc, ec]].copy()
        tmp["_vec"] = tmp[ec].map(parse_vec)

        lengths = tmp["_vec"].dropna().map(lambda v: len(v)).tolist()
        if len(lengths) == 0:
            results["offtopic"] = pd.Series(0.0, index=master_urls.index)
            debug_cols["offtopic"] = {"similarity": pd.Series([np.nan]*len(master_urls))}
        else:
            dominant_len = Counter(lengths).most_common(1)[0][0]

            def pad_or_trunc(v: Optional[np.ndarray], L: int) -> Optional[np.ndarray]:
                if v is None: return None
                n = len(v)
                if n == L: return v
                if n > L:  return v[:L]
                vv = np.zeros(L, dtype=float); vv[:n] = v; return vv

            tmp["_vec2"] = tmp["_vec"].map(lambda v: pad_or_trunc(v, dominant_len))
            valid = tmp.loc[tmp["_vec2"].map(lambda v: isinstance(v, np.ndarray))].copy()
            if valid.empty:
                results["offtopic"] = pd.Series(0.0, index=master_urls.index)
                debug_cols["offtopic"] = {"similarity": pd.Series([np.nan]*len(master_urls))}
            else:
                mat = np.vstack(valid["_vec2"].values)
                norms = np.linalg.norm(mat, axis=1)
                p5, p95 = np.percentile(norms, [5, 95])
                keep_mask = (norms >= p5) & (norms <= p95)
                mat_robust = mat[keep_mask] if keep_mask.any() else mat
                centroid = mat_robust.mean(axis=0)

                def cos_sim(vec):
                    a = vec/(np.linalg.norm(vec)+1e-12); b = centroid/(np.linalg.norm(centroid)+1e-12)
                    return float(np.dot(a,b))

                valid.loc[:, "_sim"] = valid["_vec2"].map(cos_sim)
                d = master_urls.merge(valid[[urlc,"_sim"]], left_on="url_norm", right_on=urlc, how="left")
                sim = d["_sim"].fillna(-1.0)
                results["offtopic"] = pd.Series((sim >= st.session_state.get("offtopic_tau", 0.5)).astype(float))
                debug_cols["offtopic"] = {"similarity": sim}
    elif master_urls is not None:
        results["offtopic"] = pd.Series(0.0, index=master_urls.index)

if active.get("offtopic") and "offtopic" in debug_cols and "similarity" in debug_cols["offtopic"]:
    tau = st.sidebar.slider("Offtopic-Threshold τ (Ähnlichkeit)", 0.0, 1.0, 0.5, 0.01)
    st.session_state["offtopic_tau"] = tau
    sim = debug_cols["offtopic"]["similarity"]
    results["offtopic"] = pd.Series((sim >= tau).astype(float))

# Revenue
if active.get("revenue"):
    found = find_df_with_targets(["url","revenue"], prefer_keys=["rev"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_r, cm = found
        d = join_on_master(df_r, cm["url"], [cm["revenue"]])
        results["revenue"] = mode_score(d[cm["revenue"]]).fillna(0.0)
        debug_cols["revenue"] = {"revenue_raw": d[cm["revenue"]]}
    elif master_urls is not None:
        results["revenue"] = pd.Series(0.0, index=master_urls.index)

# SEO Efficiency
if active.get("seo_eff"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("SEO-Effizienz – Optionen")
    s0 = st.sidebar.slider(
        "Glättungsstärke s₀ (Bayes-Prior)",
        min_value=0, max_value=200, value=20, step=5,
        help="Je höher s₀, desto stärker wird der Top-5-Anteil bei wenigen Rankings zum globalen Durchschnitt p₀ hingezogen."
    )
    vol_weight = st.sidebar.slider(
        "Zusatzgewicht fürs Ranking-Volumen",
        min_value=0.0, max_value=0.5, value=0.2, step=0.05,
        help="Optional: Leichte Bevorzugung von URLs mit vielen Rankings. 0 = aus."
    )

    found = find_df_with_targets(
        ["keyword","url","position"],
        prefer_keys=["eff_kw", "sc_eff", "sc", "sc_perf"],
        use_autodiscovery=use_autodiscovery
    )
    if found and master_urls is not None:
        _, df_e, cm = found
        urlc, posc = cm["url"], cm["position"]
        df_e = ensure_url_column(df_e, urlc).copy()
        pos_num = to_numeric_smart(df_e[posc])

        # k = #Top-5 je URL, n = #Keywords je URL
        grp_k = (pos_num <= 5).groupby(df_e[urlc]).sum().rename("_k_top5").astype(float)
        grp_n = pos_num.groupby(df_e[urlc]).size().rename("_n_all").astype(float)

        # globaler Durchschnitt p0
        total_k = float(grp_k.sum())
        total_n = float(grp_n.sum())
        p0 = (total_k / total_n) if total_n > 0 else 0.0

        # Bayes-Glättung: (k + p0*s0) / (n + s0)
        post = ((grp_k + p0 * s0) / (grp_n + s0)).rename("_eff_bayes").astype(float)

        # Join auf Master + Scores
        tmp = pd.concat([grp_k, grp_n, post], axis=1)
        d = master_urls.merge(tmp, left_on="url_norm", right_index=True, how="left")

        eff_series = d["_eff_bayes"].fillna(0.0)

        if vol_weight > 0:
            n_score = mode_score(d["_n_all"].fillna(0.0))
            eff_score = mode_score(eff_series)
            # Mischung: (1 - vol_weight) * Effizienz + vol_weight * Volumen
            mixed = (1.0 - vol_weight) * eff_score + vol_weight * n_score
            results["seo_eff"] = mixed.fillna(0.0)
        else:
            results["seo_eff"] = mode_score(eff_series).fillna(0.0)

        debug_cols["seo_eff"] = {
            "k_top5": d["_k_top5"],
            "n_all": d["_n_all"],
            "p0_global": pd.Series([p0] * len(d)),
            "eff_bayes": eff_series,
            "seo_eff_score": results["seo_eff"],
        }

        with st.expander("Vorschau: SEO-Effizienz (Top 10)", expanded=False):
            prev = pd.DataFrame({
                "url": d["url_norm"],
                "k_top5": d["_k_top5"],
                "n_all": d["_n_all"],
                "p0_global": p0,
                "eff_bayes": eff_series,
                "seo_eff_score": results["seo_eff"],
            })
            st.dataframe(prev.head(10), use_container_width=True)

    elif master_urls is not None:
        results["seo_eff"] = pd.Series(0.0, index=master_urls.index)


# Priority override
priority_url = None
if active.get("priority"):
    found = find_df_with_targets(["url","priority_factor"], prefer_keys=["prio"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_p, cm = found
        d = join_on_master(df_p, cm["url"], [cm["priority_factor"]])
        pf = to_numeric_smart(d[cm["priority_factor"]]).fillna(1.0)
        priority_url = pf.clip(lower=0.5, upper=2.0)
    elif master_urls is not None:
        priority_url = pd.Series(1.0, index=master_urls.index)

# Hauptkeyword Expected Clicks
if active.get("main_kw_exp"):
    found_any = find_df_with_targets(["url","main_keyword"], prefer_keys=["main_kw"], use_autodiscovery=use_autodiscovery)
    if found_any and master_urls is not None:
        _, df_m, cm = found_any
        urlc = cm["url"]
        expc = find_first_alias(df_m, "expected_clicks")
        if expc:
            j = join_on_master(df_m, urlc, [cm["main_keyword"], expc])
            vals = to_numeric_smart(j[expc])
            results["main_kw_exp"] = mode_score(vals).fillna(0.0)
            debug_cols["main_kw_exp"] = {"main_keyword": j[cm["main_keyword"]], "expected_clicks_raw": j[expc]}
        else:
            svc = find_first_alias(df_m, "search_volume")
            posc = find_first_alias(df_m, "position")
            if svc and posc:
                found_ctr = find_df_with_targets(["position","ctr"], prefer_keys=["ctr_curve"], use_autodiscovery=use_autodiscovery)
                if found_ctr:
                    _, ctr_df, ctr_map = found_ctr
                    ctr_df = ctr_df[[ctr_map["position"], ctr_map["ctr"]]].rename(columns={ctr_map["position"]:"position", ctr_map["ctr"]:"ctr"})
                else:
                    ctr_df = default_ctr_curve()
                df_m = ensure_url_column(df_m, urlc)
                ctrs = df_m[posc].map(lambda p: get_ctr_for_pos(p, ctr_df))
                exp_calc = to_numeric_smart(df_m[svc]).fillna(0) * ctrs
                agg = df_m.assign(_exp=exp_calc).groupby(urlc, as_index=False)["_exp"].sum()
                j = master_urls.merge(agg, left_on="url_norm", right_on=urlc, how="left")
                vals = j["_exp"]
                results["main_kw_exp"] = mode_score(vals).fillna(0.0)
                debug_cols["main_kw_exp"] = {"expected_clicks_raw": vals}
            else:
                results["main_kw_exp"] = pd.Series(0.0, index=master_urls.index)
    elif master_urls is not None:
        results["main_kw_exp"] = pd.Series(0.0, index=master_urls.index)

# Hauptkeyword Suchvolumen
if active.get("main_kw_sv"):
    found_any = find_df_with_targets(["url","main_keyword"], prefer_keys=["main_kw"], use_autodiscovery=use_autodiscovery)
    if found_any and master_urls is not None:
        _, df_m, cm = found_any
        urlc = cm["url"]
        svc = find_first_alias(df_m, "search_volume")
        if svc:
            j = join_on_master(df_m, urlc, [cm["main_keyword"], svc])
            vals = to_numeric_smart(j[svc])
            results["main_kw_sv"] = mode_score(vals).fillna(0.0)
            debug_cols["main_kw_sv"] = {"main_keyword": j[cm["main_keyword"]], "search_volume_raw": vals}
        else:
            results["main_kw_sv"] = pd.Series(0.0, index=master_urls.index)
    elif master_urls is not None:
        results["main_kw_sv"] = pd.Series(0.0, index=master_urls.index)

# ============= Gewichte & Aggregation =============
st.subheader("Gewichtung der aktiven Kriterien")
weight_keys = [k for k in [
    "sc_clicks","sc_impr","sc_perf_class","seo_eff","main_kw_sv","main_kw_exp",
    "ai_overview",
    "ext_pop","int_pop","llm_ref","llm_crawl",
    "otv","revenue","offtopic",
    "overall_traffic",
    "llm_citations",
] if active.get(k)]

weights: Dict[str, float] = {}
if weight_keys:
    cols = st.columns(len(weight_keys))
    for i, k in enumerate(weight_keys):
        label = None
        for _, crits in CRITERIA_GROUPS.items():
            for code, l, _ in crits:
                if code == k:
                    label = l; break
            if label: break
        weights[k] = cols[i].number_input(f"Gewicht: {label}", min_value=0.0, value=1.0, step=0.1, key=f"w_{k}")
else:
    st.info("Keine Kriterien aktiv. Bitte mindestens ein Kriterium aktivieren.")

w_sum = sum(weights.values()) if weights else 0.0
weights_norm = {k: (v / w_sum) for k, v in weights.items()} if w_sum > 0 else {k: 0.0 for k in weight_keys}

if master_urls is not None and weight_keys:
    df_out = master_urls.copy()
    for k in weight_keys:
        s = results.get(k, pd.Series(0.0, index=df_out.index))
        df_out[f"score__{k}"] = s.values
    base = np.zeros(len(df_out))
    for k, wn in weights_norm.items():
        base += wn * df_out[f"score__{k}"].values
    df_out["base_score"] = base

    if active.get("priority"):
        if 'priority_url' in locals() and priority_url is not None:
            df_out["priority_factor_url"] = priority_url.values
        else:
            df_out["priority_factor_url"] = 1.0
        df_out["final_score"] = df_out["base_score"] * df_out["priority_factor_url"]
    else:
        df_out["priority_factor_url"] = 1.0
        df_out["final_score"] = df_out["base_score"]

    df_out = df_out.sort_values("final_score", ascending=False).reset_index(drop=True)

    st.subheader("Ergebnis")
    st.dataframe(df_out.head(100), use_container_width=True, hide_index=True)

    st.markdown("### Export")
    csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ CSV herunterladen", data=csv_bytes, file_name="one_url_scoreboard.csv", mime="text/csv")

    # (Bonus) XLSXWriter: Hinweis-Button, falls nicht installiert
    try:
        import xlsxwriter
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="scores")
            for k, cols in debug_cols.items():
                if isinstance(cols, dict) and cols:
                    pd.DataFrame(cols).to_excel(writer, index=False, sheet_name=f"raw_{k}"[:31])
        st.download_button("⬇️ XLSX herunterladen", data=buf.getvalue(), file_name="one_url_scoreboard.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except ImportError:
        st.warning("Für den XLSX-Export wird das Paket `xlsxwriter` benötigt.")
        if st.button("Installationshinweis anzeigen"):
            st.code("pip install XlsxWriter")
    except Exception:
        st.caption("Hinweis: Für XLSX-Export kann das Paket `xlsxwriter` erforderlich sein.")

    config = {
        "scoring_mode": scoring_mode,
        "min_score": min_score if scoring_mode == "Rank (linear)" else None,
        "bucket_quantiles": [0.0, 0.5, 0.75, 0.9, 0.97, 1.0] if scoring_mode != "Rank (linear)" else None,
        "bucket_values": [0.0, 0.25, 0.5, 0.75, 1.0] if scoring_mode != "Rank (linear)" else None,
        "use_autodiscovery": use_autodiscovery,
        "weights": weights,
        "weights_norm": weights_norm,
        "active_criteria": [k for k, v in active.items() if v],
        "uploads_used": {k: name for k, (_, name) in st.session_state.uploads.items()},
        "column_maps": st.session_state.column_maps,
        "llm_bot_selection": {
            "detected": st.session_state.llm_bot_detected,
            "include": st.session_state.llm_bot_include,
            "exclude": st.session_state.llm_bot_exclude,
            "custom_include": st.session_state.llm_bot_custom_include,
            "custom_exclude": st.session_state.llm_bot_custom_exclude,
        },
        "notes": "Masterliste & Scoring-Setup siehe UI. CSV robust: Trenner-/Encoding-Heuristik, Komma-Dezimale, Header-Fix.",
    }
    st.download_button("⬇️ Config (JSON)",
        data=json.dumps(config, indent=2).encode("utf-8"),
        file_name="one_url_scoreboard_config.json",
        mime="application/json",
    )
else:
    st.info("Bitte Masterliste erzeugen und mindestens ein Kriterium aktivieren & gewichten.")
