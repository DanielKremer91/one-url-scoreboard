# app.py
# ONE URL Scoreboard — Streamlit App
# Author idea: Daniel Kremer (ONE Beyond Search) — implementation by ChatGPT
# Version: Union/Intersection Masterlist modes + include All-Inlinks in union + Page URL alias + robust compound column split

import io
import json
import re
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

# ============= Page & CSS =============
st.set_page_config(page_title="ONE URL Scoreboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.main .block-container { max-width: 100% !important; padding: 1rem 1rem 2rem !important; }
[data-testid="stSidebar"] { min-width: 260px !important; max-width: 260px !important; }
.reportview-container .main .block-container { max-width: 100% !important; padding: 1rem 1rem 2rem !important; }
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
**ONE URL Scoreboard** priorisiert URLs anhand wählbarer Kriterien nach globalem Scoring-Modus (*Rank linear* oder *Perzentil-Buckets*).  
Fehlende Daten im **aktiven** Kriterium ⇒ Score = 0 (kein Reweighting pro URL).

**Neu / wichtig**
- **Master-URL-Liste**: Modi **Union (Voreinstellung)**, **Gemeinsamer Nenner (Schnittmenge)** oder **Aus zwei Quellen**.
- **All-Inlinks-Datei** ist jetzt **in der Union enthalten** (wir verwenden die empfangenden Ziel-URLs).
- **Cross-File-Fallback** (Schema-Index): Wenn die „vorgesehene“ Datei/Spalte fehlt, sucht das Tool in anderen Uploads passende Spalten (per Alias).
- **SC-Aggregation**: Query-Level-SC-Dateien werden automatisch auf URL-Ebene aggregiert (Clicks/Impressions summiert).
- **Embeddings robust**: Fehlende Embeddings ⇒ Outlier (unter τ). Uneinheitliche Vektorlängen ⇒ Padding/Trunc auf dominante Dimension.
- **Compound-Spalten** (z. B. *URL;RD;BL*) werden automatisch in `url`, `ref_domains`, `backlinks` gesplittet.
""")

# ============= Session & Helpers =============
if "uploads" not in st.session_state:
    st.session_state.uploads = {}  # key -> (df, name)
if "column_maps" not in st.session_state:
    st.session_state.column_maps = {}
if "schema_index" not in st.session_state:
    st.session_state.schema_index = {}

ALIASES = {
    # URL: erweitert um "page_url"
    "url": [
        "url","page","page_url","seite","address","adresse","target","ziel","ziel_url","landing_page"
    ],
    "clicks": ["clicks","klicks","sc_clicks"],
    "impressions": ["impressions","impr","impressionen","search_impressions"],
    "position": ["position","avg_position","average_position","durchschnittliche_position","durchschn._position","rank","avg_rank"],
    "search_volume": ["search_volume","sv","volume","suchvolumen"],
    "cpc": ["cpc","cost_per_click"],
    "traffic_value": ["traffic_value","otv","organic_traffic_value","value","potential_value"],
    "potential_traffic_url": ["potential_traffic_url","potential_traffic","pot_traffic"],
    "backlinks": ["backlinks","links_total","bl","inbound_links"],
    "ref_domains": ["ref_domains","referring_domains","rd","domains_ref","verweisende_domains"],
    "unique_inlinks": ["unique_inlinks","internal_inlinks","inlinks_unique","eingehenden_links","inlinks","eingehende_links","eingehenden_link"],
    "llm_ref_traffic": ["llm_ref_traffic","llm_referrals","ai_referrals","llm_popularity","llm_traffic"],
    "llm_crawl_freq": ["llm_crawl_freq","ai_crawls","llm_crawls","llm_crawler_visits"],
    "embedding": ["embedding","embeddings","vector","vec","embedding_json"],
    "revenue": ["revenue","umsatz","organic_revenue","organic_umsatz","organic_sales"],
    "priority_factor": ["priority_factor","prio","priority","override","weight_override"],
    "keyword": ["keyword","query","suchbegriff","suchanfrage"],
    # Hauptkeyword-Potenzial
    "main_keyword": ["main_keyword","hauptkeyword","primary_keyword","focus_keyword","focus_kw","haupt_kw","haupt-keyword"],
    "expected_clicks": ["expected_clicks","exp_clicks","expected_clicks_main","expected_clicks_kw","erwartete_klicks","erw_klicks"],
}

TRACKING_PARAMS_PREFIXES = ["utm_", "icid_"]
TRACKING_PARAMS_EXACT = {"gclid","fbclid","msclkid","mc_eid","yclid"}

def normalize_header(col: str) -> str:
    c = col.strip().lower()
    c = re.sub(r"[^\w]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_header(c) for c in df.columns]
    return df

def strip_tracking_params(qs: str) -> str:
    if not qs: return ""
    kept = []
    for pair in qs.split("&"):
        if "=" in pair:
            k, v = pair.split("=", 1)
        else:
            k, v = pair, ""
        kl = k.lower()
        if any(kl.startswith(p) for p in TRACKING_PARAMS_PREFIXES): continue
        if kl in TRACKING_PARAMS_EXACT: continue
        kept.append(pair)
    return "&".join(kept)

def normalize_url(u: str) -> Optional[str]:
    if not isinstance(u, str) or not u.strip(): return None
    s = u.strip().lower()
    if "#" in s: s = s.split("#", 1)[0]
    if "?" in s:
        base, qs = s.split("?", 1)
        qs2 = strip_tracking_params(qs)
        s = base if not qs2 else f"{base}?{qs2}"
    return s

def read_table(uploaded) -> pd.DataFrame:
    data = uploaded.read()
    try:
        df = pd.read_csv(io.BytesIO(data), low_memory=False)
    except Exception:
        df = pd.read_excel(io.BytesIO(data))
    return normalize_headers(df)

# ---- Compound-Spalten automatisch splitten (URL;RD;BL) ----
def try_split_compound_url_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Wenn schon eine URL-Spalte existiert, nichts tun
    url_aliases = set(ALIASES["url"])
    if any(c in df.columns for c in url_aliases):
        return df

    # Kandidaten (z. B. "page_url_referring_domains_links_to_target") oder 1-Spalten-CSV
    cand_cols = [c for c in df.columns if ("url" in c and ("ref" in c or "link" in c))]
    if len(df.columns) == 1:  # 1-Spalten-Datei → diese Spalte testen
        cand_cols = [df.columns[0]]

    if not cand_cols:
        return df

    col = cand_cols[0]
    s = df[col].astype(str)
    if not s.str.contains(";").any():
        return df

    parts = s.str.split(";", n=2, expand=True)
    if parts.shape[1] < 1:
        return df

    df2 = df.copy()
    df2["url"] = parts[0].astype(str)
    if parts.shape[1] >= 2:
        df2["ref_domains"] = pd.to_numeric(parts[1], errors="coerce")
    if parts.shape[1] >= 3:
        df2["backlinks"] = pd.to_numeric(parts[2], errors="coerce")
    return df2

def store_upload(key: str, file):
    if file is None: return
    df = read_table(file)
    df = try_split_compound_url_metrics(df)  # robust: spaltet ggf. URL;RD;BL auf
    st.session_state.uploads[key] = (df, file.name)

def find_first_alias(df: pd.DataFrame, target: str) -> Optional[str]:
    candidates = ALIASES.get(target, [])
    for c in candidates:
        if c in df.columns: return c
    cols_norm = {re.sub(r"[_\s]+","", c): c for c in df.columns}
    for c in candidates:
        cn = re.sub(r"[_\s]+","", c)
        if cn in cols_norm: return cols_norm[cn]
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
    s = pd.to_numeric(series, errors="coerce").clip(lower=0)
    mask = s.notna()
    if mask.sum() <= 1:
        out = pd.Series(0.0, index=s.index); out[mask] = 1.0; return out
    ranks = s[mask].rank(method="average", ascending=False)  # 1=best
    out = pd.Series(0.0, index=s.index)
    n = len(ranks)
    out[mask] = 1.0 - (ranks - 1) / (n - 1) * (1.0 - min_score)
    return out

def bucket_scores(series: pd.Series,
                  quantiles: List[float] = [0.0, 0.5, 0.75, 0.9, 0.97, 1.0],
                  bucket_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").clip(lower=0)
    mask = s.notna()
    res = pd.Series(0.0, index=s.index)
    if mask.sum() == 0: return res
    try:
        qvals = s[mask].quantile(quantiles).values
        bins = np.unique(qvals)
        if len(bins) < 3:
            mn, mx = float(s[mask].min()), float(s[mask].max())
            if mx <= mn: res[mask] = bucket_values[-1]; return res
            bins = np.linspace(mn, mx + 1e-9, num=6)
        cats = pd.cut(s[mask], bins=bins, include_lowest=True, labels=False)
        bv = dict(zip(range(len(bucket_values)), bucket_values))
        res[mask] = cats.map(lambda i: bv.get(int(i), 0.0)).astype(float)
        return res
    except Exception:
        return rank_scores(series, min_score=0.2)

def score_series(series: pd.Series, mode: str, min_score: float,
                 quantiles: List[float], bucket_values: List[float]) -> pd.Series:
    return rank_scores(series, min_score) if mode == "Rank (linear)" else bucket_scores(series, quantiles, bucket_values)

def default_ctr_curve() -> pd.DataFrame:
    return pd.DataFrame({"position": list(range(1, 21)),
                         "ctr": [0.30,0.15,0.10,0.07,0.05,0.04,0.035,0.03,0.025,0.02,0.018,0.016,0.014,0.012,0.010,0.009,0.008,0.007,0.006,0.005]})

def get_ctr_for_pos(pos: float, ctr_df: pd.DataFrame) -> float:
    try: p = int(round(float(pos)))
    except Exception: return 0.0
    p = max(1, min(p, int(ctr_df["position"].max())))
    row = ctr_df.loc[ctr_df["position"] == p]
    return float(row["ctr"].values[0]) if not row.empty else 0.0

# ============= Sidebar settings (mit Hilfetexten) =============
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
        help="Nur für Rank (linear): Der schlechtesten (vorhandenen) URL wird mindestens dieser Score zugewiesen. Fehlende Werte bekommen immer 0.0."
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

offtopic_tau = st.sidebar.slider(
    "Offtopic-Threshold τ (Ähnlichkeit)",
    0.0, 1.0, 0.5, 0.01,
    help="Binäres Gate im Offtopic-Kriterium: Cosine-Similarity ≥ τ ⇒ 1.0, sonst 0.0. Fehlende Embeddings zählen als < τ."
)
priority_global = st.sidebar.slider(
    "Globaler Prioritäts-Faktor",
    0.5, 2.0, 1.0, 0.05,
    help="Skaliert den finalen Score aller URLs global (z. B. 1.2 = +20 %). In Kombination mit per-URL-Overrides nutzbar."
)
use_autodiscovery = st.sidebar.toggle(
    "Autodiscovery über alle Uploads",
    value=True,
    help="Wenn aktiv: Fehlen die vorgesehenen Dateien/Spalten, sucht das Tool automatisch in anderen Uploads nach passenden Spalten (per Alias)."
)

# ============= Kriterienauswahl (links im Hauptbereich) =============
CRITERIA = [
    ("sc_clicks", "SC Clicks", "Search Console Clicks."),
    ("sc_impr",   "SC Impressions", "Search Console Impressions."),
    ("otv",       "Organic Traffic Value", "URL-Value vorhanden ODER via Keyword × CTR."),
    ("ext_pop",   "URL-Popularität extern", "Ref. Domains 70% + Backlinks 30%."),
    ("int_pop",   "URL-Popularität intern", "Unique interne Inlinks."),
    ("llm_ref",   "LLM-Popularität (Referrals)", "LLM/AI Referral Traffic."),
    ("llm_crawl", "LLM-Crawl-Frequenz", "AI/LLM Crawler Visits."),
    ("offtopic",  "Offtopic-Score (0/1)", "Cosine-Similarity Gate via τ."),
    ("revenue",   "Umsatz", "Revenue je URL."),
    ("seo_eff",   "URL-SEO-Effizienz", "Anteil Top-5-Keywords je URL."),
    ("priority",  "Strategische Priorität (Override)", "Multiplikator je URL + global."),
    ("main_kw",   "Hauptkeyword-Potenzial (SV/Expected Clicks)", "URL + Hauptkeyword + erwartete Klicks ODER Suchvolumen."),
]

st.subheader("Kriterien auswählen")
st.caption("Wähle unten die gewünschten Kriterien. Danach erscheinen die passenden Upload-Masken.")
active = {code: st.checkbox(label, value=False, help=hover, key=f"chk_{code}") for code, label, hover in CRITERIA}

# ============= Upload-Masken (nach Auswahl) =============
uploads_needed_text: List[str] = []
def need(s: str): uploads_needed_text.append(s)

if active.get("sc_clicks") or active.get("sc_impr"): need("SC-Datei (URL, Clicks/Impressions; Query-Ebene ok)")
if active.get("otv"): need("OTV-URL (optional), OTV-Keyword (optional), CTR-Kurve (optional)")
if active.get("ext_pop"): need("Extern-Popularität (URL, backlinks, ref_domains)")
if active.get("int_pop"): need("Intern-Popularität (URL, unique_inlinks ODER Kantenliste)")
if active.get("llm_ref"): need("LLM Referral Traffic (URL, llm_ref_traffic)")
if active.get("llm_crawl"): need("LLM Crawler Frequenz (URL, llm_crawl_freq)")
if active.get("offtopic"): need("Embeddings (URL, embedding)")
if active.get("revenue"): need("Umsatz (URL, revenue)")
if active.get("seo_eff"): need("Keywords (keyword, url, position)")
if active.get("priority"): need("Prioritäten-Mapping (URL, priority_factor) — optional")
if active.get("main_kw"): need("Hauptkeyword (URL, main_keyword, expected_clicks ODER search_volume)")

st.markdown("---")
st.subheader("Basierend auf den gewählten Kriterien benötigen wir folgende Daten von dir")
st.markdown("- " + "\n- ".join(dict.fromkeys(uploads_needed_text)) if uploads_needed_text else "Noch keine Kriterien gewählt.")

# Upload-Inputs
if active.get("sc_clicks") or active.get("sc_impr"):
    st.markdown("**Search Console Organic Performance**")
    store_upload("sc", st.file_uploader("SC-Datei (CSV/XLSX)", type=["csv","xlsx"], key="upl_sc"))

if active.get("otv"):
    st.markdown("**Organic Traffic Value (OTV)**")
    c1, c2, c3 = st.columns(3)
    with c1: store_upload("otv_url", st.file_uploader("OTV: URL-Value (optional)", type=["csv","xlsx"], key="upl_otv_url"))
    with c2: store_upload("otv_kw",  st.file_uploader("OTV: Keyword-Datei (optional)", type=["csv","xlsx"], key="upl_otv_kw"))
    with c3: store_upload("ctr_curve", st.file_uploader("CTR-Kurve (optional)", type=["csv","xlsx"], key="upl_ctr"))

if active.get("ext_pop"):
    st.markdown("**URL-Popularität extern**")
    store_upload("ext", st.file_uploader("Extern-Popularität", type=["csv","xlsx"], key="upl_ext"))

if active.get("int_pop"):
    st.markdown("**URL-Popularität intern**")
    store_upload("int", st.file_uploader("Intern-Popularität (Crawl/Inlinks)", type=["csv","xlsx"], key="upl_int"))

if active.get("llm_ref"):
    st.markdown("**LLM Referral Traffic**")
    store_upload("llmref", st.file_uploader("LLM-Referral", type=["csv","xlsx"], key="upl_llmref"))

if active.get("llm_crawl"):
    st.markdown("**LLM-Crawl-Frequenz**")
    store_upload("llmcrawl", st.file_uploader("LLM-Crawl", type=["csv","xlsx"], key="upl_llmcrawl"))

if active.get("offtopic"):
    st.markdown("**Offtopic (Embeddings)**")
    store_upload("emb", st.file_uploader("Embeddings-Datei", type=["csv","xlsx"], key="upl_emb"))

if active.get("revenue"):
    st.markdown("**Umsatz**")
    store_upload("rev", st.file_uploader("Umsatz-Datei", type=["csv","xlsx"], key="upl_rev"))

if active.get("seo_eff"):
    st.markdown("**SEO-Effizienz (Top-5-Anteil)**")
    store_upload("eff_kw", st.file_uploader("Keyword-Datei (SEO-Effizienz)", type=["csv","xlsx"], key="upl_eff_kw"))

if active.get("priority"):
    st.markdown("**Strategische Priorität (optional)**")
    store_upload("prio", st.file_uploader("Priorität (optional)", type=["csv","xlsx"], key="upl_prio"))

if active.get("main_kw"):
    st.markdown("**Hauptkeyword-Potenzial**")
    store_upload("main_kw", st.file_uploader("Hauptkeyword-Mapping", type=["csv","xlsx"], key="upl_main_kw"))

# ---------- (Re)build schema index after uploads changed ----------
build_schema_index()

# ============= Master URL list builder =============
st.subheader("Master-URL-Liste")
st.markdown("Wähle, wie die Masterliste gebildet wird: **Union** (Vereinigung, Default), **Gemeinsamer Nenner (Schnittmenge)** oder **aus zwei Quellen**. Eigene Liste ist ebenfalls möglich.")

master_mode = st.radio(
    "Masterlisten-Modus",
    ["Union (alle Uploads)", "Gemeinsamer Nenner (Schnittmenge)", "Aus zwei vorhandenen Quellen"],
    index=0,
    help=(
        "**Union:** Alle URLs aus allen Uploads (Duplikate entfernt). "
        "**Schnittmenge:** Nur URLs, die in *allen* berücksichtigten Uploads vorkommen. "
        "**Zwei Quellen:** Masterliste aus genau zwei gewählten Uploads."
    )
)

use_custom_master = st.checkbox("Eigene Masterliste hochladen (statt oben gewähltem Modus)")

master_urls: Optional[pd.DataFrame] = None

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

def collect_urls_intersection() -> Optional[pd.DataFrame]:
    sets = []
    sources = 0
    for key, (df, _) in st.session_state.uploads.items():
        c = find_first_alias(df, "url")
        if not c:
            continue
        d = ensure_url_column(df[[c]], c)
        s = set(d[c].dropna().unique().tolist())
        if len(s) == 0:
            continue
        sets.append(s)
        sources += 1
    if sources == 0:
        return None
    inter = set.intersection(*sets) if len(sets) > 1 else sets[0]
    if not inter:
        return pd.DataFrame({"url_norm": []})
    return pd.DataFrame({"url_norm": sorted(inter)})

if use_custom_master:
    mf = st.file_uploader("Eigene Masterliste (Spalte: url/seite/page_url/...)", type=["csv","xlsx"], key="upl_master")
    if mf:
        dfm = read_table(mf)
        dfm = try_split_compound_url_metrics(dfm)
        url_col = find_first_alias(dfm, "url") or st.selectbox("URL-Spalte in Masterliste wählen", dfm.columns, key="map_master_url")
        dfm = ensure_url_column(dfm, url_col)
        master_urls = dfm[[url_col]].rename(columns={url_col: "url_norm"}).drop_duplicates()
else:
    if master_mode == "Union (alle Uploads)":
        master_urls = collect_urls_union()
    elif master_mode == "Gemeinsamer Nenner (Schnittmenge)":
        master_urls = collect_urls_intersection()
    else:  # aus zwei Quellen
        available = list(st.session_state.uploads.keys())
        pick1 = st.selectbox("Quelle 1", options=[None] + available, index=0, key="master_src1")
        pick2 = st.selectbox("Quelle 2 (optional)", options=[None] + available, index=0, key="master_src2")
        if pick1:
            keys = [pick1] + ([pick2] if pick2 else [])
            master_urls = collect_urls_union(include_keys=keys)

if master_urls is None or master_urls.empty:
    st.info("Noch keine Master-URLs erkannt. Lade mindestens eine Datei mit URL-Spalte hoch **oder** lade eine eigene Masterliste.")
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

# --- SC Clicks / Impressions (supports query-level rows; aggregates by URL) ---
if active.get("sc_clicks") or active.get("sc_impr"):
    need_cols = ["url"] + (["clicks"] if active.get("sc_clicks") else []) + (["impressions"] if active.get("sc_impr") else [])
    found = find_df_with_targets(need_cols, prefer_keys=["sc"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_sc, colmap = found
        urlc = colmap["url"]
        df_sc = ensure_url_column(df_sc, urlc).copy()
        metrics = []
        if active.get("sc_clicks"): metrics.append(colmap["clicks"])
        if active.get("sc_impr"):  metrics.append(colmap["impressions"])
        for m in metrics:
            df_sc[m] = pd.to_numeric(df_sc[m], errors="coerce").fillna(0)
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

# --- OTV (URL-Value bevorzugt, sonst Keyword-basiert) ---
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
                raw_val = pd.to_numeric(d[val_col], errors="coerce")
            elif pot_col is not None and cpc_col is not None:
                raw_val = pd.to_numeric(d[pot_col], errors="coerce") * pd.to_numeric(d[cpc_col], errors="coerce")
            elif pot_col is not None:
                raw_val = pd.to_numeric(d[pot_col], errors="coerce")
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
            pot_traffic = pd.to_numeric(df_k[svc], errors="coerce").fillna(0) * ctrs
            raw_row_val = pot_traffic * pd.to_numeric(df_k[cpcc], errors="coerce").fillna(0) if cpcc else pot_traffic
            agg = df_k.assign(_val=raw_row_val).groupby(urlc, as_index=False)["_val"].sum()
            d = master_urls.merge(agg, left_on="url_norm", right_on=urlc, how="left")
            raw_val = d["_val"]
    if master_urls is not None:
        results["otv"] = pd.Series(0.0, index=master_urls.index) if raw_val is None else mode_score(raw_val).fillna(0.0)
        if raw_val is not None: debug_cols["otv"] = {"otv_raw": raw_val}

# --- External popularity (Backlinks & RD) ---
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

# --- Internal popularity (Unique Inlinks) ---
if active.get("int_pop"):
    # Bevorzugt fertige Zählspalte; Fallback: Kantenliste (source -> url) → distinct Quellen zählen
    found = find_df_with_targets(["url","unique_inlinks"], prefer_keys=["int"], use_autodiscovery=use_autodiscovery)

    if not found:
        found_edges = find_df_with_targets(["url"], prefer_keys=["int"], use_autodiscovery=use_autodiscovery)
        if found_edges:
            _, df_edges, cm_edges = found_edges
            urlc = cm_edges["url"]
            src_candidates = [
                "source","from","src","source_url","referrer","referrer_url",
                "origin","origin_url","quelle","von","inlink_source","inlink_from"
            ]
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
        vals = pd.to_numeric(d[cm["unique_inlinks"]], errors="coerce").clip(lower=0)
        results["int_pop"] = mode_score(vals).fillna(0.0)
        debug_cols["int_pop"] = {"unique_inlinks_raw": vals}
    elif master_urls is not None:
        results["int_pop"] = pd.Series(0.0, index=master_urls.index)

# --- LLM Referral ---
if active.get("llm_ref"):
    found = find_df_with_targets(["url","llm_ref_traffic"], prefer_keys=["llmref"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_l, cm = found
        d = join_on_master(df_l, cm["url"], [cm["llm_ref_traffic"]])
        results["llm_ref"] = mode_score(d[cm["llm_ref_traffic"]]).fillna(0.0)
        debug_cols["llm_ref"] = {"llm_ref_traffic_raw": d[cm["llm_ref_traffic"]]}
    elif master_urls is not None:
        results["llm_ref"] = pd.Series(0.0, index=master_urls.index)

# --- LLM Crawl ---
if active.get("llm_crawl"):
    found = find_df_with_targets(["url","llm_crawl_freq"], prefer_keys=["llmcrawl"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_lc, cm = found
        d = join_on_master(df_lc, cm["url"], [cm["llm_crawl_freq"]])
        results["llm_crawl"] = mode_score(d[cm["llm_crawl_freq"]]).fillna(0.0)
        debug_cols["llm_crawl"] = {"llm_crawl_freq_raw": d[cm["llm_crawl_freq"]]}
    elif master_urls is not None:
        results["llm_crawl"] = pd.Series(0.0, index=master_urls.index)

# --- Offtopic (0/1) — robust gegen fehlende/uneinheitliche Embeddings ---
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
                if v is None: 
                    return None
                n = len(v)
                if n == L: 
                    return v
                if n > L:
                    return v[:L]
                vv = np.zeros(L, dtype=float)
                vv[:n] = v
                return vv

            tmp["_vec2"] = tmp["_vec"].map(lambda v: pad_or_trunc(v, dominant_len))
            valid = tmp[tmp["_vec2"].map(lambda v: isinstance(v, np.ndarray))]
            mat = np.vstack(valid["_vec2"].values) if not valid.empty else None

            if mat is None or mat.size == 0:
                results["offtopic"] = pd.Series(0.0, index=master_urls.index)
                debug_cols["offtopic"] = {"similarity": pd.Series([np.nan]*len(master_urls))}
            else:
                centroid = mat.mean(axis=0)

                def cos_sim(vec):
                    a = vec/(np.linalg.norm(vec)+1e-12); b = centroid/(np.linalg.norm(centroid)+1e-12)
                    return float(np.dot(a,b))

                valid["_sim"] = valid["_vec2"].map(cos_sim)

                d = master_urls.merge(valid[[urlc,"_sim"]], left_on="url_norm", right_on=urlc, how="left")
                sim = d["_sim"].copy()
                sim = sim.fillna(-1.0)  # fehlend ⇒ sicher < τ
                s = pd.Series(0.0, index=d.index)
                s.loc[sim >= offtopic_tau] = 1.0
                results["offtopic"] = s.astype(float)
                debug_cols["offtopic"] = {"similarity": sim}

    elif master_urls is not None:
        results["offtopic"] = pd.Series(0.0, index=master_urls.index)

# --- Revenue ---
if active.get("revenue"):
    found = find_df_with_targets(["url","revenue"], prefer_keys=["rev"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_r, cm = found
        d = join_on_master(df_r, cm["url"], [cm["revenue"]])
        results["revenue"] = mode_score(d[cm["revenue"]]).fillna(0.0)
        debug_cols["revenue"] = {"revenue_raw": d[cm["revenue"]]}
    elif master_urls is not None:
        results["revenue"] = pd.Series(0.0, index=master_urls.index)

# --- SEO Efficiency (Top-5 share) ---
if active.get("seo_eff"):
    found = find_df_with_targets(["keyword","url","position"], prefer_keys=["eff_kw"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_e, cm = found
        urlc, posc = cm["url"], cm["position"]
        df_e = ensure_url_column(df_e, urlc)
        grp = df_e.groupby(urlc)[posc].apply(lambda s: (pd.to_numeric(s, errors="coerce") <= 5).sum() / max(1, s.shape[0]))
        d = master_urls.merge(grp.rename("eff"), left_on="url_norm", right_index=True, how="left")
        eff_series = d["eff"].fillna(0.0)
        results["seo_eff"] = mode_score(eff_series).fillna(0.0)
        debug_cols["seo_eff"] = {"eff_raw": eff_series}
    elif master_urls is not None:
        results["seo_eff"] = pd.Series(0.0, index=master_urls.index)

# --- Priority override ---
priority_url = None
if active.get("priority"):
    found = find_df_with_targets(["url","priority_factor"], prefer_keys=["prio"], use_autodiscovery=use_autodiscovery)
    if found and master_urls is not None:
        _, df_p, cm = found
        d = join_on_master(df_p, cm["url"], [cm["priority_factor"]])
        priority_url = pd.to_numeric(d[cm["priority_factor"]], errors="coerce").fillna(1.0).clip(0.5, 2.0)
    elif master_urls is not None:
        priority_url = pd.Series(1.0, index=master_urls.index)

# --- Hauptkeyword-Potenzial (Expected Clicks bevorzugt, sonst SV) ---
if active.get("main_kw"):
    found_any = find_df_with_targets(["url","main_keyword"], prefer_keys=["main_kw"], use_autodiscovery=use_autodiscovery)
    if found_any and master_urls is not None:
        _, df_m, cm = found_any
        urlc, mkc = cm["url"], cm["main_keyword"]
        expc = find_first_alias(df_m, "expected_clicks")
        svc  = find_first_alias(df_m, "search_volume")
        keep = [urlc, mkc] + [c for c in [expc, svc] if c]
        j = join_on_master(df_m, urlc, [c for c in keep if c != urlc])
        val_col = expc or svc
        vals = pd.to_numeric(j[val_col], errors="coerce") if val_col else pd.Series(0.0, index=j.index)
        results["main_kw"] = mode_score(vals).fillna(0.0)
        dbg = {"main_keyword": j[mkc]}
        if val_col: dbg[f"{val_col}_raw"] = j[val_col]
        debug_cols["main_kw"] = dbg
    elif master_urls is not None:
        results["main_kw"] = pd.Series(0.0, index=master_urls.index)

# ============= Gewichte & Aggregation =============
st.subheader("Gewichtung der aktiven Kriterien")
weight_keys = [k for k in ["sc_clicks","sc_impr","otv","ext_pop","int_pop","llm_ref","llm_crawl","offtopic","revenue","seo_eff","main_kw"] if active.get(k)]
weights: Dict[str, float] = {}
if weight_keys:
    cols = st.columns(len(weight_keys))
    for i, k in enumerate(weight_keys):
        label = next(lbl for code,lbl,_ in CRITERIA if code == k)
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
    df_out["priority_factor_url"] = priority_url.values if (active.get("priority") and priority_url is not None) else 1.0
    df_out["priority_factor_global"] = priority_global
    df_out["final_score"] = df_out["base_score"] * df_out["priority_factor_url"] * df_out["priority_factor_global"]
    df_out = df_out.sort_values("final_score", ascending=False).reset_index(drop=True)

    st.subheader("Ergebnis")
    st.dataframe(df_out.head(100), use_container_width=True, hide_index=True)

    # Export
    st.markdown("### Export")
    csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ CSV herunterladen", data=csv_bytes, file_name="one_url_scoreboard.csv", mime="text/csv")

    try:
        import xlsxwriter  # optional
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="scores")
            for k, cols in debug_cols.items():
                if isinstance(cols, dict) and cols:
                    pd.DataFrame(cols).to_excel(writer, index=False, sheet_name=f"raw_{k}"[:31])
        st.download_button("⬇️ XLSX herunterladen", data=buf.getvalue(), file_name="one_url_scoreboard.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.caption("Hinweis: Für XLSX-Export kann das Paket `xlsxwriter` erforderlich sein.")

    config = {
        "scoring_mode": scoring_mode,
        "min_score": min_score if scoring_mode == "Rank (linear)" else None,
        "bucket_quantiles": [0.0, 0.5, 0.75, 0.9, 0.97, 1.0] if scoring_mode != "Rank (linear)" else None,
        "bucket_values": [0.0, 0.25, 0.5, 0.75, 1.0] if scoring_mode != "Rank (linear)" else None,
        "offtopic_tau": offtopic_tau,
        "priority_global": priority_global,
        "use_autodiscovery": use_autodiscovery,
        "weights": weights,
        "weights_norm": weights_norm,
        "active_criteria": [k for k in active.keys() if active[k]],
        "uploads_used": {k: name for k, (_, name) in st.session_state.uploads.items()},
        "column_maps": st.session_state.column_maps,
        "notes": "Masterliste: Union/Schnittmenge/Zwei Quellen. All-Inlinks in Union enthalten. SC Query→URL aggregiert. Embeddings robust. Compound-Spalten gesplittet.",
    }
    st.download_button("⬇️ Config (JSON)",
        data=json.dumps(config, indent=2).encode("utf-8"),
        file_name="one_url_scoreboard_config.json",
        mime="application/json",
    )
else:
    st.info("Bitte Masterliste erzeugen und mindestens ein Kriterium aktivieren & gewichten.")
