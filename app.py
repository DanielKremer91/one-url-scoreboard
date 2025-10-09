# app.py
# ONE URL Scoreboard — Streamlit App
# Idea: Daniel Kremer (ONE Beyond Search) — Implementation by ChatGPT
# Master-URL modes: Union (default), Own upload, Merge from up to two files, Pick one file, Intersection (all uploads)
# Criteria grouped by clusters; main_kw_exp supports direct value or SV×CTR(position)

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
**ONE URL Scoreboard** priorisiert URLs über wählbare Kriterien (globaler Scoring-Modus: *Rank linear* oder *Perzentil-Buckets*).  
Fehlende Daten im **aktiven** Kriterium ⇒ Score = 0 (kein Reweighting pro URL).

**Wichtig**
- **Master-URL-Liste**: Modi **Union** (Voreinstellung), **Eigene Masterliste**, **Merge aus bis zu zwei Dateien**, **Basis „eine Datei“**, **Schnittmenge (alle Uploads)**.
- **All-Inlinks-Datei** (Crawl/Kantenliste) wird **in die Union aufgenommen** (empfangende Ziel-URLs).
- **Search Console Query-Level** wird **automatisch** pro URL aggregiert.
- **Embeddings robust**: Fehlende Embeddings ⇒ Outlier (unter τ). Uneinheitliche Längen ⇒ Padding/Trunc auf dominante Dimension.
- **CSV-Fallback**: Falls eine Datei nur **eine** Spalte hat, wird automatisch nach `,` `;` `|` oder Tab gesplittet; **erste Zeile = Header**.
- **Kein globaler Prioritätsfaktor**. **Strategische Priorität** per URL bleibt optional.
""")

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

ALIASES = {
    # URL-Aliase erweitert: erkennt u. a. Page URL / Address
    "url": [
        "url","page","page_url","seite","address","adresse","target","ziel","ziel_url","landing_page"
    ],
    "clicks": ["clicks","klicks","traffic","besuche","sc_clicks"],
    "impressions": ["impressions","impr","impressionen","search_impressions"],
    "position": ["position","avg_position","ranking","ranking_position","average_position","durchschnittliche_position","durchschn._position","rank","avg_rank"],
    "search_volume": ["search_volume","sv","volume","suchvolumen","sv_monat"],
    "cpc": ["cpc","cost_per_click"],
    "traffic_value": ["traffic_value","otv","organic_traffic_value","value","potential_value","trafficwert","traffic_wert"],
    "potential_traffic_url": ["potential_traffic_url","potential_traffic","pot_traffic","potentielle_klicks","erwartete_klicks","erwarteter_traffic","estimated_traffic","estimated_clicks"],
    # extern
    "backlinks": ["backlinks","links_total","bl","inbound_links","links_to_target"],
    "ref_domains": ["ref_domains","referring_domains","rd","domains_ref","verweisende_domains","referring_domains"],
    # intern
    "unique_inlinks": ["unique_inlinks","internal_inlinks","inlinks_unique","eingehenden_links","inlinks","eingehende_links","inlinks_unique_count","incoming_links_unique"],
    # LLM referrals/crawl
    "llm_ref_traffic": ["llm_ref_traffic","llm_referrals","ai_referrals","llm_popularity","llm_traffic","sessions","sitzungen","visits","hits","traffic"],
    "llm_crawl_freq": ["llm_crawl_freq","ai_crawls","llm_crawls","llm_crawler_visits","crawls","visits","hits","requests"],
    "user_agent": ["user_agent","ua","agent","crawler","bot","useragent"],
    "embedding": ["embedding","embeddings","vector","vec","embedding_json"],
    "revenue": ["revenue","umsatz","organic_revenue","organic_umsatz","organic_sales"],
    "priority_factor": ["priority_factor","prio","priority","priorität","gewicht","gewichtung","boost","override","weight_override","wichtigkeit","boost_faktor","faktor","manual_weight"],
    "keyword": ["keyword","query","suchbegriff","suchanfrage"],
    # Hauptkeyword-Potenzial
    "main_keyword": ["main_keyword","hauptkeyword","primary_keyword","focus_keyword","focus_kw","haupt_kw","haupt-keyword"],
    "expected_clicks": ["expected_clicks","exp_clicks","expected_clicks_main","expected_clicks_kw","erwartete_klicks","erw_klicks"],
}

TRACKING_PARAMS_PREFIXES = ["utm_", "icid_"]
TRACKING_PARAMS_EXACT = {"gclid","fbclid","msclkid","mc_eid","yclid"}

# ---- AI/LLM Bot Muster (inkludieren / exkludieren) ----
EXCLUDE_CLASSIC_BOTS = [
    "googlebot", "googlebot smartphone", "bingbot", "yandex", "baidu"
]
INCLUDE_AI_BOTS = [
    "gptbot", "openai", "anthropic", "claudebot", "perplexitybot",
    "perplexity", "cohere", "ccbot", "bytespider", "google-extended",
    "meta-ai", "facebook ai", "chatgpt", "duckassist", "quorabots", "youbot"
]
GENERIC_BOT_TOKENS = [
    "bot", "ai", "gpt", "claude", "perplexity", "cohere", "bytespider", "ccbot",
    "facebook ai", "google-extended", "openai", "anthropic", "metabot", "meta-ai", "youbot", "duckassist"
]

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
    """
    Robust CSV/Excel reader:
    - Für CSV: Wenn nur eine Spalte vorhanden ist, wird automatisch nach , ; | oder Tab gesplittet.
    - Erste Zeile wird IMMER als Header interpretiert.
    - Für Excel bleibt alles unverändert.
    """
    data = uploaded.read()
    name = (uploaded.name or "").lower()

    def _as_bytesio():
        return io.BytesIO(data)

    if name.endswith(".csv"):
        # 1) Normales Einlesen
        try:
            df = pd.read_csv(_as_bytesio(), low_memory=False)
        except Exception:
            try:
                df = pd.read_csv(_as_bytesio(), sep=None, engine="python", low_memory=False)
            except Exception:
                df = pd.DataFrame()
        # 2) Single-column → rate Delimiter
        if df.shape[1] == 1:
            for sep in [";", ",", "\t", "|"]:
                try:
                    df2 = pd.read_csv(_as_bytesio(), sep=sep, engine="python", low_memory=False)
                    if df2.shape[1] > 1:
                        df = df2
                        break
                except Exception:
                    pass
        # 3) Immer noch single-column → manuell splitten
        if df.shape[1] == 1:
            col = df.columns[0]
            s = df[col].astype(str)
            counts = {";": s.str.count(";").sum(), ",": s.str.count(",").sum(), "\t": s.str.count("\t").sum(), "|": s.str.count("|").sum()}
            delim = max(counts, key=counts.get) if any(v > 0 for v in counts.values()) else ","
            parts = s.str.split(delim, expand=True)
            if parts.shape[0] > 1:
                header = parts.iloc[0].fillna("").astype(str).tolist()
                parts = parts[1:].reset_index(drop=True)
                parts.columns = [normalize_header(h) for h in header]
            else:
                parts.columns = [f"col_{i+1}" for i in range(parts.shape[1])]
            df = parts
        df = normalize_headers(df)
        return df

    # Excel
    try:
        df = pd.read_excel(io.BytesIO(data))
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(data), low_memory=False)
        except Exception:
            df = pd.DataFrame()
    df = normalize_headers(df)
    return df

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
    if parts.shape[1] >= 2: df2["ref_domains"] = pd.to_numeric(parts[1], errors="coerce")
    if parts.shape[1] >= 3: df2["backlinks"] = pd.to_numeric(parts[2], errors="coerce")
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
    cols_norm = {re.sub(r"[_\s]+","", c): c for c in df.columns}
    for c in candidates:
        cn = re.sub(r"[_\s]+","", c)
        if cn in cols_norm: return cols_norm[cn]
    # zusätzliche Heuristik für URL
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

use_autodiscovery = st.sidebar.toggle(
    "Autodiscovery über alle Uploads",
    value=True,
    help="Wenn aktiv: Fehlen die vorgesehenen Dateien/Spalten, sucht das Tool automatisch in anderen Uploads nach passenden Spalten (per Alias)."
)

# ============= Kriterienauswahl (CLUSTERED) =============
st.subheader("Kriterien auswählen")
st.caption("Wähle unten die gewünschten Kriterien. Danach erscheinen die passenden Upload-Masken.")

CRITERIA_GROUPS = {
    "Performance & Nachfrage": [
        ("sc_clicks", "Search Console Klicks",
         "Search Console Klicks – wie viele Klicks die URL geholt hat. Je mehr, desto besser."),
        ("sc_impr", "Search Console Impressions",
         "Search Console Impressions – wie oft die URL in den SERPs eingeblendet wurde. Je mehr, desto besser."),
        ("seo_eff", "URL-SEO-Effizienz",
         "Anteil der Keywords einer URL mit durchschnittlicher Position ≤ 5. Je höher, desto effizienter."),
        ("main_kw_sv", "Hauptkeyword-Potenzial (Suchvolumen)",
         "URL + Hauptkeyword + monatliches Suchvolumen des Hauptkeywords. Je höher, desto besser."),
        ("main_kw_exp", "Hauptkeyword-Potenzial (Expected Clicks)",
         "Entweder fertiger Wert **expected_clicks** ODER Berechnung aus (Suchvolumen × CTR(Position)). Je mehr, desto besser."),
    ],
    "Popularität & Autorität": [
        ("ext_pop", "URL-Popularität extern",
         "Backlinks & Referring Domains: 70% Ref. Domains + 30% Backlinks. Je höher, desto besser."),
        ("int_pop", "URL-Popularität intern",
         "Eindeutige interne Inlinks pro URL, entweder direkt `unique_inlinks` oder aus Kantenliste aggregiert."),
        ("llm_ref", "LLM-Popularität (Referrals)",
         "AI/LLM-generierte Referrals/Sessions. Je höher, desto besser."),
        ("llm_crawl", "LLM-Crawl-Frequenz",
         "Besuche durch AI/LLM-Crawler. Klassische Bots (Googlebot, Bingbot, Yandex, Baidu) sind exkludiert."),
    ],
    "Wirtschaftlicher Impact": [
        ("otv", "Organic Traffic Value",
         "Geschätzter organischer Traffic-Wert der URL (direkt vorhanden oder aus Keyword-Datei via CTR-Kurve)."),
        ("revenue", "Umsatz",
         "Tatsächlich erzielter Umsatz der URL."),
    ],
    "Qualität & Relevanz": [
        ("offtopic", "Offtopic-Score (0/1)",
         "Semantische Nähe zum Themen-Centroid (Cosine-Similarity). Threshold τ: ≥ τ = 1, sonst 0. Fehlende Embeddings gelten als < τ."),
    ],
    "Strategische Steuerung": [
        ("priority", "Strategische Priorität (Override)",
         "Manueller Multiplikator pro URL, skaliert den finalen Score nur für diese URL. Standardmäßig hat jede URL die Prio 1. Wenn hier in der Input-Datei eine URL die Priorität 1,2 bekommt, bekommt sie +20% Verstärkung für den finalen Score. Eine URL mit einer Prio von 2 wird doppelt gewichtet usw. In die Input-Datei müssen nur die URLs eingetragen werden, die geboostet werden soll, inkl. Prioritätsfaktor."),
    ],
}

active: Dict[str, bool] = {}
for group, crits in CRITERIA_GROUPS.items():
    st.markdown(f"### {group}")
    for code, label, helptext in crits:
        active[code] = st.checkbox(label, value=False, help=helptext, key=f"chk_{code}")

# ============= Upload-Masken (nach Auswahl) =============
st.markdown("---")
st.subheader("Basierend auf den gewählten Kriterien benötigen wir folgende Dateien")

# Search Console
if active.get("sc_clicks") or active.get("sc_impr"):
    st.markdown("**Search Console — erwartet:** `URL` (Alias: url/page/page_url/address), `Clicks/Klicks`, `Impressions/Impressionen`. **Query-Ebene ist ok** – wird pro URL aggregiert.")
    store_upload("sc", st.file_uploader("Search Console Datei (CSV/XLSX)", type=["csv","xlsx"], key="upl_sc"))

# OTV
if active.get("otv"):
    st.markdown("**Organic Traffic Value — erwartet:**")
    st.markdown("- **Variante A (URL-Value):** `URL`, `traffic_value` **oder** `potential_traffic_url` (+ optional `cpc`).")
    st.markdown("- **Variante B (Keyword-basiert):** `keyword`, `URL`, `position`, `search_volume` (+ optional `cpc`). CTR-Kurve optional (`position`, `ctr`).")
    c1, c2, c3 = st.columns(3)
    with c1: store_upload("otv_url", st.file_uploader("OTV: URL-Value (optional)", type=["csv","xlsx"], key="upl_otv_url"))
    with c2: store_upload("otv_kw",  st.file_uploader("OTV: Keyword-Datei (optional)", type=["csv","xlsx"], key="upl_otv_kw"))
    with c3: store_upload("ctr_curve", st.file_uploader("CTR-Kurve (optional, genutzt auch für Expected Clicks)", type=["csv","xlsx"], key="upl_ctr"))

# Extern-Popularität
if active.get("ext_pop"):
    st.markdown("**URL-Popularität extern — erwartet:** `URL`, `backlinks` (Alias inkl. `links_to_target`), `ref_domains` (Alias inkl. `referring_domains`).")
    store_upload("ext", st.file_uploader("Extern-Popularität (CSV/XLSX)", type=["csv","xlsx"], key="upl_ext"))

# Intern-Popularität
if active.get("int_pop"):
    st.markdown("**URL-Popularität intern — erwartet:**")
    st.markdown("- **Variante A:** `URL`, `unique_inlinks`.")
    st.markdown("- **Variante B (Kantenliste):** `URL` (= Ziel) **und** eine Quellspalte (z. B. `source`, `source_url`, `referrer`) – wird zu `unique_inlinks` aggregiert.")
    store_upload("int", st.file_uploader("Intern-Popularität (Crawl/Inlinks)", type=["csv","xlsx"], key="upl_int"))

# LLM Referral
if active.get("llm_ref"):
    st.markdown("**LLM Referrals — erwartet:** **genau zwei Spalten**: `URL`, `Sitzungen / LLM-Traffic` (Alias: sessions/sitzungen/visits/hits/traffic).")
    store_upload("llmref", st.file_uploader("LLM-Referrals (CSV/XLSX)", type=["csv","xlsx"], key="upl_llmref"))

# === LLM Crawl (aggregiert ODER Logfile) — Upload + Bot-Auswahl-UI ===
if active.get("llm_crawl"):
    st.markdown("**LLM Crawler Frequenz — erwartet:**")
    st.markdown("- **Variante A (aggregiert):** `URL` + eine oder mehrere Spalten mit Bot-Besuchen (z. B. `GPTBot`, `ClaudeBot`, `PerplexityBot`, `OAI-SearchBot`, …).")
    st.markdown("- **Variante B (Logfile):** `URL`, `user_agent` (+ optional `sessions/visits/hits/requests`). Klassische Bots (Googlebot, Bingbot, Yandex, Baidu) sind exkludiert.")
    store_upload("llmcrawl", st.file_uploader("LLM-Crawl (CSV/XLSX)", type=["csv", "xlsx"], key="upl_llmcrawl"))

    # State für Modus & Auswahl initialisieren
    st.session_state.setdefault("llm_crawl_mode", None)
    st.session_state.setdefault("llm_bot_column_choices", [])

    if "llmcrawl" in st.session_state.uploads:
        df_llm, _ = st.session_state.uploads["llmcrawl"]
        cols = list(df_llm.columns)
        url_col = find_first_alias(df_llm, "url")
        ua_col  = find_first_alias(df_llm, "user_agent")

        # Modus bestimmen: wenn 'user_agent' existiert => Logfile, sonst Aggregat
        mode = "log" if ua_col else "aggregated"
        st.session_state["llm_crawl_mode"] = mode

        if url_col is None:
            st.error("Konnte keine URL-Spalte erkennen. Bitte prüfe die Datei (Header `URL`).")
        else:
            if mode == "aggregated":
                # Spalten-Kandidaten ermitteln: numerisch, keine klassischen Summen-/Meta-/Bot-Gesamtspalten
                classic_bot_cols = {
                    "googlebot", "googlebot_smartphone", "bingbot", "yandex", "baidu",
                }
                ignore_cols_exact = {"alle_bots", "gesamt", "total", "summe", "sum", "events", "anzahl_ereignisse"}
                ignore_like_tokens = {"co2", "antwortzeit", "response", "ms"}

                def is_numeric_series(s: pd.Series) -> bool:
                    try:
                        pd.to_numeric(s, errors="coerce")
                        return True
                    except Exception:
                        return False

                cand_cols = []
                for c in cols:
                    if c == url_col:
                        continue
                    cl = c.lower()
                    if cl in ignore_cols_exact:
                        continue
                    if any(tok in cl for tok in ignore_like_tokens):
                        continue
                    if cl in classic_bot_cols:
                        continue
                    # nur numerische Spalten zulassen
                    if is_numeric_series(df_llm[c]):
                        cand_cols.append(c)

                # sinnvolle Default-Auswahl: Spalten, die wie AI/LLM-Botnamen aussehen
                ai_pref_tokens = [
                    "gpt", "openai", "oai", "oai-searchbot", "claude", "anthropic",
                    "perplexity", "perplexitybot", "bytespider", "ccbot", "cohere",
                    "meta-ai", "facebook", "youbot", "duckassist", "kagi"
                ]
                default_ai = [c for c in cand_cols if any(tok in c.lower() for tok in ai_pref_tokens)]

                st.info("Wähle, **welche Bot-Spalten** in die Berechnung einfließen sollen. "
                        "Klassische Bots (Googlebot, Bingbot, Yandex, Baidu) werden ignoriert.")
                st.session_state["llm_bot_column_choices"] = st.multiselect(
                    "Bot-Spalten auswählen",
                    options=cand_cols,
                    default=default_ai or cand_cols,
                    help="Nur die hier ausgewählten Spalten werden je URL aufsummiert."
                )

                # kleine Vorschau
                if st.session_state["llm_bot_column_choices"]:
                    tmp = df_llm.copy()
                    tmp = ensure_url_column(tmp, url_col)
                    for c in st.session_state["llm_bot_column_choices"]:
                        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)
                    tmp["_llm_sum_preview"] = tmp[st.session_state["llm_bot_column_choices"]].sum(axis=1)
                    prev = tmp[[url_col, "_llm_sum_preview"]].groupby(url_col, as_index=False)["_llm_sum_preview"].sum().rename(
                        columns={url_col: "URL", "_llm_sum_preview": "LLM-Crawls (Auswahl)"}
                    )
                    st.dataframe(prev.head(10), use_container_width=True)
                else:
                    st.warning("Keine Bot-Spalten ausgewählt. Es werden 0 Besuche gezählt.")

            else:
                st.info("Logfile erkannt: Bitte unten die Bots wählen (User-Agent enthält …). "
                        "Klassische Crawler (Googlebot, Bingbot, Yandex, Baidu) werden ausgeschlossen.")


# Embeddings
if active.get("offtopic"):
    st.markdown("**Embeddings — erwartet:** `URL` (Alias inkl. `Address`), `embedding` (JSON-Liste oder Zahlen-Sequenz). Fehlende Embeddings ⇒ Outlier (< τ).")
    store_upload("emb", st.file_uploader("Embeddings-Datei (CSV/XLSX)", type=["csv","xlsx"], key="upl_emb"))

# Revenue
if active.get("revenue"):
    st.markdown("**Umsatz — erwartet:** `URL`, `revenue`.")
    store_upload("rev", st.file_uploader("Umsatz-Datei (CSV/XLSX)", type=["csv","xlsx"], key="upl_rev"))

# SEO Efficiency
if active.get("seo_eff"):
    st.markdown("**URL-SEO-Effizienz — erwartet:** `keyword`, `URL`, `position` (Top-5-Anteil je URL wird berechnet).")
    store_upload("eff_kw", st.file_uploader("Keyword-Datei (SEO-Effizienz)", type=["csv","xlsx"], key="upl_eff_kw"))

# Priority
if active.get("priority"):
    st.markdown("**Strategische Priorität — erwartet (optional):** `URL`, `priority_factor` (0.5–2.0).")
    store_upload("prio", st.file_uploader("Priorität (optional)", type=["csv","xlsx"], key="upl_prio"))

# Hauptkeyword (für beide Teil-Kriterien)
if active.get("main_kw_exp") or active.get("main_kw_sv"):
    st.markdown("**Hauptkeyword-Potenzial — erwartet:** `URL`, `main_keyword` **und** je nach Kriterium:")
    st.markdown("- Für **Expected Clicks**: Spalte `expected_clicks` **oder** `search_volume` + `position` (wir berechnen dann `expected_clicks = SV × CTR(Position)`).")
    st.markdown("- Für **Suchvolumen**: Spalte `search_volume`.")
    store_upload("main_kw", st.file_uploader("Hauptkeyword-Mapping (CSV/XLSX)", type=["csv","xlsx"], key="upl_main_kw"))

# ---------- (Re)build schema index after uploads changed ----------
build_schema_index()

# ============= Master URL list builder =============
st.subheader("Master-URL-Liste")
st.markdown(
    """
Wähle, wie die Masterliste gebildet wird. **Union** ist Standard.

Die **Master-URL-Liste** ist die zentrale Ausgangsliste aller URLs, die das Scoreboard überhaupt bewertet.  
Alle Kriterien (Clicks, Backlinks, Umsatz, etc.) werden **an diese Liste joined**.  
Nur URLs, die in der Masterliste stehen, erhalten am Ende einen Score und erscheinen im Ergebnis/Export.
"""
)

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
    help=(
        "**Union:** Alle URLs aus allen Uploads (Duplikate entfernt). "
        "**Eigene Masterliste:** externe Datei mit URL-Spalte. "
        "**Merge aus bis zu zwei Dateien:** Masterliste aus 1–2 ausgewählten Uploads. "
        "**Aus einer bestimmten Datei wählen:** Nur URLs aus genau einem Upload. "
        "**Schnittmenge:** Nur URLs, die in *allen* Uploads mit URL-Spalte vorkommen."
    )
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
    url_sets = []
    for key, (df, _) in st.session_state.uploads.items():
        c = find_first_alias(df, "url")
        if not c:
            continue
        d = ensure_url_column(df[[c]], c)
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

# Ergebnis der Masterliste anzeigen
if master_urls is None or master_urls.empty:
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

# --- Search Console (Clicks / Impressions) — Query-Level Aggregation ---
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
        for m in metrics: df_sc[m] = pd.to_numeric(df_sc[m], errors="coerce").fillna(0)
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

# === LLM Crawl (Berechnung) ===
def _contains_any(s: str, needles: List[str]) -> bool:
    s = s or ""
    l = s.lower()
    return any(n.lower() in l for n in needles if n)

if active.get("llm_crawl"):
    found = None

    # Aggregiertes Format (mehrere Bot-Spalten, z. B. GPTBot / ClaudeBot / PerplexityBot / OAI-SearchBot …)
    if "llmcrawl" in st.session_state.uploads and st.session_state.get("llm_crawl_mode") == "aggregated":
        df_aggr, _ = st.session_state.uploads["llmcrawl"]
        urlc = find_first_alias(df_aggr, "url")
        sel_cols = st.session_state.get("llm_bot_column_choices", [])
        if urlc and sel_cols and master_urls is not None:
            df_aggr = ensure_url_column(df_aggr, urlc).copy()
            for c in sel_cols:
                df_aggr[c] = pd.to_numeric(df_aggr[c], errors="coerce").fillna(0)
            df_aggr["_llm_crawl_freq"] = df_aggr[sel_cols].sum(axis=1)
            agg = df_aggr.groupby(urlc, as_index=False)["_llm_crawl_freq"].sum().rename(
                columns={"_llm_crawl_freq": "llm_crawl_freq"}
            )
            found = ("llmcrawl_aggr", agg, {"url": urlc, "llm_crawl_freq": "llm_crawl_freq"})
        else:
            found = None

    # Fallback: Logfile-Format mit User-Agent
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

                INCLUDE_AI_BOTS = [
                    "gptbot", "openai", "oai", "oai-searchbot", "anthropic", "claude",
                    "perplexity", "perplexitybot", "cohere", "ccbot", "bytespider",
                    "meta-ai", "facebook ai", "youbot", "duckassist", "kagi"
                ]
                EXCLUDE_CLASSIC_BOTS = ["googlebot", "googlebot smartphone", "bingbot", "yandex", "baidu"]

                include_needles = include_from_ui + custom_inc
                if not include_needles:
                    include_needles = INCLUDE_AI_BOTS[:]
                exclude_needles = list(set((exclude_from_ui + custom_exc) + EXCLUDE_CLASSIC_BOTS))

                mask_inc = df_log[uac].astype(str).map(lambda s: _contains_any(s, include_needles))
                mask_exc = df_log[uac].astype(str).map(lambda s: _contains_any(s, exclude_needles))
                mask = mask_inc & (~mask_exc)

                df_sel = df_log.loc[mask, [urlc] + ([sess_col] if has_amount else [])].copy()
                if not df_sel.empty:
                    if has_amount:
                        df_sel[sess_col] = pd.to_numeric(df_sel[sess_col], errors="coerce").fillna(0)
                        agg = df_sel.groupby(urlc, as_index=False)[sess_col].sum().rename(columns={sess_col: "llm_crawl_freq"})
                    else:
                        agg = df_sel.groupby(urlc, as_index=False).size().rename(columns={"size": "llm_crawl_freq"})
                    found = ("llmcrawl_log", agg, {"url": urlc, "llm_crawl_freq": "llm_crawl_freq"})
                else:
                    found = None

    # Join + Score
    if found and master_urls is not None:
        _, df_lc, cm = found
        d = join_on_master(df_lc, cm["url"], [cm["llm_crawl_freq"]])
        results["llm_crawl"] = mode_score(d[cm["llm_crawl_freq"]]).fillna(0.0)
        debug_cols["llm_crawl"] = {"llm_crawl_freq_raw": pd.to_numeric(d[cm["llm_crawl_freq"]], errors="coerce")}
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
                if v is None: return None
                n = len(v)
                if n == L: return v
                if n > L:  return v[:L]
                vv = np.zeros(L, dtype=float); vv[:n] = v; return vv

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
                sim = d["_sim"].fillna(-1.0)  # fehlend ⇒ sicher < τ
                results["offtopic"] = pd.Series((sim >= st.session_state.get("offtopic_tau", 0.5)).astype(float))
                debug_cols["offtopic"] = {"similarity": sim}
    elif master_urls is not None:
        results["offtopic"] = pd.Series(0.0, index=master_urls.index)

# τ-Slider für Offtopic (nach Berechnung setzen/überschreiben)
if active.get("offtopic") and "offtopic" in debug_cols and "similarity" in debug_cols["offtopic"]:
    tau = st.sidebar.slider(
        "Offtopic-Threshold τ (Ähnlichkeit)",
        0.0, 1.0, 0.5, 0.01,
        help="Binäres Gate im Offtopic-Kriterium: Cosine-Similarity ≥ τ ⇒ 1.0, sonst 0.0. Fehlende Embeddings zählen als < τ."
    )
    st.session_state["offtopic_tau"] = tau
    sim = debug_cols["offtopic"]["similarity"]
    results["offtopic"] = pd.Series((sim >= tau).astype(float))

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
        priority_url = (
            pd.to_numeric(d[cm["priority_factor"]], errors="coerce")
            .fillna(1.0)
            .clip(lower=1.0)  # alles unter 1.0 wird auf 1.0 gesetzt
        )
      
    elif master_urls is not None:
        priority_url = pd.Series(1.0, index=master_urls.index)

# --- Hauptkeyword-Potenzial (Expected Clicks) ---
if active.get("main_kw_exp"):
    found_any = find_df_with_targets(["url","main_keyword"], prefer_keys=["main_kw"], use_autodiscovery=use_autodiscovery)
    if found_any and master_urls is not None:
        _, df_m, cm = found_any
        urlc = cm["url"]
        # Variante 1: fertige expected_clicks Spalte
        expc = find_first_alias(df_m, "expected_clicks")
        if expc:
            j = join_on_master(df_m, urlc, [cm["main_keyword"], expc])
            vals = pd.to_numeric(j[expc], errors="coerce")
            results["main_kw_exp"] = mode_score(vals).fillna(0.0)
            debug_cols["main_kw_exp"] = {"main_keyword": j[cm["main_keyword"]], "expected_clicks_raw": j[expc]}
        else:
            # Variante 2: compute expected_clicks = SV × CTR(position)
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
                exp_calc = pd.to_numeric(df_m[svc], errors="coerce").fillna(0) * ctrs
                agg = df_m.assign(_exp=exp_calc).groupby(urlc, as_index=False)["_exp"].sum()
                j = master_urls.merge(agg, left_on="url_norm", right_on=urlc, how="left")
                vals = j["_exp"]
                results["main_kw_exp"] = mode_score(vals).fillna(0.0)
                debug_cols["main_kw_exp"] = {"expected_clicks_raw": vals}
            else:
                results["main_kw_exp"] = pd.Series(0.0, index=master_urls.index)
    elif master_urls is not None:
        results["main_kw_exp"] = pd.Series(0.0, index=master_urls.index)

# --- Hauptkeyword-Potenzial (Suchvolumen) ---
if active.get("main_kw_sv"):
    found_any = find_df_with_targets(["url","main_keyword"], prefer_keys=["main_kw"], use_autodiscovery=use_autodiscovery)
    if found_any and master_urls is not None:
        _, df_m, cm = found_any
        urlc = cm["url"]
        svc = find_first_alias(df_m, "search_volume")
        if svc:
            j = join_on_master(df_m, urlc, [cm["main_keyword"], svc])
            vals = pd.to_numeric(j[svc], errors="coerce")
            results["main_kw_sv"] = mode_score(vals).fillna(0.0)
            debug_cols["main_kw_sv"] = {"main_keyword": j[cm["main_keyword"]], "search_volume_raw": j[svc]}
        else:
            results["main_kw_sv"] = pd.Series(0.0, index=master_urls.index)
    elif master_urls is not None:
        results["main_kw_sv"] = pd.Series(0.0, index=master_urls.index)

# ============= Gewichte & Aggregation =============
st.subheader("Gewichtung der aktiven Kriterien")
weight_keys = [k for k in [
    "sc_clicks","sc_impr","seo_eff","main_kw_sv","main_kw_exp",
    "ext_pop","int_pop","llm_ref","llm_crawl",
    "otv","revenue",
    "offtopic",
] if active.get(k)]
# priority ist ein Multiplikator, kein gewichtetes Kriterium → separate Behandlung

weights: Dict[str, float] = {}
if weight_keys:
    cols = st.columns(len(weight_keys))
    for i, k in enumerate(weight_keys):
        # Label lookup aus Gruppen
        label = None
        for _, crits in CRITERIA_GROUPS.items():
            for code, l, _ in crits:
                if code == k:
                    label = l
                    break
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

    # Strategische Priorität (per-URL) — optional
    if active.get("priority"):
        # priority_url wurde oben gesetzt, wenn Datei vorhanden war – sonst 1.0
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
        "notes": "Masterliste: Union / Eigene / Merge(1-2) / Eine Datei / Schnittmenge. All-Inlinks in Union enthalten. SC Query→URL aggregiert. Embeddings robust. CSV-Repair aktiv (1-Spalten-Files). LLM-Crawl: wahlweise manuelle Bot-Auswahl oder Heuristik. main_kw_exp kann direkt oder via SV×CTR berechnet werden.",
    }
    st.download_button("⬇️ Config (JSON)",
        data=json.dumps(config, indent=2).encode("utf-8"),
        file_name="one_url_scoreboard_config.json",
        mime="application/json",
    )
else:
    st.info("Bitte Masterliste erzeugen und mindestens ein Kriterium aktivieren & gewichten.")
