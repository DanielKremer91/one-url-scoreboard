import io
import json
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# 1) Seite breit + Sidebar standardmäßig ausgeklappt
st.set_page_config(
    page_title="ONE URL Scoreboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2) CSS: Haupt-Container auf volle Breite ziehen, Padding schlank,
#    Sidebar etwas schmaler (optional)
st.markdown("""
<style>
.main .block-container {
  max-width: 100% !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
  padding-top: 1rem !important;
  padding-bottom: 2rem !important;
}
[data-testid="stSidebar"] {
  min-width: 260px !important;
  max-width: 260px !important;
}
.reportview-container .main .block-container {
  max-width: 100% !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
  padding-top: 1rem !important;
  padding-bottom: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Branding
# =============================
try:
    st.image(
        "https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png",
        width=250,
    )
except Exception:
    pass

st.title("ONE URL Scoreboard")

st.markdown(
    """
<div style="background-color: #f2f2f2; color: #000000; padding: 15px 20px; border-radius: 6px; font-size: 0.9em; max-width: 850px; margin-bottom: 1.5em; line-height: 1.5;">
  Entwickelt von <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">Daniel Kremer</a> von <a href="https://onebeyondsearch.com/" target="_blank">ONE Beyond Search</a> &nbsp;|&nbsp;
  Folge mir auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a> für mehr SEO-Insights und Tool-Updates
</div>
<hr>
""",
    unsafe_allow_html=True,
)

# =============================
# Hilfe / Tool-Dokumentation (Expander)
# =============================
with st.expander("❓ Hilfe / Tool-Dokumentation", expanded=False):
    st.markdown(
        """
## Was macht das Tool „ONE URL Scoreboard“?

**ONE URL Scoreboard** hilft dir, URLs anhand mehrerer optionaler Kriterien zu **bewerten und zu priorisieren**.  
Du aktivierst die gewünschten Kriterien (On/Off), lädst – falls vorhanden – die passenden Dateien hoch und vergibst Gewichte.  
Das Tool berechnet **Teil-Scores** pro Kriterium (global einstellbar: *Rank-basiert* oder *Perzentil-Buckets*) und erzeugt einen **Gesamtscore** pro URL.  
Fehlende Daten für ein **aktives** Kriterium ⇒ **Score = 0** (keine Re-Gewichtung pro URL).

**Master-URL-Liste (Default: Union):**  
Standardmäßig wird die Masterliste aus der **Vereinigungsmenge** aller hochgeladenen URLs gebildet.  
Optional kannst du:
- eine **eigene Masterliste** hochladen oder
- die Masterliste aus **bis zu zwei** bereits hochgeladenen Dateien zusammensetzen.

**URL-Normalisierung (streng nach deinen Regeln):**
- `#fragment` wird **abgeschnitten**
- **Trailing Slash bleibt** ( `/pfad` ≠ `/pfad/` )
- **Tracking-Parameter** (z. B. `utm_*`, `gclid`, `fbclid`, `msclkid`) werden entfernt
- **lowercase** für die URL
- **http** wird **nicht** zu **https** zusammengeführt

**Scoring-Modi (global):**
- **Rank-basiert (Default)**
- **Perzentil-Buckets**

**Kriterien:**
SC Clicks / SC Impressions • Organic Traffic Value • externe & interne Popularität • LLM-Referral • LLM-Crawl-Frequenz • Offtopic (0/1) • Umsatz • SEO-Effizienz (Top-5-Anteil) • Strategische Priorität
""",
        unsafe_allow_html=False,
    )

# =============================
# Session State & Helpers
# =============================
if "uploads" not in st.session_state:
    st.session_state.uploads = {}  # key -> (df, name)

if "column_maps" not in st.session_state:
    st.session_state.column_maps = {}  # key -> {target: source_col}

# Global aliases (normalized header names)
ALIASES = {
    "url": ["url","page","seite","address","adresse","target","ziel","ziel_url","landing_page"],
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
}

TRACKING_PARAMS_PREFIXES = ["utm_", "icid_"]
TRACKING_PARAMS_EXACT = {"gclid", "fbclid", "msclkid", "mc_eid", "yclid"}

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
    if not qs:
        return ""
    kept = []
    for pair in qs.split("&"):
        if "=" in pair:
            k, v = pair.split("=", 1)
        else:
            k, v = pair, ""
        kl = k.lower()
        if any(kl.startswith(p) for p in TRACKING_PARAMS_PREFIXES):
            continue
        if kl in TRACKING_PARAMS_EXACT:
            continue
        kept.append(pair)
    return "&".join(kept)

def normalize_url(u: str) -> Optional[str]:
    if not isinstance(u, str) or not u.strip():
        return None
    s = u.strip().lower()
    if "#" in s:
        s = s.split("#", 1)[0]
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

def store_upload(key: str, file):
    if file is None:
        return
    df = read_table(file)
    st.session_state.uploads[key] = (df, file.name)

def find_first_alias(df: pd.DataFrame, target: str) -> Optional[str]:
    candidates = ALIASES.get(target, [])
    for c in candidates:
        if c in df.columns:
            return c
    cols_norm = {re.sub(r"[_\s]+", "", c): c for c in df.columns}
    for c in candidates:
        cn = re.sub(r"[_\s]+", "", c)
        if cn in cols_norm:
            return cols_norm[cn]
    return None

def require_columns_ui(key: str, df: pd.DataFrame, targets: List[str], label: str) -> Dict[str, str]:
    colmap = {}
    for t in targets:
        hit = find_first_alias(df, t)
        if hit:
            colmap[t] = hit
    missing = [t for t in targets if t not in colmap]
    if missing:
        st.warning(f"**{label}**: Es konnten nicht alle Spalten automatisch erkannt werden. Bitte zuordnen.")
        with st.expander(f"Spalten-Mapping für {label}", expanded=True):
            for t in targets:
                options = [None] + list(df.columns)
                default_idx = 0
                if t in colmap:
                    default_idx = options.index(colmap[t])
                sel = st.selectbox(f"{t} →", options, index=default_idx, key=f"map_{key}_{t}")
                if sel:
                    colmap[t] = sel
    st.session_state.column_maps[key] = colmap
    return colmap

def ensure_url_column(df: pd.DataFrame, url_col: str) -> pd.DataFrame:
    df = df.copy()
    df[url_col] = df[url_col].map(normalize_url)
    df = df[df[url_col].notna()]
    return df

# =============================
# Scoring functions
# =============================
def rank_scores(series: pd.Series, min_score: float = 0.2) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(np.nan).clip(lower=0)
    mask = s.notna()
    ranks = s[mask].rank(method="average", ascending=False)  # 1=best
    n = len(ranks)
    if n <= 1:
        res = pd.Series(0.0, index=s.index)
        res[mask] = 1.0
        return res
    res = pd.Series(0.0, index=s.index)
    res[mask] = 1.0 - (ranks - 1) / (n - 1) * (1.0 - min_score)
    return res

def bucket_scores(series: pd.Series,
                  quantiles: List[float] = [0.0, 0.5, 0.75, 0.9, 0.97, 1.0],
                  bucket_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").clip(lower=0)
    mask = s.notna()
    res = pd.Series(0.0, index=s.index)
    if mask.sum() == 0:
        return res
    try:
        qvals = s[mask].quantile(quantiles).values
        bins = np.unique(qvals)
        if len(bins) < 3:
            mn, mx = float(s[mask].min()), float(s[mask].max())
            if mx <= mn:
                res[mask] = bucket_values[-1]
                return res
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

# =============================
# CTR curve (preset) for OTV
# =============================
def default_ctr_curve() -> pd.DataFrame:
    data = {
        "position": list(range(1, 21)),
        "ctr": [0.30,0.15,0.10,0.07,0.05,0.04,0.035,0.03,0.025,0.02,0.018,0.016,0.014,0.012,0.010,0.009,0.008,0.007,0.006,0.005],
    }
    return pd.DataFrame(data)

def get_ctr_for_pos(pos: float, ctr_df: pd.DataFrame) -> float:
    try:
        p = int(round(float(pos)))
    except Exception:
        return 0.0
    p = max(1, min(p, int(ctr_df["position"].max())))
    row = ctr_df.loc[ctr_df["position"] == p]
    if row.empty:
        return 0.0
    return float(row["ctr"].values[0])

# =============================
# Sidebar — Global settings (bleiben in der Sidebar)
# =============================
st.sidebar.header("⚙️ Einstellungen")

scoring_mode = st.sidebar.radio(
    "Scoring-Modus (global)",
    ["Rank (linear)", "Perzentil-Buckets"],
    index=0,
    help="Globaler Modus für alle Kriterien.",
)

min_score = 0.2
bucket_quantiles = [0.0, 0.5, 0.75, 0.9, 0.97, 1.0]
bucket_values = [0.0, 0.25, 0.5, 0.75, 1.0]

if scoring_mode == "Rank (linear)":
    min_score = st.sidebar.slider(
        "Min-Score für schlechteste (vorhandene) URL",
        min_value=0.0, max_value=0.5, value=0.2, step=0.05,
        help="URLs ohne Daten im Kriterium bekommen immer 0.0.",
    )
else:
    with st.sidebar.expander("Bucket-Setup", expanded=False):
        st.write("Standard: Quantile [0, .5, .75, .9, .97, 1.0] → Scores [0, .25, .5, .75, 1.0]")

offtopic_tau = st.sidebar.slider(
    "Offtopic-Threshold τ (Ähnlichkeit)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    help="Unter τ = 0, ab τ = 1 (binär)."
)

priority_global = st.sidebar.slider(
    "Globaler Prioritäts-Faktor",
    min_value=0.5, max_value=2.0, value=1.0, step=0.05,
)

# =============================
# Kriterienauswahl im Hauptteil (rechts)
# =============================
CRITERIA = [
    ("sc_clicks", "SC Clicks", "Anzahl Clicks (Search Console Organic Performance)."),
    ("sc_impr", "SC Impressions", "Anzahl Impressions (Search Console Organic Performance)."),
    ("otv", "Organic Traffic Value", "URL-Value vorhanden ODER aus Keyword-Datei via CTR-Kurve berechnet."),
    ("ext_pop", "URL-Popularität extern", "Ref. Domains (70 %) & Backlinks (30 %)."),
    ("int_pop", "URL-Popularität intern", "Unique interne Inlinks."),
    ("llm_ref", "LLM-Popularität (Referrals)", "Referral-Traffic aus LLM/AI."),
    ("llm_crawl", "LLM-Crawl-Frequenz", "Anzahl AI/LLM-Crawler-Besuche."),
    ("offtopic", "Offtopic-Score (0/1)", "Cosine-Similarity zum Centroid mit Threshold τ."),
    ("revenue", "Umsatz", "Revenue je URL."),
    ("seo_eff", "URL-SEO-Effizienz", "Anteil Top-5-Keywords je URL."),
    ("priority", "Strategische Priorität (Override)", "Multiplikator pro URL (0.5–2.0) + global."),
]

st.subheader("Kriterien auswählen")
left, right = st.columns([2, 1])  # rechts schmaler Block für die Auswahl
with right:
    st.markdown("#### 🧩 Auswahl (rechts)")
    active = {}
    for code, label, hover in CRITERIA:
        active[code] = st.checkbox(label, value=False, help=hover, key=f"chk_{code}")
with left:
    st.markdown("Wähle rechts die Kriterien aus. Danach findest du unten die passenden Upload-Masken und die weiteren Schritte.")

# =============================
# Uploads – erscheinen NACH der Auswahl
# =============================
uploads_needed_text: List[str] = []

def need(item: str):
    uploads_needed_text.append(item)

# Sammle Bedarf je nach Auswahl (nur Liste für Übersicht)
if active.get("sc_clicks") or active.get("sc_impr"):
    need("SC-Datei (URL, Clicks/Impressions)")
if active.get("otv"):
    need("OTV-URL-Datei (optional)")
    need("OTV-Keyword-Datei (optional)")
    need("CTR-Kurve (optional)")
if active.get("ext_pop"):
    need("Extern-Popularität (Backlinks & Ref. Domains)")
if active.get("int_pop"):
    need("Intern-Popularität (Unique Inlinks)")
if active.get("llm_ref"):
    need("LLM Referral Traffic")
if active.get("llm_crawl"):
    need("LLM Crawler Frequenz")
if active.get("offtopic"):
    need("Embeddings (URL, embedding)")
if active.get("revenue"):
    need("Umsatz (URL, revenue)")
if active.get("seo_eff"):
    need("Keywords für Effizienz (keyword, url, position)")
if active.get("priority"):
    need("Prioritäten-Mapping (optional)")

st.markdown("---")
st.subheader("Basierend auf den gewählten Kriterien benötigen wir folgende Daten von dir")
if uploads_needed_text:
    st.markdown("- " + "\n- ".join(dict.fromkeys(uploads_needed_text)))  # dedupe, Reihenfolge behalten
else:
    st.info("Noch keine Kriterien gewählt.")

# Konkrete Upload-Masken (im Hauptbereich, nicht Sidebar)
if active.get("sc_clicks") or active.get("sc_impr"):
    st.markdown("**Search Console Organic Performance** (url, clicks / impressions)")
    file = st.file_uploader("SC-Datei", type=["csv", "xlsx"], key="upl_sc")
    store_upload("sc", file)

if active.get("otv"):
    st.markdown("**Organic Traffic Value (OTV)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        fileA = st.file_uploader("OTV: URL-Value (optional)", type=["csv", "xlsx"], key="upl_otv_url")
        store_upload("otv_url", fileA)
    with c2:
        fileB = st.file_uploader("OTV: Keyword-Datei (optional)", type=["csv", "xlsx"], key="upl_otv_kw")
        store_upload("otv_kw", fileB)
    with c3:
        fileC = st.file_uploader("CTR-Kurve (optional)", type=["csv", "xlsx"], key="upl_ctr")
        store_upload("ctr_curve", fileC)

if active.get("ext_pop"):
    st.markdown("**URL-Popularität extern** (url, backlinks, ref_domains)")
    file = st.file_uploader("Extern-Popularität", type=["csv", "xlsx"], key="upl_ext")
    store_upload("ext", file)

if active.get("int_pop"):
    st.markdown("**URL-Popularität intern** (url, unique_inlinks)")
    file = st.file_uploader("Intern-Popularität", type=["csv", "xlsx"], key="upl_int")
    store_upload("int", file)

if active.get("llm_ref"):
    st.markdown("**LLM Referral Traffic** (url, llm_ref_traffic)")
    file = st.file_uploader("LLM-Referral", type=["csv", "xlsx"], key="upl_llmref")
    store_upload("llmref", file)

if active.get("llm_crawl"):
    st.markdown("**LLM-Crawl-Frequenz** (url, llm_crawl_freq)")
    file = st.file_uploader("LLM-Crawl", type=["csv", "xlsx"], key="upl_llmcrawl")
    store_upload("llmcrawl", file)

if active.get("offtopic"):
    st.markdown("**Offtopic (Embeddings)** (url, embedding)")
    file = st.file_uploader("Embeddings-Datei", type=["csv", "xlsx"], key="upl_emb")
    store_upload("emb", file)

if active.get("revenue"):
    st.markdown("**Umsatz** (url, revenue)")
    file = st.file_uploader("Umsatz-Datei", type=["csv", "xlsx"], key="upl_rev")
    store_upload("rev", file)

if active.get("seo_eff"):
    st.markdown("**SEO-Effizienz (Top-5-Anteil)** (keyword, url, position)")
    file = st.file_uploader("Keyword-Datei (SEO-Effizienz)", type=["csv", "xlsx"], key="upl_eff_kw")
    store_upload("eff_kw", file)

if active.get("priority"):
    st.markdown("**Strategische Priorität (optional)** (url, priority_factor)")
    file = st.file_uploader("Priorität (optional)", type=["csv", "xlsx"], key="upl_prio")
    store_upload("prio", file)

# =============================
# Master URL list builder
# =============================
st.subheader("Master-URL-Liste")
st.markdown(
    "Default ist die **Union aller hochgeladenen URLs**. Optional kannst du eine **eigene Liste** hochladen oder eine Masterliste **aus bis zu zwei** vorhandenen Uploads bilden."
)

use_custom_master = st.checkbox("Eigene Masterliste hochladen (statt Union)")

master_urls: Optional[pd.DataFrame] = None

if use_custom_master:
    master_file = st.file_uploader("Eigene Masterliste (Spalte: url/seite/address/...)", type=["csv", "xlsx"], key="upl_master")
    if master_file:
        dfm = read_table(master_file)
        url_col = find_first_alias(dfm, "url") or st.selectbox("URL-Spalte in Masterliste wählen", dfm.columns, key="map_master_url")
        dfm = ensure_url_column(dfm, url_col)
        master_urls = dfm[[url_col]].rename(columns={url_col: "url_norm"}).drop_duplicates()
else:
    use_two_sources = st.checkbox("Statt Union: Masterliste aus bis zu zwei vorhandenen Dateien mergen")
    if use_two_sources and st.session_state.uploads:
        available = list(st.session_state.uploads.keys())
        pick1 = st.selectbox("Quelle 1", options=[None] + available, index=0, key="master_src1")
        pick2 = st.selectbox("Quelle 2 (optional)", options=[None] + available, index=0, key="master_src2")
        urls = []
        for pick in [pick1, pick2]:
            if pick and pick in st.session_state.uploads:
                df, name = st.session_state.uploads[pick]
                c = find_first_alias(df, "url")
                if c:
                    urls.append(ensure_url_column(df[[c]], c).rename(columns={c: "url_norm"}))
        if urls:
            master_urls = pd.concat(urls, axis=0, ignore_index=True).drop_duplicates()
    else:
        urls = []
        for key, (df, name) in st.session_state.uploads.items():
            c = find_first_alias(df, "url")
            if c:
                urls.append(ensure_url_column(df[[c]], c).rename(columns={c: "url_norm"}))
        if urls:
            master_urls = pd.concat(urls, axis=0, ignore_index=True).drop_duplicates()

if master_urls is None or master_urls.empty:
    st.info("Noch keine Master-URLs erkannt. Lade mindestens eine Datei mit URL-Spalte hoch **oder** lade eine eigene Masterliste.")
else:
    st.success(f"Master-URLs: {len(master_urls):,}")

# =============================
# Compute per-criterion scores
# =============================
def mode_score(series: pd.Series) -> pd.Series:
    return score_series(series, scoring_mode, min_score, bucket_quantiles, bucket_values)

def join_on_master(df: pd.DataFrame, url_col: str, val_cols: List[str]) -> pd.DataFrame:
    d = ensure_url_column(df[[url_col] + val_cols], url_col)
    d = d.rename(columns={url_col: "url_norm"})
    return master_urls.merge(d, on="url_norm", how="left") if master_urls is not None else d

results = {}   # key -> pd.Series
debug_cols = {}  # key -> raw columns for export

# --- SC Clicks / SC Impressions ---
if active.get("sc_clicks") or active.get("sc_impr"):
    if "sc" in st.session_state.uploads:
        df_sc, _ = st.session_state.uploads["sc"]
        need = ["url"]
        if active.get("sc_clicks"): need.append("clicks")
        if active.get("sc_impr"):  need.append("impressions")
        colmap = require_columns_ui("sc", df_sc, need, "SC-Datei")
        urlc = colmap.get("url")
        df_sc = ensure_url_column(df_sc, urlc)
        if master_urls is None:
            st.warning("Masterliste fehlt noch; bitte oben konfigurieren.")
        else:
            debug_cols["sc"] = {}
            if active.get("sc_clicks"):
                ccol = colmap.get("clicks")
                scj = join_on_master(df_sc, urlc, [ccol])
                s = mode_score(scj[ccol])
                results["sc_clicks"] = s.fillna(0.0)
                debug_cols["sc"]["clicks_raw"] = scj[ccol]
            if active.get("sc_impr"):
                icol = colmap.get("impressions")
                scj = join_on_master(df_sc, urlc, [icol])
                s = mode_score(scj[icol])
                results["sc_impr"] = s.fillna(0.0)
                debug_cols["sc"]["impressions_raw"] = scj[icol]
    elif master_urls is not None:
        if active.get("sc_clicks"):
            results["sc_clicks"] = pd.Series(0.0, index=master_urls.index)
        if active.get("sc_impr"):
            results["sc_impr"] = pd.Series(0.0, index=master_urls.index)

# --- OTV (Organic Traffic Value) ---
if active.get("otv"):
    s_otv = None
    raw_val = None
    if "otv_url" in st.session_state.uploads:
        df_u, _ = st.session_state.uploads["otv_url"]
        colmap = require_columns_ui("otv_url", df_u, ["url"], "OTV URL-Datei")
        urlc = colmap.get("url")
        val_col = find_first_alias(df_u, "traffic_value")
        pot_col = find_first_alias(df_u, "potential_traffic_url")
        cpc_col = find_first_alias(df_u, "cpc")
        df_u = ensure_url_column(df_u, urlc)
        d = master_urls.merge(df_u, left_on="url_norm", right_on=urlc, how="left") if master_urls is not None else df_u
        if val_col:
            raw_val = d[val_col]
        elif pot_col and cpc_col:
            raw_val = d[pot_col] * d[cpc_col]
        elif pot_col:
            raw_val = d[pot_col]
    if raw_val is None and "otv_kw" in st.session_state.uploads:
        df_k, _ = st.session_state.uploads["otv_kw"]
        colmap = require_columns_ui("otv_kw", df_k, ["keyword","url","position","search_volume"], "OTV Keyword-Datei")
        urlc = colmap.get("url"); posc = colmap.get("position"); svc = colmap.get("search_volume")
        cpcc = find_first_alias(df_k, "cpc")
        ctr_df = None
        if "ctr_curve" in st.session_state.uploads:
            ctr_df, _ = st.session_state.uploads["ctr_curve"]
            ctr_map = require_columns_ui("ctr_curve", ctr_df, ["position","ctr"], "CTR-Kurve")
            ctr_df = ctr_df[[ctr_map["position"], ctr_map["ctr"]]].rename(columns={ctr_map["position"]:"position", ctr_map["ctr"]:"ctr"})
        else:
            ctr_df = default_ctr_curve()
        df_k = ensure_url_column(df_k, urlc)
        ctrs = df_k[posc].map(lambda p: get_ctr_for_pos(p, ctr_df))
        pot_traffic = pd.to_numeric(df_k[svc], errors="coerce").fillna(0) * ctrs
        raw_row_val = pot_traffic * pd.to_numeric(df_k[cpcc], errors="coerce").fillna(0) if cpcc else pot_traffic
        agg = df_k.assign(_val=raw_row_val).groupby(urlc, as_index=False)["_val"].sum()
        d = master_urls.merge(agg, left_on="url_norm", right_on=urlc, how="left") if master_urls is not None else agg
        raw_val = d["_val"]
    if master_urls is not None:
        results["otv"] = pd.Series(0.0, index=master_urls.index) if raw_val is None else mode_score(raw_val).fillna(0.0)
        if raw_val is not None:
            debug_cols["otv"] = {"otv_raw": raw_val}

# --- External popularity (Backlinks & RD) ---
if active.get("ext_pop"):
    if "ext" in st.session_state.uploads:
        df_e, _ = st.session_state.uploads["ext"]
        colmap = require_columns_ui("ext", df_e, ["url","backlinks","ref_domains"], "Extern-Popularität")
        urlc = colmap.get("url"); blc = colmap.get("backlinks"); rdc = colmap.get("ref_domains")
        df_e = ensure_url_column(df_e, urlc)
        d = join_on_master(df_e, urlc, [blc, rdc])
        ext = 0.3 * mode_score(d[blc]) + 0.7 * mode_score(d[rdc])
        results["ext_pop"] = ext.fillna(0.0)
        debug_cols["ext_pop"] = {"backlinks_raw": d[blc], "ref_domains_raw": d[rdc]}
    elif master_urls is not None:
        results["ext_pop"] = pd.Series(0.0, index=master_urls.index)

# --- Internal popularity (Unique Inlinks) ---
if active.get("int_pop"):
    if "int" in st.session_state.uploads:
        df_i, _ = st.session_state.uploads["int"]
        colmap = require_columns_ui("int", df_i, ["url","unique_inlinks"], "Interne Links")
        urlc = colmap.get("url"); inc = colmap.get("unique_inlinks")
        df_i = ensure_url_column(df_i, urlc)
        d = join_on_master(df_i, urlc, [inc])
        results["int_pop"] = mode_score(d[inc]).fillna(0.0)
        debug_cols["int_pop"] = {"unique_inlinks_raw": d[inc]}
    elif master_urls is not None:
        results["int_pop"] = pd.Series(0.0, index=master_urls.index)

# --- LLM Referral ---
if active.get("llm_ref"):
    if "llmref" in st.session_state.uploads:
        df_l, _ = st.session_state.uploads["llmref"]
        colmap = require_columns_ui("llmref", df_l, ["url","llm_ref_traffic"], "LLM Referral")
        urlc = colmap.get("url"); rc = colmap.get("llm_ref_traffic")
        df_l = ensure_url_column(df_l, urlc)
        d = join_on_master(df_l, urlc, [rc])
        results["llm_ref"] = mode_score(d[rc]).fillna(0.0)
        debug_cols["llm_ref"] = {"llm_ref_traffic_raw": d[rc]}
    elif master_urls is not None:
        results["llm_ref"] = pd.Series(0.0, index=master_urls.index)

# --- LLM Crawl ---
if active.get("llm_crawl"):
    if "llmcrawl" in st.session_state.uploads:
        df_lc, _ = st.session_state.uploads["llmcrawl"]
        colmap = require_columns_ui("llmcrawl", df_lc, ["url","llm_crawl_freq"], "LLM Crawl")
        urlc = colmap.get("url"); cc = colmap.get("llm_crawl_freq")
        df_lc = ensure_url_column(df_lc, urlc)
        d = join_on_master(df_lc, urlc, [cc])
        results["llm_crawl"] = mode_score(d[cc]).fillna(0.0)
        debug_cols["llm_crawl"] = {"llm_crawl_freq_raw": d[cc]}
    elif master_urls is not None:
        results["llm_crawl"] = pd.Series(0.0, index=master_urls.index)

# --- Offtopic 0/1 ---
if active.get("offtopic"):
    if "emb" in st.session_state.uploads:
        df_emb, _ = st.session_state.uploads["emb"]
        colmap = require_columns_ui("emb", df_emb, ["url","embedding"], "Embeddings")
        urlc = colmap.get("url"); ec = colmap.get("embedding")
        df_emb = ensure_url_column(df_emb, urlc)
        def parse_vec(x):
            if isinstance(x, (list, tuple, np.ndarray)): return np.array(x, dtype=float)
            if isinstance(x, str):
                xs = x.strip()
                if xs.startswith("[") and xs.endswith("]"):
                    try: return np.array(json.loads(xs), dtype=float)
                    except Exception: pass
                parts = re.split(r"[,\s;|]+", xs.strip("[]() "))
                try: return np.array([float(p) for p in parts if p != ""], dtype=float)
                except Exception: return None
            return None
        tmp = df_emb[[urlc, ec]].copy()
        tmp["_vec"] = tmp[ec].map(parse_vec)
        valid = tmp[tmp["_vec"].map(lambda v: isinstance(v, np.ndarray))]
        if valid.empty:
            if master_urls is not None:
                results["offtopic"] = pd.Series(0.0, index=master_urls.index)
        else:
            mat = np.vstack(valid["_vec"].values)
            centroid = mat.mean(axis=0)
            def cos_sim(vec):
                a = vec / (np.linalg.norm(vec) + 1e-12)
                b = centroid / (np.linalg.norm(centroid) + 1e-12)
                return float(np.dot(a, b))
            valid["_sim"] = valid["_vec"].map(cos_sim)
            d = master_urls.merge(valid[[urlc, "_sim"]], left_on="url_norm", right_on=urlc, how="left") if master_urls is not None else valid
            s = pd.Series(0.0, index=d.index)
            s.loc[d["_sim"] >= offtopic_tau] = 1.0
            results["offtopic"] = s.fillna(0.0)
            debug_cols["offtopic"] = {"similarity": d["_sim"]}
    elif master_urls is not None:
        results["offtopic"] = pd.Series(0.0, index=master_urls.index)

# --- Revenue ---
if active.get("revenue"):
    if "rev" in st.session_state.uploads:
        df_r, _ = st.session_state.uploads["rev"]
        colmap = require_columns_ui("rev", df_r, ["url","revenue"], "Umsatz")
        urlc = colmap.get("url"); rc = colmap.get("revenue")
        df_r = ensure_url_column(df_r, urlc)
        d = join_on_master(df_r, urlc, [rc])
        results["revenue"] = mode_score(d[rc]).fillna(0.0)
        debug_cols["revenue"] = {"revenue_raw": d[rc]}
    elif master_urls is not None:
        results["revenue"] = pd.Series(0.0, index=master_urls.index)

# --- SEO Efficiency (Top-5 share) ---
if active.get("seo_eff"):
    eff_series = None
    if "eff_kw" in st.session_state.uploads:
        df_e, _ = st.session_state.uploads["eff_kw"]
        colmap = require_columns_ui("eff_kw", df_e, ["keyword","url","position"], "SEO-Effizienz Keywords")
        urlc = colmap.get("url"); posc = colmap.get("position")
        df_e = ensure_url_column(df_e, urlc)
        grp = df_e.groupby(urlc)[posc].apply(lambda s: (pd.to_numeric(s, errors="coerce") <= 5).sum() / max(1, s.shape[0]))
        d = master_urls.merge(grp.rename("eff"), left_on="url_norm", right_index=True, how="left") if master_urls is not None else grp
        eff_series = d["eff"] if isinstance(d, pd.DataFrame) else grp
    if eff_series is None and master_urls is not None:
        results["seo_eff"] = pd.Series(0.0, index=master_urls.index)
    elif eff_series is not None:
        results["seo_eff"] = mode_score(eff_series).fillna(0.0)
        debug_cols["seo_eff"] = {"eff_raw": eff_series}

# --- Priority override ---
priority_url = None
if active.get("priority"):
    if "prio" in st.session_state.uploads:
        df_p, _ = st.session_state.uploads["prio"]
        colmap = require_columns_ui("prio", df_p, ["url","priority_factor"], "Priorität")
        urlc = colmap.get("url"); pc = colmap.get("priority_factor")
        df_p = ensure_url_column(df_p, urlc)
        d = join_on_master(df_p, urlc, [pc])
        priority_url = pd.to_numeric(d[pc], errors="coerce").fillna(1.0).clip(0.5, 2.0)
    elif master_urls is not None:
        priority_url = pd.Series(1.0, index=master_urls.index)

# =============================
# Gewichte & Aggregation
# =============================
st.subheader("Gewichtung der aktiven Kriterien")

weight_keys = [k for k in ["sc_clicks","sc_impr","otv","ext_pop","int_pop","llm_ref","llm_crawl","offtopic","revenue","seo_eff"] if active.get(k)]
weights = {}
if weight_keys:
    cols = st.columns(len(weight_keys))
    for i, k in enumerate(weight_keys):
        label = next(lbl for code,lbl,_ in CRITERIA if code == k)
        weights[k] = cols[i].number_input(f"Gewicht: {label}", min_value=0.0, value=1.0, step=0.1, key=f"w_{k}")
else:
    st.info("Keine Kriterien aktiv. Bitte mindestens ein Kriterium aktivieren.")

w_sum = sum(weights.values()) if weights else 0.0
weights_norm = {k: (v / w_sum) for k, v in weights.items()} if w_sum > 0 else {k: 0.0 for k in weight_keys}

# Compute base and final
if master_urls is not None and weight_keys:
    df_out = master_urls.copy()
    for k in weight_keys:
        s = results.get(k, pd.Series(0.0, index=df_out.index))
        df_out[f"score__{k}"] = s.values
    base = np.zeros(len(df_out))
    for k, wn in weights_norm.items():
        base += wn * df_out[f"score__{k}"].values
    df_out["base_score"] = base
    if active.get("priority") and priority_url is not None:
        df_out["priority_factor_url"] = priority_url.values
    else:
        df_out["priority_factor_url"] = 1.0
    df_out["priority_factor_global"] = priority_global
    df_out["final_score"] = df_out["base_score"] * df_out["priority_factor_url"] * df_out["priority_factor_global"]
    df_out = df_out.sort_values("final_score", ascending=False).reset_index(drop=True)

    st.subheader("Ergebnis")
    st.dataframe(df_out.head(100), use_container_width=True, hide_index=True)

    # Downloads
    st.markdown("### Export")
    def to_csv(df):
        return df.to_csv(index=False).encode("utf-8-sig")
    csv_bytes = to_csv(df_out)
    st.download_button("⬇️ CSV herunterladen", data=csv_bytes, file_name="one_url_scoreboard.csv", mime="text/csv")

    try:
        import xlsxwriter  # optional
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="scores")
            for k, cols in debug_cols.items():
                if isinstance(cols, dict) and cols:
                    dd = pd.DataFrame(cols)
                    dd.to_excel(writer, index=False, sheet_name=f"raw_{k}"[:31])
        st.download_button("⬇️ XLSX herunterladen", data=buf.getvalue(), file_name="one_url_scoreboard.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.caption("Hinweis: Für XLSX-Export kann das Paket `xlsxwriter` erforderlich sein.")

    config = {
        "scoring_mode": scoring_mode,
        "min_score": min_score if scoring_mode == "Rank (linear)" else None,
        "bucket_quantiles": bucket_quantiles if scoring_mode != "Rank (linear)" else None,
        "bucket_values": bucket_values if scoring_mode != "Rank (linear)" else None,
        "offtopic_tau": offtopic_tau,
        "priority_global": priority_global,
        "weights": weights,
        "weights_norm": weights_norm,
        "active_criteria": [k for k in active.keys() if active[k]],
        "uploads_used": {k: name for k, (_, name) in st.session_state.uploads.items()},
        "column_maps": st.session_state.column_maps,
        "notes": "URLs ohne Daten im aktiven Kriterium erhalten 0.0. Masterliste standardmäßig als Union aller hochgeladenen URLs (oder gemäß Auswahl).",
    }
    st.download_button(
        "⬇️ Config (JSON)",
        data=json.dumps(config, indent=2).encode("utf-8"),
        file_name="one_url_scoreboard_config.json",
        mime="application/json",
    )
else:
    st.info("Bitte Masterliste erzeugen und mindestens ein Kriterium aktivieren & gewichten.")
