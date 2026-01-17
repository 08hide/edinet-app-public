# app_db_compare.py（JPX33業種・確定仕様 + CSVをDB比較に混ぜる版・全文 / 上部操作UI横並び / 削除・CSV全削除安定版）
# ✅ 仕様
# - 業界名はJPX33（industry_master.sector33）だけ（EDINET業種は使わない）
# - 会社検索：会社名検索のみ
# - 検索候補：metrics_yearlyにデータがある企業のみ
# - 候補表示：会社名（JPX33業種）※紐付かない場合は「業界不明」
# - 上部UI：企業検索 / 業界 / 期間 / CSV を横並び
# - 業界平均トグル／分析トグルあり
# - CSVはDB比較に混ぜる（CSVは企業として追加、最大8件はDB+CSV合算）
# - 比較中：最大8件、1件1行、右端×削除、全解除（on_clickで安定）
# - 期間：2016〜2025 slider（DB/CSV共通）
# - 最新年比較表：なし
# - グラフ：単位明示（売上/利益/CF=億円、比率=%）
# - CF表：企業ごとに分けて表示、最初から展開
# - 生データ表示トグル、CSVダウンロードあり
# ✅ テンプレ：ボタンのみ（cf_type列なし）
# ✅ CF型：CSVに無くてもCFO/CFI/CFFの符号から自動補完（企業タイプまとめセクションは作らない）
# ✅ 「グラフからわかること」：企業ごとにブロック分け（改行）
# ✅ CSVの削除が復活しない対策：
#    - file_uploader が保持するアップロード済みファイルで再追加されるのを防ぐため、
#      追加済みCSVファイル名を csv_uploaded_names に記録し、同名は自動再追加しない
#    - CSV全削除 / 全解除のときは uploader_key を増やして file_uploader を作り直し「選択済み」を消す
# ❌ 排除：履歴リセットボタン

import sqlite3
from io import BytesIO
import pandas as pd
import streamlit as st

DB_PATH = "test.db"

MAX_COMPANIES = 8
Y_MIN_FIXED = 2016
Y_MAX_FIXED = 2025


# =========================
# DB helpers
# =========================
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def table_exists(con, name: str) -> bool:
    r = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return r is not None


@st.cache_data(show_spinner=False)
def has_industry_master() -> bool:
    with db() as con:
        return table_exists(con, "industry_master")


@st.cache_data(show_spinner=False)
def list_sector33() -> list[str]:
    with db() as con:
        if not table_exists(con, "industry_master"):
            return []
        rows = con.execute(
            """
            SELECT DISTINCT sector33
            FROM industry_master
            WHERE sector33 IS NOT NULL AND TRIM(sector33) <> ''
            ORDER BY sector33
            """
        ).fetchall()
    return [r[0] for r in rows]


@st.cache_data(show_spinner=False)
def count_visible_companies(sector33: str | None = None) -> int:
    """候補に出る企業数（metrics_yearlyに1行以上ある企業）"""
    with db() as con:
        if not (table_exists(con, "companies") and table_exists(con, "metrics_yearly")):
            return 0

        im_ok = table_exists(con, "industry_master")

        if sector33 and im_ok:
            row = con.execute(
                """
                SELECT COUNT(DISTINCT c.edinet_code)
                FROM companies c
                JOIN industry_master im
                  ON im.securities_code = c.securities_code
                WHERE im.sector33 = ?
                  AND EXISTS (
                    SELECT 1 FROM metrics_yearly my
                    WHERE my.edinet_code = c.edinet_code
                    LIMIT 1
                  )
                """,
                (sector33,),
            ).fetchone()
        else:
            row = con.execute(
                """
                SELECT COUNT(DISTINCT c.edinet_code)
                FROM companies c
                WHERE EXISTS (
                    SELECT 1 FROM metrics_yearly my
                    WHERE my.edinet_code = c.edinet_code
                    LIMIT 1
                )
                """
            ).fetchone()

        return int(row[0] or 0)


@st.cache_data(show_spinner=False)
def company_search(keyword: str, sector33: str | None, limit: int = 120) -> pd.DataFrame:
    """
    ・会社名検索のみ
    ・metrics_yearlyにデータがある企業のみ
    ・候補表示用に sector33 を付与（JPX33）
    ・sector33指定があればJPX33で絞る
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return pd.DataFrame(columns=["edinet_code", "name", "sector33"])

    with db() as con:
        if not (table_exists(con, "companies") and table_exists(con, "metrics_yearly")):
            return pd.DataFrame(columns=["edinet_code", "name", "sector33"])

        im_ok = table_exists(con, "industry_master")

        if sector33 and im_ok:
            rows = con.execute(
                """
                SELECT c.edinet_code, c.name, im.sector33
                FROM companies c
                JOIN industry_master im
                  ON im.securities_code = c.securities_code
                WHERE im.sector33 = ?
                  AND c.name LIKE ?
                  AND EXISTS (
                    SELECT 1 FROM metrics_yearly my
                    WHERE my.edinet_code = c.edinet_code
                    LIMIT 1
                  )
                ORDER BY c.name
                LIMIT ?
                """,
                (sector33, f"%{keyword}%", limit),
            ).fetchall()
        else:
            if im_ok:
                rows = con.execute(
                    """
                    SELECT c.edinet_code, c.name, im.sector33
                    FROM companies c
                    LEFT JOIN industry_master im
                      ON im.securities_code = c.securities_code
                    WHERE c.name LIKE ?
                      AND EXISTS (
                        SELECT 1 FROM metrics_yearly my
                        WHERE my.edinet_code = c.edinet_code
                        LIMIT 1
                      )
                    ORDER BY c.name
                    LIMIT ?
                    """,
                    (f"%{keyword}%", limit),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT c.edinet_code, c.name, NULL as sector33
                    FROM companies c
                    WHERE c.name LIKE ?
                      AND EXISTS (
                        SELECT 1 FROM metrics_yearly my
                        WHERE my.edinet_code = c.edinet_code
                        LIMIT 1
                      )
                    ORDER BY c.name
                    LIMIT ?
                    """,
                    (f"%{keyword}%", limit),
                ).fetchall()

    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(show_spinner=False)
def fetch_labels_for_codes(codes: list[str]) -> dict[str, dict]:
    """比較中の表示：会社名（JPX33業種）"""
    if not codes:
        return {}

    with db() as con:
        if not table_exists(con, "companies"):
            return {}

        ph = ",".join(["?"] * len(codes))
        im_ok = table_exists(con, "industry_master")

        if im_ok:
            rows = con.execute(
                f"""
                SELECT c.edinet_code, c.name, im.sector33
                FROM companies c
                LEFT JOIN industry_master im
                  ON im.securities_code = c.securities_code
                WHERE c.edinet_code IN ({ph})
                """,
                codes,
            ).fetchall()
            out = {}
            for r in rows:
                sec = (r["sector33"] or "").strip() or "業界不明"
                out[r["edinet_code"]] = {"name": r["name"], "label": f"{r['name']}（{sec}）"}
            return out

        rows = con.execute(
            f"SELECT edinet_code, name FROM companies WHERE edinet_code IN ({ph})",
            codes,
        ).fetchall()
        return {r["edinet_code"]: {"name": r["name"], "label": f"{r['name']}（業界不明）"} for r in rows}


@st.cache_data(show_spinner=False)
def load_metrics_yearly(edinet_codes: list[str]) -> pd.DataFrame:
    if not edinet_codes:
        return pd.DataFrame()

    ph = ",".join(["?"] * len(edinet_codes))
    sql = f"""
        SELECT
          edinet_code,
          fiscal_year,
          sales,
          net_income,
          equity_ratio,
          current_ratio,
          cfo,
          cfi,
          cff,
          cf_type
        FROM metrics_yearly
        WHERE edinet_code IN ({ph})
        ORDER BY fiscal_year
    """
    with db() as con:
        if not table_exists(con, "metrics_yearly"):
            return pd.DataFrame()
        df = pd.read_sql_query(sql, con, params=edinet_codes)

    for col in ["fiscal_year", "sales", "net_income", "equity_ratio", "current_ratio", "cfo", "cfi", "cff"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_years(df: pd.DataFrame, y_min: int, y_max: int) -> pd.DataFrame:
    return df[(df["fiscal_year"] >= y_min) & (df["fiscal_year"] <= y_max)].copy()


def to_pct(series_or_value):
    """0-1 / 0-100 の揺れを吸収して % へ寄せる"""
    s = pd.to_numeric(series_or_value, errors="coerce")
    if isinstance(s, pd.Series):
        if s.dropna().empty:
            return s
        mx = s.dropna().abs().max()
        return s * 100 if mx <= 1.5 else s
    else:
        if pd.isna(s):
            return s
        return s * 100 if abs(float(s)) <= 1.5 else s


# =========================
# CF型（cf_type）自動補完
# =========================
def classify_cf_type(cfo, cfi, cff) -> str | None:
    """CFO/CFI/CFF の符号でCF型を返す（欠損があれば None）"""
    cfo = pd.to_numeric(cfo, errors="coerce")
    cfi = pd.to_numeric(cfi, errors="coerce")
    cff = pd.to_numeric(cff, errors="coerce")
    if pd.isna(cfo) or pd.isna(cfi) or pd.isna(cff):
        return None

    key = ("+" if cfo >= 0 else "-", "+" if cfi >= 0 else "-", "+" if cff >= 0 else "-")
    return {
        ("+","-","-"): "優良企業型",
        ("+","-","+"): "積極投資型",
        ("+","+","-"): "選択と集中型",
        ("-","-","+"): "ベンチャー型",
        ("-","+","-"): "じり貧型",
        ("-","+","+"): "危険水域型",
    }.get(key, None)


def fill_cf_type(df: pd.DataFrame) -> pd.DataFrame:
    """df の cf_type が空の行を CFO/CFI/CFF から補完する"""
    if df is None or df.empty:
        return df

    if "cf_type" not in df.columns:
        df["cf_type"] = pd.NA

    need = df["cf_type"].isna() | (df["cf_type"].astype(str).str.strip() == "")
    if need.any():
        df.loc[need, "cf_type"] = df.loc[need].apply(
            lambda r: classify_cf_type(r.get("cfo"), r.get("cfi"), r.get("cff")),
            axis=1
        )
    return df


# =========================
# CSV helpers（DB比較に混ぜる）
# =========================
def _read_csv_uploaded(file) -> pd.DataFrame:
    data = file.getvalue()
    try:
        return pd.read_csv(BytesIO(data), encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(BytesIO(data), encoding="cp932")


def _normalize_csv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV → metrics_yearly互換の列に寄せる
    必須: year or fiscal_year
    任意: sales/sales_yen, net_income/net_income_yen, equity_ratio(_pct), current_ratio(_pct),
         cfo/cfo_yen, cfi/cfi_yen, cff/cff_yen
    ※ cf_type は不要（自動で埋める）
    """
    df = df.copy()

    if "fiscal_year" not in df.columns and "year" in df.columns:
        df["fiscal_year"] = df["year"]
    df["fiscal_year"] = pd.to_numeric(df.get("fiscal_year"), errors="coerce")

    rename_map = {
        "sales_yen": "sales",
        "net_income_yen": "net_income",
        "cfo_yen": "cfo",
        "cfi_yen": "cfi",
        "cff_yen": "cff",
        "equity_ratio_pct": "equity_ratio",
        "current_ratio_pct": "current_ratio",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    for col in ["sales", "net_income", "equity_ratio", "current_ratio", "cfo", "cfi", "cff"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "equity_ratio" in df.columns:
        df["equity_ratio"] = to_pct(df["equity_ratio"])
    if "current_ratio" in df.columns:
        df["current_ratio"] = to_pct(df["current_ratio"])

    if df["fiscal_year"].dropna().empty:
        raise ValueError("CSVに year または fiscal_year がありません。")

    df = df.dropna(subset=["fiscal_year"]).sort_values("fiscal_year")
    df = fill_cf_type(df)
    return df


@st.cache_data(show_spinner=False)
def load_industry_avg_jpx33(sector33: str, y_min: int, y_max: int) -> pd.DataFrame:
    """JPX33業種の年次平均（mean）※DB企業だけで算出"""
    with db() as con:
        if not (
            table_exists(con, "industry_master")
            and table_exists(con, "metrics_yearly")
            and table_exists(con, "companies")
        ):
            return pd.DataFrame()

        df = pd.read_sql_query(
            """
            SELECT
              my.fiscal_year,
              my.sales,
              my.net_income,
              my.equity_ratio,
              my.current_ratio,
              my.cfo,
              my.cfi,
              my.cff
            FROM metrics_yearly my
            JOIN companies c
              ON c.edinet_code = my.edinet_code
            JOIN industry_master im
              ON im.securities_code = c.securities_code
            WHERE im.sector33 = ?
              AND my.fiscal_year BETWEEN ? AND ?
            """,
            con,
            params=[sector33, y_min, y_max],
        )

    if df.empty:
        return df

    for col in ["sales", "net_income", "equity_ratio", "current_ratio", "cfo", "cfi", "cff"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    g = df.groupby("fiscal_year")
    out = pd.DataFrame(index=sorted(df["fiscal_year"].dropna().unique()))
    out.index.name = "fiscal_year"

    out["sales_oku"] = g["sales"].mean() / 1e8
    out["net_income_oku"] = g["net_income"].mean() / 1e8
    out["equity_ratio_pct"] = to_pct(g["equity_ratio"].mean())
    out["current_ratio_pct"] = to_pct(g["current_ratio"].mean())
    out["cfo_oku"] = g["cfo"].mean() / 1e8
    out["cfi_oku"] = g["cfi"].mean() / 1e8
    out["cff_oku"] = g["cff"].mean() / 1e8

    return out.sort_index()


def pivot_by_code(df: pd.DataFrame, value_col: str, code_to_label: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    p = (
        df.pivot_table(index="fiscal_year", columns="edinet_code", values=value_col, aggfunc="first")
        .sort_index()
    )
    return p.rename(columns=code_to_label)


def add_avg_line(chart_df: pd.DataFrame, avg_series: pd.Series, label: str) -> pd.DataFrame:
    out = chart_df.copy()
    out[label] = avg_series.reindex(chart_df.index)
    return out


def make_analysis(chart_df: pd.DataFrame, unit: str, avg_label: str | None = None) -> list[str]:
    """企業ごとにブロック分けして返す（改行あり）"""
    if chart_df.empty or len(chart_df.index) < 2:
        return ["データが少ないため分析できません。"]

    years = list(chart_df.index)
    y0, y1 = years[0], years[-1]
    results: list[str] = []

    avg_series = chart_df[avg_label] if (avg_label and avg_label in chart_df.columns) else None

    for col in chart_df.columns:
        if avg_label and col == avg_label:
            continue

        s = pd.to_numeric(chart_df[col], errors="coerce").dropna()
        if len(s) < 2:
            continue

        a, b = float(s.iloc[0]), float(s.iloc[-1])
        trend = "増加" if b > a else ("減少" if b < a else "横ばい")

        block = []
        block.append(f"【{col}】")
        block.append(f"{y0}年 → {y1}年で{trend}")
        block.append(f"{a:,.2f}{unit} → {b:,.2f}{unit}")

        if avg_series is not None:
            avg_last = pd.to_numeric(avg_series.reindex(chart_df.index), errors="coerce").iloc[-1]
            if pd.notna(avg_last) and abs(float(avg_last)) > 1e-9:
                diff_pct = (b - float(avg_last)) / float(avg_last) * 100
                block.append(f"業界平均との差：{diff_pct:+.1f}%")

        results.append("\n".join(block))
        results.append("")

    return results[:-1] if results else ["分析できるデータがありませんでした。"]


# =========================
# UI style（hero/panel）
# =========================
st.set_page_config(page_title="企業分析を5分で！", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(135deg, #eef2ff, #f8fafc); color: #0f172a; }
      .block-container { padding-top: 3.5rem; padding-bottom: 2.6rem; max-width: 1400px; }

      .hero {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        padding: 18px 18px; border-radius: 16px; color: white;
        box-shadow: 0 8px 22px rgba(0,0,0,0.14); margin-bottom: 14px;
      }
      .panel {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(255,255,255,0.55);
        border-radius: 16px; padding: 14px 14px;
        box-shadow: 0 10px 24px rgba(15,23,42,0.08);
        margin: 10px 0 14px; backdrop-filter: blur(10px);
      }

      /* 見出しを大きく */
      .panel-title { font-size: 18px; font-weight: 900; margin: 0 0 10px; color: #0f172a; }
      .section-title { font-size: 20px; font-weight: 950; margin: 18px 0 10px; color: #0f172a; }

      .muted { font-size: 12px; color: #64748b; margin-top: 4px; }
      button[kind="secondary"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div style="font-size:26px; font-weight:900; margin:0; line-height:1.1;">
        企業分析を5分で！
      </div>
      <div style="margin-top:6px; font-size:13px; opacity:0.95;">
        売上、利益、業界平均を一目で比較
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# Session state
# =========================
if "selected_codes" not in st.session_state:
    st.session_state.selected_codes = []
if "sector33_sel" not in st.session_state:
    st.session_state.sector33_sel = "（全業界）"
if "show_avg" not in st.session_state:
    st.session_state.show_avg = False
if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False
if "year_range" not in st.session_state:
    st.session_state.year_range = (Y_MIN_FIXED, Y_MAX_FIXED)
if "show_raw" not in st.session_state:
    st.session_state.show_raw = False
if "search_prev_picks" not in st.session_state:
    st.session_state.search_prev_picks = []
if "csv_items" not in st.session_state:
    st.session_state.csv_items = []
if "csv_uploaded_names" not in st.session_state:
    st.session_state.csv_uploaded_names = set()
if "csv_uploader_key" not in st.session_state:
    # file_uploader を作り直して「選択済みファイル」を消すため
    st.session_state.csv_uploader_key = 0


# =========================
# 削除（×）安定化：コールバック
# =========================
def _remove_from_compare(code: str):
    if str(code).startswith("CSV:"):
        st.session_state.csv_items = [it for it in st.session_state.csv_items if it["id"] != code]
        # csv_uploaded_names は消さない（削除後に同名が自動復活しない）
    else:
        st.session_state.selected_codes = [c for c in st.session_state.selected_codes if c != code]


def _clear_all():
    st.session_state.selected_codes = []
    st.session_state.csv_items = []
    st.session_state.search_prev_picks = []
    st.session_state.csv_uploaded_names = set()
    st.session_state.csv_uploader_key += 1  # uploaderを作り直して選択済みを消す


def _clear_csv_only():
    st.session_state.csv_items = []
    st.session_state.csv_uploaded_names = set()
    st.session_state.csv_uploader_key += 1  # uploaderを作り直して選択済みを消す


# =========================
# 上部操作UI（横並び）
# =========================
use_sector = None if st.session_state.sector33_sel == "（全業界）" else st.session_state.sector33_sel
col_search, col_sector, col_year, col_csv = st.columns([2.3, 1.2, 1.2, 1.5], gap="large")

with col_search:
    st.markdown('<div class="panel"><div class="panel-title">企業検索（DB）</div>', unsafe_allow_html=True)

    kw = st.text_input("会社名で検索", placeholder="例：ニトリ / トヨタ / ソニー", key="kw")

    total_n = count_visible_companies(None)
    if use_sector:
        n2 = count_visible_companies(use_sector)
        st.caption(f"対象：{total_n:,}社 / 絞り込み：{n2:,}社（{use_sector}）")
    else:
        st.caption(f"対象：{total_n:,}社（指標データあり）")

    df_hits = company_search(kw, use_sector, limit=120)

    picked_labels = []
    label_to_code = {}

    if not df_hits.empty:
        def _lab(r):
            s = (r.get("sector33") or "").strip() or "業界不明"
            return f"{r['name']}（{s}）"

        df_hits["label"] = df_hits.apply(_lab, axis=1)
        label_to_code = dict(zip(df_hits["label"].tolist(), df_hits["edinet_code"].tolist()))

        picked_labels = st.multiselect(
            "候補（選んだ瞬間に追加／保持）",
            options=df_hits["label"].tolist(),
            default=[],
            key="pick_labels",
        )

        prev = set(st.session_state.search_prev_picks)
        newly = [x for x in picked_labels if x not in prev]

        if newly:
            before = list(st.session_state.selected_codes)

            combined_now = list(st.session_state.selected_codes) + [it["id"] for it in st.session_state.csv_items]
            capacity = max(0, MAX_COMPANIES - len(combined_now))

            for lab in newly:
                if capacity <= 0:
                    break
                code = label_to_code.get(lab)
                if not code:
                    continue
                if code not in st.session_state.selected_codes:
                    st.session_state.selected_codes.append(code)
                    capacity -= 1

            added = [c for c in st.session_state.selected_codes if c not in before]
            if not added and (len(list(st.session_state.selected_codes) + [it["id"] for it in st.session_state.csv_items]) >= MAX_COMPANIES):
                st.warning(f"最大{MAX_COMPANIES}件までです。先に『比較中』から外してください。")
            elif added:
                st.success(f"{len(added)}件 追加しました。")

        st.session_state.search_prev_picks = picked_labels
    else:
        if kw.strip():
            st.caption("一致なし（データがある企業のみ表示）")
        else:
            st.caption("検索すると候補が表示されます。")

    st.markdown("</div>", unsafe_allow_html=True)

with col_sector:
    st.markdown('<div class="panel"><div class="panel-title">業界</div>', unsafe_allow_html=True)

    sector33_options = ["（全業界）"]
    if has_industry_master():
        sector33_options += list_sector33()

    new_sector = st.selectbox(
        "JPX33業種",
        options=sector33_options,
        index=sector33_options.index(st.session_state.sector33_sel)
        if st.session_state.sector33_sel in sector33_options else 0,
        key="sector_box",
    )

    if new_sector != st.session_state.sector33_sel:
        st.session_state.sector33_sel = new_sector
        st.session_state.search_prev_picks = []
        st.rerun()

    st.session_state.show_avg = st.toggle("業界平均", value=st.session_state.show_avg, key="tog_avg")
    st.session_state.show_analysis = st.toggle("分析文", value=st.session_state.show_analysis, key="tog_analysis")

    st.markdown("</div>", unsafe_allow_html=True)

with col_year:
    st.markdown('<div class="panel"><div class="panel-title">期間</div>', unsafe_allow_html=True)

    y_min, y_max = st.slider(
        "2016〜2025",
        min_value=Y_MIN_FIXED,
        max_value=Y_MAX_FIXED,
        value=st.session_state.year_range,
        step=1,
        key="year_slider",
    )
    st.session_state.year_range = (int(y_min), int(y_max))

    st.session_state.show_raw = st.toggle("生データ", value=st.session_state.show_raw, key="tog_raw")

    st.markdown("</div>", unsafe_allow_html=True)

with col_csv:
    st.markdown('<div class="panel"><div class="panel-title">CSV</div>', unsafe_allow_html=True)

    # テンプレ（cf_typeなし）
    template_csv = "fiscal_year,sales,net_income,equity_ratio,current_ratio,cfo,cfi,cff\n"
    st.download_button(
        "テンプレDL",
        data=template_csv.encode("utf-8-sig"),
        file_name="template_financial_metrics.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_template",
    )

    st.caption("金額=円 / 比率=% or 0〜1")

    uploaded = st.file_uploader(
        "追加（複数可）",
        type=["csv"],
        accept_multiple_files=True,
        key=f"csv_uploader_in_main_{st.session_state.csv_uploader_key}",
    )

    if uploaded:
        added_n = 0
        combined_now = list(st.session_state.selected_codes) + [it["id"] for it in st.session_state.csv_items]
        capacity = max(0, MAX_COMPANIES - len(combined_now))

        for f in uploaded:
            if capacity <= 0:
                break

            # ★削除後に復活しない：追加済みファイル名は再追加しない
            if f.name in st.session_state.csv_uploaded_names:
                continue

            csv_id = f"CSV:{f.name}"

            try:
                d0 = _read_csv_uploaded(f)
                d1 = _normalize_csv_df(d0)

                d1 = d1.copy()
                d1["edinet_code"] = csv_id

                # 足りない列を補完（欠けてても動く）
                for col in ["sales", "net_income", "equity_ratio", "current_ratio", "cfo", "cfi", "cff", "cf_type"]:
                    if col not in d1.columns:
                        d1[col] = pd.NA

                d1 = fill_cf_type(d1)

                d1 = d1[["edinet_code", "fiscal_year", "sales", "net_income", "equity_ratio", "current_ratio", "cfo", "cfi", "cff", "cf_type"]]

                st.session_state.csv_items.append(
                    {"id": csv_id, "label": f"{f.name}（CSV）", "df": d1}
                )

                st.session_state.csv_uploaded_names.add(f.name)

                added_n += 1
                capacity -= 1

            except Exception as e:
                st.error(f"{f.name}: 取り込み失敗（{e}）")

        if added_n:
            st.success(f"{added_n}件 追加")

    # CSVだけ全削除（完全に消す：items + 履歴 + uploader）
    if st.session_state.csv_items:
        if st.button("CSV全削除", use_container_width=True, key="csv_clear_all", on_click=_clear_csv_only):
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# 比較中（DB+CSV 合算で最大8）
# =========================
db_labels_map = fetch_labels_for_codes(st.session_state.selected_codes)
csv_map = {it["id"]: {"name": it["label"], "label": it["label"]} for it in st.session_state.csv_items}
labels_map = {**db_labels_map, **csv_map}

combined_codes = list(st.session_state.selected_codes) + [it["id"] for it in st.session_state.csv_items]
ordered_codes = combined_codes[:MAX_COMPANIES]
ordered_display = [labels_map.get(c, {}).get("label", c) for c in ordered_codes]

st.markdown(
    f'<div class="panel"><div class="panel-title">比較中（最大{MAX_COMPANIES}件）</div>',
    unsafe_allow_html=True,
)

if not ordered_codes:
    st.info("比較リストが空です。上で企業を選ぶかCSVを追加してください。")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

for i, (code, disp) in enumerate(zip(ordered_codes, ordered_display), start=1):
    left, right = st.columns([18, 1])
    with left:
        st.markdown(
            f"""
            <div style="padding:6px 10px;margin:4px 0;border-radius:10px;
                        background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.25);font-size:13px;">
              <strong>{i}.</strong> {disp}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.button(
            "×",
            key=f"rm_{i}_{code}",
            help="この項目を比較から外す",
            on_click=_remove_from_compare,
            args=(code,),
        )

st.markdown('<div class="muted">※ 上から順に比較されます。</div>', unsafe_allow_html=True)
st.button("全解除", on_click=_clear_all)
st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Load & transform（DB+CSV）
# =========================
db_codes = [c for c in ordered_codes if not str(c).startswith("CSV:")]
df_db = load_metrics_yearly(db_codes)

csv_frames = []
for it in st.session_state.csv_items:
    if it["id"] in ordered_codes:
        csv_frames.append(it["df"])

df = df_db.copy()
if csv_frames:
    df_csv = pd.concat(csv_frames, ignore_index=True)
    df = pd.concat([df, df_csv], ignore_index=True)

if df.empty:
    st.error("比較データがありません（DBもCSVも空）。")
    st.stop()

# 期間フィルタ（stateから）
y_min, y_max = int(st.session_state.year_range[0]), int(st.session_state.year_range[1])
df = filter_years(df, y_min, y_max)

# CF型を保険で補完（DB側の欠損も埋める）
df = fill_cf_type(df)

# 単位変換
df["sales_oku"] = pd.to_numeric(df.get("sales"), errors="coerce") / 1e8
df["net_income_oku"] = pd.to_numeric(df.get("net_income"), errors="coerce") / 1e8
df["equity_ratio_pct"] = to_pct(df.get("equity_ratio"))
df["current_ratio_pct"] = to_pct(df.get("current_ratio"))
df["cfo_oku"] = pd.to_numeric(df.get("cfo"), errors="coerce") / 1e8
df["cfi_oku"] = pd.to_numeric(df.get("cfi"), errors="coerce") / 1e8
df["cff_oku"] = pd.to_numeric(df.get("cff"), errors="coerce") / 1e8

code_to_label = {c: labels_map.get(c, {}).get("label", c) for c in ordered_codes}
ordered_labels = [code_to_label[c] for c in ordered_codes]

# =========================
# 業界平均（DBのみ）
# =========================
use_sector = None if st.session_state.sector33_sel == "（全業界）" else st.session_state.sector33_sel
avg_label = None
avg_df = pd.DataFrame()

if st.session_state.show_avg and use_sector:
    avg_df = load_industry_avg_jpx33(use_sector, y_min, y_max)
    if not avg_df.empty:
        avg_label = f"業界平均（{use_sector}）"


def render_chart(title: str, value_col: str, unit: str, avg_key: str | None):
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)

    base = pivot_by_code(df, value_col, code_to_label=code_to_label)
    if not base.empty:
        base = base[[c for c in ordered_labels if c in base.columns]]

    if avg_label and avg_key and not avg_df.empty:
        base = add_avg_line(base, avg_df[avg_key], avg_label)

    st.line_chart(base)

    if st.session_state.show_analysis:
        with st.expander("このグラフからわかること", expanded=False):
            st.write("\n\n".join(make_analysis(base, unit=unit, avg_label=avg_label)))

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Charts
# =========================
st.markdown('<div class="section-title">収益性</div>', unsafe_allow_html=True)
cA, cB = st.columns(2)
with cA:
    render_chart("売上高（億円）", "sales_oku", "億円", "sales_oku")
with cB:
    render_chart("当期純利益（億円）", "net_income_oku", "億円", "net_income_oku")

st.markdown('<div class="section-title">安全性</div>', unsafe_allow_html=True)
cC, cD = st.columns(2)
with cC:
    render_chart("自己資本比率（％）", "equity_ratio_pct", "％", "equity_ratio_pct")
with cD:
    render_chart("流動比率（％）", "current_ratio_pct", "％", "current_ratio_pct")

st.markdown('<div class="section-title">キャッシュフロー</div>', unsafe_allow_html=True)
st.markdown('<div class="panel"><div class="panel-title">営業CF / 投資CF / 財務CF（億円）</div>', unsafe_allow_html=True)

cc1, cc2, cc3 = st.columns(3)
for title, key, avg_key, col in [
    ("営業CF（億円）", "cfo_oku", "cfo_oku", cc1),
    ("投資CF（億円）", "cfi_oku", "cfi_oku", cc2),
    ("財務CF（億円）", "cff_oku", "cff_oku", cc3),
]:
    with col:
        st.caption(title)
        base = pivot_by_code(df, key, code_to_label=code_to_label)
        if not base.empty:
            base = base[[c for c in ordered_labels if c in base.columns]]
        if avg_label and not avg_df.empty:
            base = add_avg_line(base, avg_df[avg_key], avg_label)
        st.line_chart(base)
        if st.session_state.show_analysis:
            with st.expander("このグラフからわかること", expanded=False):
                st.write("\n\n".join(make_analysis(base, unit="億円", avg_label=avg_label)))

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# キャッシュフロー表（企業ごと・展開済み）
# =========================
st.markdown('<div class="section-title">キャッシュフロー表（企業ごと）</div>', unsafe_allow_html=True)
st.markdown('<div class="panel"><div class="panel-title">CFO / CFI / CFF（億円）とCF型（年別）</div>', unsafe_allow_html=True)

cf_cols = ["fiscal_year", "cfo_oku", "cfi_oku", "cff_oku", "cf_type"]

for code in ordered_codes:
    disp = code_to_label.get(code, code)
    d1 = df[df["edinet_code"] == code].copy()
    if d1.empty:
        continue

    d1 = d1.sort_values("fiscal_year")[cf_cols].rename(
        columns={
            "fiscal_year": "年度",
            "cfo_oku": "営業CF（億円）",
            "cfi_oku": "投資CF（億円）",
            "cff_oku": "財務CF（億円）",
            "cf_type": "CF型",
        }
    )

    with st.expander(f"{disp}", expanded=True):
        st.dataframe(d1, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 生データ
# =========================
if st.session_state.show_raw:
    st.markdown('<div class="section-title">生データ（DB+CSV）</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">結合データ（抽出範囲）</div>', unsafe_allow_html=True)
    st.dataframe(df.sort_values(["edinet_code", "fiscal_year"]), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Download
# =========================
st.markdown('<div class="panel"><div class="panel-title">ダウンロード</div>', unsafe_allow_html=True)
st.download_button(
    "比較データ（DB+CSV）をCSVでダウンロード",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="compare_db_plus_csv.csv",
    mime="text/csv",
)
st.markdown(
    f'<div class="muted">※ 選択（最大{MAX_COMPANIES}件：DB＋CSV合算）・選択期間（{Y_MIN_FIXED}〜{Y_MAX_FIXED}）のデータを出力します。</div></div>',
    unsafe_allow_html=True,
)
