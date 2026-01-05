import pandas as pd

def build_event_strings(
    df: pd.DataFrame,
    cols_map: dict | None = None,
    out_cols: list[str] | None = None,
    group_col: str = "case_id",
    include: list[str] | None = None,
    sort_groups: bool = False,
) -> tuple[pd.DataFrame, list[str]]:

    if cols_map is None:
        cols_map = {
            "Case": "case_id",
            "Activity": "activity",
            "Timestamp": "timestamp",
            "Injection": "pattern",
            "org": "corrected",
        }
    if out_cols is None:
        out_cols = ["event_id", "case_id", "timestamp", "activity", "pattern", "corrected"]

    df_new = df[list(cols_map.keys())].rename(columns=cols_map).copy()
    df_new.insert(0, "event_id", range(1, len(df_new) + 1))
    df_new = df_new[out_cols]

    if include is None:
        include = out_cols

    def is_empty(v):
        return pd.isna(v) or v == ""

    cases = []
    for _, g in df_new.groupby(group_col, sort=sort_groups):
        lines = []
        for _, row in g.iterrows():
            pattern_empty = "pattern" in row and is_empty(row["pattern"])
            parts = []
            for k in include:
                v = row[k]
                if k == "corrected" and pattern_empty:
                    v = ""
                if pd.isna(v):
                    v = ""
                if k in ("event_id", "case_id"):
                    parts.append(f"{k}={v}")
                else:
                    parts.append(f'{k}="{v}"')
            lines.append("- " + " | ".join(parts))
        cases.append("\n".join(lines))

    return df_new, cases
    
def build_event_jsons(
    df: pd.DataFrame,
    cols_map: dict | None = None,
    out_cols: list[str] | None = None,
    group_col: str = "case_id",
    include: list[str] | None = None,
    sort_groups: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:

    if cols_map is None:
        cols_map = {
            "Case": "case_id",
            "Activity": "activity",
            "Timestamp": "timestamp",
            "Injection": "pattern",
            "org": "corrected",
        }
    if out_cols is None:
        out_cols = ["event_id", "case_id", "timestamp", "activity", "pattern", "corrected"]

    df_new = df[list(cols_map.keys())].rename(columns=cols_map).copy()
    df_new.insert(0, "event_id", range(1, len(df_new) + 1))
    df_new = df_new[out_cols]

    if include is None:
        include = ["event_id", "case_id", "timestamp", "activity"]  # 기본은 네가 쓰는 4개 추천

    def is_empty(v):
        return pd.isna(v) or v == ""

    cases = []
    for case_id, g in df_new.groupby(group_col, sort=sort_groups):
        events = []
        for _, row in g.iterrows():
            pattern_empty = ("pattern" in row) and is_empty(row["pattern"])

            e = {}
            for k in include:
                v = row[k]

                # 기존 로직 유지: pattern이 비어 있으면 corrected는 빈 문자열로
                if k == "corrected" and pattern_empty:
                    v = ""

                if pd.isna(v):
                    v = ""

                # 타입 강제: event_id, case_id는 int로 (LLM에 중요)
                if k in ("event_id", "case_id") and v != "":
                    try:
                        v = int(v)
                    except Exception:
                        pass

                e[k] = v

            events.append(e)

        cases.append({"case_id": int(case_id), "events": events})

    return df_new, cases