
# embedslr/colab_app.py
from __future__ import annotations

"""
Colab mini-GUI for SMART multi-criteria selection.

Usage in Colab after installing the repo:
    !pip install -U "git+https://github.com/s-matysik/EmbedSLR_v2.git"
    from embedslr.colab_app import run
    run()

The GUI:
 - uses only EXISTING metrics (no recomputation),
 - lets users map columns and set weights (ranks 4–10 or direct),
 - saves and offers to download SMART_result.csv.

Dependencies: ipywidgets, pandas (install separately if not present).
"""

import math
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from IPython.display import display, clear_output
import ipywidgets as W

# Prefer package-internal SMART implementation
try:
    from .smart_mcdm_biblio import (
        SMARTConfig, rank_with_smart_biblio, read_candidates
    )
except Exception as e:  # pragma: no cover - fallback shouldn't happen if installed properly
    raise ImportError(
        "embedslr.smart_mcdm_biblio not available. "
        "Install EmbedSLR_v2 with SMART module or update your package."
    ) from e


def _none_if_blank(x: Optional[str]) -> Optional[str]:
    if x is None or x.strip() == "" or x == "— brak —":
        return None
    return x


def run() -> None:
    # Optional: enable widget manager in Colab
    try:
        from google.colab import output  # type: ignore
        output.enable_custom_widget_manager()
    except Exception:
        pass

    hdr = W.HTML("<h3>SMART – wybór wielokryterialny (Colab GUI)</h3>")
    uploader = W.FileUpload(accept=".csv,.tsv,.txt,.xlsx,.xls,.parquet,.feather", multiple=False)
    btn_load  = W.Button(description="Wczytaj plik", button_style="primary", icon="upload")
    lbl_file  = W.HTML("<i>Wybierz plik i kliknij 'Wczytaj plik'.</i>")

    # Column selectors
    ddl_sem = W.Dropdown(options=[], description="Semantyka:", disabled=True)
    ddl_kw  = W.Dropdown(options=[], description="Sł. kluczowe:", disabled=True)
    ddl_ref = W.Dropdown(options=[], description="Referencje:", disabled=True)
    ddl_mut = W.Dropdown(options=[], description="Wzajemne cyt.:", disabled=True)

    ch_sem_is_dist = W.Checkbox(value=False, description="Semantyka to 'distance' (odwróć)", disabled=True)
    ddl_norm = W.Dropdown(options=[("minmax","minmax"),("max","max")], value="minmax", description="Normalizacja:")
    chk_scale = W.Checkbox(value=False, description="Agregacja na skali 4–10")
    chk_avail = W.Checkbox(value=True, description="Pomiń brakujące kryteria")

    weight_mode = W.ToggleButtons(options=[("Rangi 4–10","ranks"),("Wagi bezp.","weights")], value="ranks", description="Tryb wag:")

    # Rank sliders
    sl_sem_r = W.IntSlider(min=4, max=10, value=8, step=1, description="rank(seman.)")
    sl_kw_r  = W.IntSlider(min=4, max=10, value=7, step=1, description="rank(keyw.)")
    sl_ref_r = W.IntSlider(min=4, max=10, value=7, step=1, description="rank(ref.)")
    sl_mut_r = W.IntSlider(min=4, max=10, value=6, step=1, description="rank(mutual)")

    # Weight sliders
    sl_sem_w = W.FloatSlider(min=0, max=1, value=0.40, step=0.01, description="w(seman.)", readout_format=".2f")
    sl_kw_w  = W.FloatSlider(min=0, max=1, value=0.25, step=0.01, description="w(keyw.)",  readout_format=".2f")
    sl_ref_w = W.FloatSlider(min=0, max=1, value=0.20, step=0.01, description="w(ref.)",   readout_format=".2f")
    sl_mut_w = W.FloatSlider(min=0, max=1, value=0.15, step=0.01, description="w(mutual)", readout_format=".2f")
    lbl_wsum = W.HTML("Suma wag (zostanie znormalizowana): 1.00")

    box_ranks  = W.VBox([sl_sem_r, sl_kw_r, sl_ref_r, sl_mut_r])
    box_weights= W.VBox([sl_sem_w, sl_kw_w, sl_ref_w, sl_mut_w, lbl_wsum])

    def _update_weight_sum(*_):
        s = sl_sem_w.value + sl_kw_w.value + sl_ref_w.value + sl_mut_w.value
        lbl_wsum.value = f"Suma wag (zostanie znormalizowana): <b>{s:.2f}</b>"

    sl_sem_w.observe(_update_weight_sum, "value")
    sl_kw_w.observe(_update_weight_sum, "value")
    sl_ref_w.observe(_update_weight_sum, "value")
    sl_mut_w.observe(_update_weight_sum, "value")

    def _toggle_weight_mode(*_):
        if weight_mode.value == "ranks":
            box_ranks.layout.display = "block"
            box_weights.layout.display = "none"
        else:
            box_ranks.layout.display = "none"
            box_weights.layout.display = "block"

    weight_mode.observe(_toggle_weight_mode, "value")
    _toggle_weight_mode()

    topN = W.BoundedIntText(min=1, max=1_000_000, value=100, description="TOP-N:")
    btn_run = W.Button(description="Uruchom SMART", button_style="success", icon="cogs")
    btn_prev= W.Button(description="Podgląd danych", icon="eye")
    out = W.Output()

    # State
    UPLOAD_DIR = Path("/content/uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _state: Dict[str, object] = {"df": None, "path": None}

    def _write_upload_to_disk(uploader: W.FileUpload) -> str | None:
        if not uploader.value:
            return None
        meta = next(iter(uploader.value.values()))
        name = meta["metadata"]["name"]
        content = meta["content"]
        path = UPLOAD_DIR / name
        with open(path, "wb") as f:
            f.write(content)
        return str(path)

    def _guess(df: pd.DataFrame, cands: list[str]) -> str | None:
        for c in cands:
            if c in df.columns: 
                return c
        return None

    def on_load_clicked(_b):
        with out:
            clear_output()
            try:
                path = _write_upload_to_disk(uploader)
                if not path:
                    print("Nie wybrano pliku.")
                    return
                df = read_candidates(path)
                _state["df"] = df
                _state["path"] = path
                print(f"✓ Wczytano: {Path(path).name}  |  wiersze: {len(df)}, kolumny: {len(df.columns)}")

                opts = ["— brak —"] + list(df.columns)
                for w in (ddl_sem, ddl_kw, ddl_ref, ddl_mut):
                    w.options = opts
                    w.disabled = False

                sem_guess = _guess(df, ["semantic_similarity","cosine_similarity","similarity","cos_sim",
                                        "distance_cosine","cosine_distance","semantic_dist"])
                kw_guess  = _guess(df, ["kw_similarity","author_keywords_similarity","kw_jaccard","keyword_overlap_score"])
                ref_guess = _guess(df, ["bibliographic_coupling","biblio_coupling","bc_score","common_references","co_citation_score"])
                mut_guess = _guess(df, ["mutual_citations","reciprocal_citations","two_way_citations","mutual_citations_count"])

                ddl_sem.value = sem_guess or "— brak —"
                ddl_kw.value  = kw_guess  or "— brak —"
                ddl_ref.value = ref_guess or "— brak —"
                ddl_mut.value = mut_guess or "— brak —"

                ch_sem_is_dist.disabled = False
                ch_sem_is_dist.value = (sem_guess is not None) and (("dist" in sem_guess.lower()) or ("distance" in sem_guess.lower()))

                print("\nDomyślne mapowanie (możesz zmienić):")
                print("  semantic   ->", ddl_sem.value)
                print("  keywords   ->", ddl_kw.value)
                print("  references ->", ddl_ref.value)
                print("  mutual     ->", ddl_mut.value)

            except Exception as e:
                print("Błąd wczytywania:", e)

    btn_load.on_click(on_load_clicked)

    def on_prev_clicked(_b):
        with out:
            clear_output()
            df = _state["df"]
            if df is None:
                print("Najpierw wczytaj plik.")
                return
            display(df.head(10))

    btn_prev.on_click(on_prev_clicked)

    def on_run_clicked(_b):
        with out:
            clear_output()
            df = _state["df"]
            if df is None:
                print("Najpierw wczytaj plik.")
                return

            colmap = {
                "semantic": _none_if_blank(ddl_sem.value),
                "keywords": _none_if_blank(ddl_kw.value),
                "references": _none_if_blank(ddl_ref.value),
                "mutual": _none_if_blank(ddl_mut.value),
            }

            explicit = None
            ranks    = None
            if weight_mode.value == "weights":
                explicit = {
                    "semantic": float(sl_sem_w.value),
                    "keywords": float(sl_kw_w.value),
                    "references": float(sl_ref_w.value),
                    "mutual": float(sl_mut_w.value),
                }
            else:
                ranks = {
                    "semantic": int(sl_sem_r.value),
                    "keywords": int(sl_kw_r.value),
                    "references": int(sl_ref_r.value),
                    "mutual": int(sl_mut_r.value),
                }

            cfg = SMARTConfig(
                column_map=colmap,
                explicit_weights=explicit,
                importance_ranks=ranks or {'semantic':8,'keywords':7,'references':7,'mutual':6},
                scale_4to10=bool(chk_scale.value),
                available_only=bool(chk_avail.value),
                normalize_strategy=str(ddl_norm.value),
                semantic_is_distance=bool(ch_sem_is_dist.value),
                verbose=True
            )

            res = rank_with_smart_biblio(df, cfg, top_n=int(topN.value))

            # Output
            weights = res.weights
            used_cols = res.used_columns
            dropped = res.dropped_criteria

            print("Wagi (po normalizacji):", weights)
            print("Użyte kolumny:", used_cols)
            if dropped:
                print("Pominięto kryteria:", dropped)

            display(res.df.head(20))

            out_path = "/content/SMART_result.csv"
            res.df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"✓ Zapisano: {out_path}")
            try:
                from google.colab import files as _files  # type: ignore
                _files.download(out_path)
            except Exception:
                print("Pobierz ręcznie plik /content/SMART_result.csv")

    btn_run.on_click(on_run_clicked)

    # Layout
    box_top = W.HBox([uploader, btn_load, btn_prev])
    box_cols = W.VBox([ddl_sem, ddl_kw, ddl_ref, ddl_mut, ch_sem_is_dist])
    box_opts = W.VBox([ddl_norm, chk_scale, chk_avail, weight_mode, box_ranks, box_weights, topN, btn_run])

    app = W.VBox([hdr, lbl_file, box_top, W.HTML("<hr/>"), W.HTML("<b>Mapowanie kolumn</b>"),
                  box_cols, W.HTML("<b>Ustawienia</b>"), box_opts, W.HTML("<hr/>"), out])
    display(app)
