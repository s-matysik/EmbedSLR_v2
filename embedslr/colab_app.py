
# embedslr/colab_app.py
from __future__ import annotations

"""
Colab mini-GUI for EmbedSLR: Embeddings + SMART (MCDM)

Usage in Colab after installing the repo:
    !pip -q install -U "git+https://github.com/s-matysik/EmbedSLR_v2.git"
    from embedslr.colab_app import run
    run()

What it does:
 1) Lets you load a dataset (CSV/TSV/XLSX/Parquet/Feather).
 2) (Optional) Computes semantic embeddings for the query and documents
    using `embedslr.embeddings.get_embeddings` and produces a
    `distance_cosine` column (lower=better).
 3) Runs SMART aggregation on existing metrics + the semantic distance
    (or similarity) as one of the criteria.
 4) Saves and offers to download SMART_result.csv.

Notes:
 - Bibliometric metrics are NOT recomputed here. The app only uses the columns
   already present in the dataset (keywords similarity, coupling, mutual citations, ...).
 - Embeddings depend on the chosen provider/model. For `sbert` you may need:
     pip install -U sentence-transformers
 - OpenAI/Cohere/Nomic/Jina providers require API keys in environment variables,
   as expected by embedslr.embeddings.
"""

from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
from IPython.display import display, clear_output
import ipywidgets as W

from .io import autodetect_columns, combine_title_abstract
from .similarity import rank_by_cosine
from .smart_mcdm_biblio import SMARTConfig, rank_with_smart_biblio, read_candidates
from .embeddings import get_embeddings, list_models


def _none_if_blank(x: Optional[str]) -> Optional[str]:
    if x is None or str(x).strip() == "" or x == "— brak —":
        return None
    return str(x)


def run() -> None:
    # Enable widget manager in Colab (no-op elsewhere)
    try:
        from google.colab import output  # type: ignore
        output.enable_custom_widget_manager()
    except Exception:
        pass

    # ─────────────── UI widgets ───────────────
    hdr = W.HTML("<h3>EmbedSLR Colab App — Embeddings + SMART</h3>")
    uploader = W.FileUpload(accept=".csv,.tsv,.txt,.xlsx,.xls,.parquet,.feather", multiple=False)
    btn_load  = W.Button(description="Wczytaj plik", button_style="primary", icon="upload")
    lbl_file  = W.HTML("<i>Wybierz plik i kliknij 'Wczytaj plik'.</i>")
    out = W.Output()

    # 1) Embedding section
    box_emb_hdr = W.HTML("<b>1) Ustawienia embeddingów (opcjonalne)</b>")
    txt_query = W.Textarea(placeholder="Wpisz zapytanie (query) do embeddingu...", description="Query:")
    btn_guess_cols = W.Button(description="Auto‑wybór kolumn Tytuł/Abstrakt", icon="magic")
    ddl_title = W.Dropdown(options=[], description="Tytuł:", disabled=True)
    ddl_abs   = W.Dropdown(options=[], description="Abstrakt:", disabled=True)
    ch_combine = W.Checkbox(value=True, description="Tytuł + Abstrakt (zalecane)")

    # Provider/model
    providers = ["sbert", "openai", "cohere", "nomic", "jina"]
    ddl_provider = W.Dropdown(options=providers, value="sbert", description="Provider:")
    # models list (will be filled dynamically)
    ddl_model = W.Dropdown(options=[], description="Model:")
    btn_refresh_models = W.Button(description="Odśwież modele", icon="refresh")
    ch_compute_emb = W.Checkbox(value=True, description="Wylicz embeddings i kolumnę distance_cosine")
    lim_docs = W.BoundedIntText(min=0, max=10_000_000, value=0, description="Limit dokumentów (0=wszystkie):")

    # 2) SMART section
    box_smart_hdr = W.HTML("<b>2) Wybór wielokryterialny (SMART)</b>")
    ddl_sem = W.Dropdown(options=[], description="Semantyka:", disabled=True)
    ddl_kw  = W.Dropdown(options=[], description="Sł. kluczowe:", disabled=True)
    ddl_ref = W.Dropdown(options=[], description="Referencje:", disabled=True)
    ddl_mut = W.Dropdown(options=[], description="Wzajemne cyt.:", disabled=True)
    ch_sem_is_dist = W.Checkbox(value=True, description="Semantyka to 'distance' (odwróć)", disabled=True)

    ddl_norm = W.Dropdown(options=[("minmax","minmax"),("max","max")], value="minmax", description="Normalizacja:")
    chk_scale = W.Checkbox(value=False, description="Agregacja na skali 4–10")
    chk_avail = W.Checkbox(value=True, description="Pomiń brakujące kryteria")
    weight_mode = W.ToggleButtons(options=[("Rangi 4–10","ranks"),("Wagi bezp.","weights")], value="ranks", description="Tryb wag:")

    sl_sem_r = W.IntSlider(min=4, max=10, value=8, step=1, description="rank(seman.)")
    sl_kw_r  = W.IntSlider(min=4, max=10, value=7, step=1, description="rank(keyw.)")
    sl_ref_r = W.IntSlider(min=4, max=10, value=7, step=1, description="rank(ref.)")
    sl_mut_r = W.IntSlider(min=4, max=10, value=6, step=1, description="rank(mutual)")

    sl_sem_w = W.FloatSlider(min=0, max=1, value=0.40, step=0.01, description="w(seman.)", readout_format=".2f")
    sl_kw_w  = W.FloatSlider(min=0, max=1, value=0.25, step=0.01, description="w(keyw.)",  readout_format=".2f")
    sl_ref_w = W.FloatSlider(min=0, max=1, value=0.20, step=0.01, description="w(ref.)",   readout_format=".2f")
    sl_mut_w = W.FloatSlider(min=0, max=1, value=0.15, step=0.01, description="w(mutual)", readout_format=".2f")
    lbl_wsum = W.HTML("Suma wag (zostanie znormalizowana): 1.00")

    def _update_weight_sum(*_):
        s = sl_sem_w.value + sl_kw_w.value + sl_ref_w.value + sl_mut_w.value
        lbl_wsum.value = f"Suma wag (zostanie znormalizowana): <b>{s:.2f}</b>"

    sl_sem_w.observe(_update_weight_sum, "value")
    sl_kw_w.observe(_update_weight_sum, "value")
    sl_ref_w.observe(_update_weight_sum, "value")
    sl_mut_w.observe(_update_weight_sum, "value")

    def _toggle_weight_mode(*_):
        ranks_vis = "block" if weight_mode.value == "ranks" else "none"
        weights_vis = "none" if weight_mode.value == "ranks" else "block"
        box_ranks.layout.display = ranks_vis
        box_weights.layout.display = weights_vis

    box_ranks  = W.VBox([sl_sem_r, sl_kw_r, sl_ref_r, sl_mut_r])
    box_weights= W.VBox([sl_sem_w, sl_kw_w, sl_ref_w, sl_mut_w, lbl_wsum])
    _toggle_weight_mode()
    weight_mode.observe(_toggle_weight_mode, "value")

    topN = W.BoundedIntText(min=1, max=10_000_000, value=100, description="TOP-N:")
    btn_run = W.Button(description="Uruchom (Embeddings → SMART)", button_style="success", icon="cogs")
    btn_prev= W.Button(description="Podgląd danych", icon="eye")

    # State
    state: Dict[str, object] = {"df": None, "path": None, "title_col": None, "abs_col": None}

    # ---- helpers ----
    def _write_upload_to_disk(upl: W.FileUpload) -> Optional[str]:
        if not upl.value:
            return None
        meta = next(iter(upl.value.values()))
        name = meta["metadata"]["name"]
        content = meta["content"]
        updir = Path("/content/uploads"); updir.mkdir(parents=True, exist_ok=True)
        path = updir / name
        with open(path, "wb") as f:
            f.write(content)
        return str(path)

    def _fill_models(*_):
        try:
            models_dict = list_models()
        except Exception as e:
            # Static fallback for SBERT
            models_dict = {
                "sbert": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/distiluse-base-multilingual-cased-v2",
                ],
                "openai": ["text-embedding-3-small","text-embedding-3-large"],
                "cohere": ["embed-english-v3.0","embed-multilingual-v3.0"],
                "nomic": ["nomic-embed-text-v1"],
                "jina": ["jina-embeddings-v2-base-en","jina-embeddings-v3"]
            }
        prov = ddl_provider.value
        ddl_model.options = models_dict.get(prov, models_dict.get("sbert", []))
        if ddl_model.options:
            ddl_model.value = ddl_model.options[0]

    def _guess_title_abs(df: pd.DataFrame):
        try:
            title_col, abs_col = autodetect_columns(df)
        except Exception:
            # fallback: any string columns
            str_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
            title_col = str_cols[0] if str_cols else None
            abs_col = (str_cols[1] if len(str_cols) > 1 else None)
        return title_col, abs_col

    # ---- callbacks ----
    def on_load_clicked(_b):
        with out:
            clear_output()
            try:
                path = _write_upload_to_disk(uploader)
                if not path:
                    print("Nie wybrano pliku.")
                    return
                df = read_candidates(path)
                state["df"] = df
                state["path"] = path
                print(f"✓ Wczytano: {Path(path).name}  |  wiersze: {len(df)}, kolumny: {len(df.columns)}")

                # Fill column dropdowns
                opts = ["— brak —"] + list(df.columns)
                for w in (ddl_sem, ddl_kw, ddl_ref, ddl_mut, ddl_title, ddl_abs):
                    w.options = opts
                    w.disabled = False

                # Guess title/abstract
                tcol, acol = _guess_title_abs(df)
                state["title_col"], state["abs_col"] = tcol, acol
                ddl_title.value = tcol or "— brak —"
                ddl_abs.value = acol or "— brak —"

                # Guess semantic column
                sem_guess = None
                for cand in ["distance_cosine","cosine_distance","semantic_similarity","cosine_similarity","similarity"]:
                    if cand in df.columns:
                        sem_guess = cand; break
                ddl_sem.value = sem_guess or "— brak —"
                ch_sem_is_dist.disabled = False
                ch_sem_is_dist.value = (sem_guess or "").lower().find("dist") != -1 or (sem_guess or "").lower().find("distance") != -1

                # Other columns
                for cands, widget in [
                    (["kw_similarity","author_keywords_similarity","kw_jaccard","keyword_overlap_score"], ddl_kw),
                    (["bibliographic_coupling","biblio_coupling","bc_score","common_references","co_citation_score"], ddl_ref),
                    (["mutual_citations","reciprocal_citations","two_way_citations","mutual_citations_count"], ddl_mut),
                ]:
                    for c in cands:
                        if c in df.columns:
                            widget.value = c; break

                print("Mapowanie (możesz zmienić):")
                print("  semantic   ->", ddl_sem.value)
                print("  keywords   ->", ddl_kw.value)
                print("  references ->", ddl_ref.value)
                print("  mutual     ->", ddl_mut.value)

                _fill_models()

            except Exception as e:
                print("Błąd wczytywania:", e)

    btn_load.on_click(on_load_clicked)
    btn_refresh_models.on_click(lambda _b: _fill_models())

    def on_guess_cols(_b):
        df = state["df"]
        if df is None:
            with out:
                clear_output()
                print("Najpierw wczytaj plik.")
            return
        tcol, acol = _guess_title_abs(df)
        state["title_col"], state["abs_col"] = tcol, acol
        ddl_title.value = tcol or "— brak —"
        ddl_abs.value = acol or "— brak —"
    btn_guess_cols.on_click(on_guess_cols)

    def on_prev_clicked(_b):
        with out:
            clear_output()
            df = state["df"]
            if df is None:
                print("Najpierw wczytaj plik.")
                return
            display(df.head(10))

    btn_prev.on_click(on_prev_clicked)

    def on_run_clicked(_b):
        with out:
            clear_output()
            df = state["df"]
            if df is None:
                print("Najpierw wczytaj plik.")
                return

            # 1) Embeddings (optional)
            if ch_compute_emb.value:
                q = txt_query.value.strip()
                if not q:
                    print("Podaj tekst zapytania (query) dla embeddingu.")
                    return
                prov = ddl_provider.value
                model = ddl_model.value
                # build doc texts
                title_col = _none_if_blank(ddl_title.value) or state["title_col"]
                abs_col   = _none_if_blank(ddl_abs.value) or state["abs_col"]

                if not title_col and not abs_col:
                    print("Nie wskazano kolumn tytułu/abstraktu.")
                    return

                if ch_combine.value and title_col and abs_col:
                    texts = combine_title_abstract(df, title_col, abs_col).fillna("").astype(str).tolist()
                else:
                    col = title_col or abs_col
                    texts = df[col].fillna("").astype(str).tolist()

                # limit docs?
                lim = int(lim_docs.value or 0)
                if lim and lim < len(texts):
                    texts = texts[:lim]
                    df = df.iloc[:lim].copy()

                # compute embeddings
                try:
                    q_vec = get_embeddings([q], provider=prov, model=model)[0]
                    d_vecs = get_embeddings(texts, provider=prov, model=model)
                except Exception as e:
                    print("Błąd embeddingu:", e)
                    print("Wskazówka: dla provider='sbert' zainstaluj 'sentence-transformers', "
                          "dla OpenAI/Cohere/Nomic/Jina ustaw odpowiednie klucze API w środowisku.")
                    return

                # add distance_cosine via helper
                df = rank_by_cosine(q_vec, d_vecs, df)
                # prefer semantic 'distance_cosine' as default
                ddl_sem.value = "distance_cosine"
                ch_sem_is_dist.value = True

                print(f"✓ Obliczono embeddings ({prov}:{model}) i kolumnę 'distance_cosine'.")

            # 2) SMART
            def _none_if_noopt(x):
                return None if (x is None or x == "— brak —") else x

            colmap = {
                "semantic": _none_if_noopt(ddl_sem.value),
                "keywords": _none_if_noopt(ddl_kw.value),
                "references": _none_if_noopt(ddl_ref.value),
                "mutual": _none_if_noopt(ddl_mut.value),
            }

            explicit = None
            ranks = None
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

            print("Wagi (po normalizacji):", res.weights)
            print("Użyte kolumny:", res.used_columns)
            if res.dropped_criteria:
                print("Pominięto kryteria:", res.dropped_criteria)

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
    box_emb = W.VBox([box_emb_hdr, txt_query, W.HBox([btn_guess_cols]), ddl_title, ddl_abs,
                      ch_combine, W.HBox([ddl_provider, ddl_model, btn_refresh_models]),
                      ch_compute_emb, lim_docs])
    box_cols = W.VBox([box_smart_hdr, ddl_sem, ddl_kw, ddl_ref, ddl_mut, ch_sem_is_dist])
    box_opts = W.VBox([ddl_norm, chk_scale, chk_avail, weight_mode, box_ranks, box_weights, topN, btn_run])

    app = W.VBox([hdr, lbl_file, box_top, W.HTML("<hr/>"), box_emb,
                  W.HTML("<hr/>"), W.HTML("<b>Mapowanie i wagi SMART</b>"),
                  box_cols, box_opts, W.HTML("<hr/>"), out])
    display(app)
