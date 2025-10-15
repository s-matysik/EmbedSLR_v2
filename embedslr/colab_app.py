from __future__ import annotations
"""
embedslr.colab_app

Interaktywny interfejs do uruchamiania EmbedSLR w Google Colab wyłącznie za pomocą:
    from embedslr.colab_app import run
    run()

Funkcje:
- Upload pliku z danymi (CSV/TSV) z Colab file picker
- Konfiguracja metod: L-Scoring, Z-Scoring, L-Scoring+
- Ustawienia wag, Top-K (słowa kluczowe i referencje), kar, bonusów
- (Opcjonalnie) Wyliczenie 'semantic_similarity' z zapytania (Sentence-Transformers)
- Ranking + raporty częstości (słowa, referencje) + przyciski pobierania

Wymaga:
- pandas, numpy, ipywidgets (domyślnie dostępne w Colab)
- sentence-transformers: tylko jeśli w UI zaznaczysz "Policz semantic_similarity z zapytania"

Zaimplementowano wg wymagań z „Update do SoftX 1”. 
"""

import io
import os
import sys
import json
import math
import textwrap
from typing import Optional, Dict

import numpy as np
import pandas as pd

from IPython.display import display, clear_output, HTML

# ipywidgets i Colab pliki
try:
    import ipywidgets as W
    _HAVE_WIDGETS = True
except Exception:
    _HAVE_WIDGETS = False

try:
    from google.colab import files as colab_files
    _IN_COLAB = True
except Exception:
    _IN_COLAB = False

# Nowe moduły scoringu
from .advanced_scoring import rank_with_advanced_scoring
from .config import ScoringConfig, ColumnMap


# --------------------------- Pomocnicze ----------------------------------

def _detect_sep(name: str, head: bytes) -> str:
    """Prosta heurystyka detekcji separatora."""
    n = head.decode("utf-8", errors="ignore")
    if name.lower().endswith(".tsv"):
        return "\t"
    if name.lower().endswith(".csv"):
        # CSV bywa też średnikowe
        score = {",": n.count(","), ";": n.count(";"), "\t": n.count("\t")}
        return max(score, key=score.get)
    # auto na podstawie liczebności
    score = {",": n.count(","), ";": n.count(";"), "\t": n.count("\t")}
    best = max(score, key=score.get)
    return best if score[best] > 0 else ","


def _safe_read_csv(name: str, raw: bytes, sep_opt: str = "auto") -> pd.DataFrame:
    """Wczytaj CSV/TSV z bezpiecznymi fallbackami kodowania i separatora."""
    head = raw[:65536]
    sep = _detect_sep(name, head) if sep_opt == "auto" else sep_opt

    for enc in ("utf-8", "utf-8-sig", "cp1250", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc)
        except Exception:
            continue
    # ostatnia próba bez deklaracji (pandas sam przetestuje)
    return pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")


def _combine_title_abstract(row: pd.Series, df_cols: list[str]) -> str:
    """Złóż tekst tytuł + abstrakt z typowych kolumn."""
    title_cols = ["Title", "Article Title", "Document Title", "TI"]
    abs_cols   = ["Abstract", "AB", "Description", "Abstract Note"]
    parts = []
    for c in title_cols:
        if c in df_cols and isinstance(row.get(c), str) and row[c].strip():
            parts.append(row[c])
            break
    for c in abs_cols:
        if c in df_cols and isinstance(row.get(c), str) and row[c].strip():
            parts.append(row[c])
            break
    return ". ".join(parts) if parts else ""


def _cosine_similarity_to_query(texts: list[str], query: str) -> np.ndarray:
    """
    Policz podobieństwo kosinusowe do zapytania za pomocą Sentence-Transformers (lekki model).
    Instalacja i pobranie modelu następuje na żądanie.
    """
    # On-demand instalacja
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_q  = model.encode([query], show_progress_bar=False)
    emb_d  = model.encode(texts, show_progress_bar=True, batch_size=64)
    q = emb_q[0].astype(np.float32)
    D = np.asarray(emb_d, dtype=np.float32)
    # kosinus bez sklearn
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    D_norm = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    sim = D_norm @ q_norm
    return sim


def _nice_html(msg: str) -> HTML:
    return HTML(f"""<div style="padding:10px;border:1px solid #ddd;border-radius:6px;background:#fafafa">{msg}</div>""")


# -------------------------- Klasa UI -------------------------------------

class _ColabApp:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.dataset_name: Optional[str] = None
        self.result = None  # ScoringResult
        self.output_dir = "/content" if _IN_COLAB else "."

        # --------- WIDGETS: plik ----------
        self.lbl_title = W.HTML("<h2>EmbedSLR – interfejs Google Colab</h2>")
        self.btn_upload = W.Button(description="Wgraj plik danych (CSV/TSV)", button_style="primary")
        self.dd_sep = W.Dropdown(options=[("auto", "auto"), (", (comma)", ","), ("; (semicolon)", ";"), ("\\t (tab)", "\\t")],
                                 value="auto", description="Separator:")
        self.out_file = W.Output(layout={"border": "1px solid #ddd"})

        # --------- WIDGETS: konfiguracja ----------
        self.dd_method = W.Dropdown(
            options=[("L‑Scoring (linear)", "linear"),
                     ("Z‑Scoring (standaryzacja)", "zscore"),
                     ("L‑Scoring+ (linear + bonusy)", "linear_plus")],
            value="linear_plus", description="Metoda:"
        )
        self.sl_topk_kw = W.IntSlider(description="Top‑K słów", min=1, max=25, step=1, value=5, continuous_update=False)
        self.sl_topk_ref = W.IntSlider(description="Top‑K referencji", min=1, max=50, step=1, value=15, continuous_update=False)
        self.sl_pen_kw = W.FloatSlider(description="Kara brak słów", min=0.0, max=0.5, step=0.01, value=0.10, readout_format=".2f")
        self.sl_pen_ref = W.FloatSlider(description="Kara brak ref.", min=0.0, max=0.5, step=0.01, value=0.10, readout_format=".2f")

        self.sl_w_sem = W.FloatSlider(description="Waga: semantyka", min=0.0, max=1.0, step=0.01, value=0.40, readout_format=".2f")
        self.sl_w_kw  = W.FloatSlider(description="Waga: słowa",     min=0.0, max=1.0, step=0.01, value=0.25, readout_format=".2f")
        self.sl_w_ref = W.FloatSlider(description="Waga: referencje",min=0.0, max=1.0, step=0.01, value=0.25, readout_format=".2f")
        self.sl_w_cit = W.FloatSlider(description="Waga: cytowania", min=0.0, max=1.0, step=0.01, value=0.10, readout_format=".2f")
        self.lbl_wsum = W.HTML()

        self.tf_bonus_start = W.FloatText(description="Bonus start Z", value=2.0)
        self.tf_bonus_full  = W.FloatText(description="Bonus pełny Z", value=4.0)
        self.tf_bonus_cap   = W.IntText(description="Limit bonusów (pkt)", value=0)  # 0 == auto == P

        self.cb_save_freq = W.Checkbox(description="Zapisz raporty częstości (CSV)", value=True)

        # (opcjonalnie) podobieństwo z zapytania
        self.cb_sem_query = W.Checkbox(description="Policz semantic_similarity z zapytania", value=False)
        self.ta_query     = W.Textarea(description="Zapytanie:", value="", placeholder="Wpisz treść pytania badawczego...")
        self.info_query   = W.HTML('<small>Uwaga: doliczenie podobieństwa wymaga pobrania lekkiego modelu (Sentence‑Transformers).</small>')

        # (opcjonalnie) mapowanie kolumn
        self.tf_col_kw  = W.Text(description="Kolumna słów (opcjonalnie)", placeholder="np. Author Keywords")
        self.tf_col_ref = W.Text(description="Kolumna ref. (opcjonalnie)", placeholder="np. References")
        self.tf_col_cit = W.Text(description="Kolumna cytowań (opcjonalnie)", placeholder="np. Cited by")
        self.tf_col_sim = W.Text(description="Kolumna similarity (opcjonalnie)", placeholder="np. semantic_similarity")
        self.tf_col_dist= W.Text(description="Kolumna distance (opcjonalnie)", placeholder="np. distance_cosine")

        self.adv_box = W.Accordion(children=[
            W.VBox([self.cb_sem_query, self.ta_query, self.info_query]),
            W.VBox([self.tf_col_kw, self.tf_col_ref, self.tf_col_cit, self.tf_col_sim, self.tf_col_dist])
        ])
        self.adv_box.set_title(0, "Opcjonalnie: policz podobieństwo z zapytania")
        self.adv_box.set_title(1, "Opcjonalnie: własne nazwy kolumn")

        # --------- WIDGETS: akcje / wyniki ----------
        self.btn_run = W.Button(description="Uruchom ranking", button_style="success")
        self.btn_download = W.Button(description="Pobierz wyniki (CSV)", disabled=True)
        self.out_run = W.Output(layout={"border": "1px solid #ddd"})

        # Handlery
        self.btn_upload.on_click(self._on_upload_clicked)
        self.btn_run.on_click(self._on_run_clicked)
        self.btn_download.on_click(self._on_download_clicked)

        for w in (self.sl_w_sem, self.sl_w_kw, self.sl_w_ref, self.sl_w_cit):
            w.observe(self._on_weights_changed, names="value")
        self._on_weights_changed(None)

    # -------------------- Eventy --------------------

    def _on_weights_changed(self, _):
        s = sum([self.sl_w_sem.value, self.sl_w_kw.value, self.sl_w_ref.value, self.sl_w_cit.value]) or 1.0
        norm = [self.sl_w_sem.value/s, self.sl_w_kw.value/s, self.sl_w_ref.value/s, self.sl_w_cit.value/s]
        self.lbl_wsum.value = f"<i>Aktualna suma wag:</i> {s:.3f} &nbsp;|&nbsp; <i>Po normalizacji</i> → sem:{norm[0]:.2f}, kw:{norm[1]:.2f}, ref:{norm[2]:.2f}, cit:{norm[3]:.2f}"

    def _on_upload_clicked(self, _btn):
        if not _IN_COLAB:
            with self.out_file:
                clear_output()
                display(_nice_html("To okno uploadu używa <b>google.colab.files</b>. Uruchom ten interfejs w Google Colab."))
            return
        with self.out_file:
            clear_output()
            print("Wybierz plik CSV/TSV…")
        up = colab_files.upload()
        if not up:
            with self.out_file:
                print("Anulowano.")
            return
        name, raw = next(iter(up.items()))
        self.dataset_name = name
        sep_choice = self.dd_sep.value
        try:
            df = _safe_read_csv(name, raw, sep_choice)
            self.df = df
            with self.out_file:
                clear_output()
                print(f"Wczytano: {name}  → shape={df.shape}")
                display(df.head(10))
        except Exception as ex:
            with self.out_file:
                clear_output()
                display(_nice_html(f"<b>Błąd wczytywania:</b> {ex}"))

    def _maybe_compute_semantic_similarity(self):
        if not (self.cb_sem_query.value and self.ta_query.value.strip()):
            return
        assert self.df is not None
        texts = [ _combine_title_abstract(self.df.iloc[i], self.df.columns.tolist()) for i in range(len(self.df)) ]
        sim = _cosine_similarity_to_query(texts, self.ta_query.value.strip())
        self.df["semantic_similarity"] = sim

    def _on_run_clicked(self, _btn):
        with self.out_run:
            clear_output()
            if self.df is None:
                display(_nice_html("Najpierw <b>wgraj plik danych</b> (CSV/TSV)."))
                return
            print("Przygotowuję konfigurację…")

        # opcjonalnie policz semantykę z zapytania
        try:
            if self.cb_sem_query.value and self.ta_query.value.strip():
                with self.out_run:
                    print("Liczenie semantic_similarity względem zapytania…")
                self._maybe_compute_semantic_similarity()
        except Exception as ex:
            with self.out_run:
                display(_nice_html(f"Błąd podczas liczenia podobieństwa: {ex}"))
                return

        # budowa konfiguracji
        cols = ColumnMap(
            keywords=self.tf_col_kw.value or None,
            references=self.tf_col_ref.value or None,
            citations=self.tf_col_cit.value or None,
            semantic_similarity=self.tf_col_sim.value or ("semantic_similarity" if "semantic_similarity" in self.df.columns else None),
            distance_cosine=self.tf_col_dist.value or ("distance_cosine" if "distance_cosine" in self.df.columns else None),
        )

        cfg = ScoringConfig(
            method=self.dd_method.value,
            top_keywords=int(self.sl_topk_kw.value),
            top_references=int(self.sl_topk_ref.value),
            penalty_no_keywords=float(self.sl_pen_kw.value),
            penalty_no_references=float(self.sl_pen_ref.value),
            weights={
                "semantic": float(self.sl_w_sem.value),
                "keywords": float(self.sl_w_kw.value),
                "references": float(self.sl_w_ref.value),
                "citations": float(self.sl_w_cit.value),
            },
            bonus_start_z=float(self.tf_bonus_start.value),
            bonus_full_z=float(self.tf_bonus_full.value),
            bonus_cap_points=(None if int(self.tf_bonus_cap.value or 0) == 0 else float(self.tf_bonus_cap.value)),
            save_frequencies=bool(self.cb_save_freq.value),
            out_dir=self.output_dir,
            columns=cols,
        )

        # uruchom scoring
        try:
            with self.out_run:
                print("Uruchamiam scoring…")
            res = rank_with_advanced_scoring(self.df, cfg)
            self.result = res
        except Exception as ex:
            with self.out_run:
                display(_nice_html(f"<b>Błąd podczas liczenia rankingu:</b> {ex}"))
            return

        # zapisz wyniki
        base = os.path.splitext(self.dataset_name or "advanced_ranking")[0]
        out_rank = os.path.join(self.output_dir, f"{base}_advanced_ranking.csv")
        try:
            res.df.to_csv(out_rank, index=False)
        except Exception as ex:
            out_rank = os.path.join(self.output_dir, "advanced_ranking.csv")
            res.df.to_csv(out_rank, index=False)

        with self.out_run:
            clear_output()
            P = res.P
            print(f"Liczba rekordów (P): {P}")
            print(f"Metoda: {cfg.method}")
            print(f"Ranking zapisany → {out_rank}")
            if cfg.save_frequencies:
                print(f"Raporty: {os.path.join(self.output_dir, 'keyword_frequency.csv')} , {os.path.join(self.output_dir, 'reference_frequency.csv')}")
            display(res.df.head(20))  # podgląd TOP 20
            display(_nice_html("Gotowe. Możesz pobrać pliki przyciskiem poniżej."))

        self.btn_download.disabled = False
        self.btn_download.tooltip = out_rank

    def _on_download_clicked(self, _btn):
        if not _IN_COLAB:
            with self.out_run:
                display(_nice_html("Pobieranie obsługiwane jest w Google Colab (google.colab.files.download)."))
            return
        # zawsze ranking
        out_rank = self.btn_download.tooltip or os.path.join(self.output_dir, "advanced_ranking.csv")
        if os.path.exists(out_rank):
            colab_files.download(out_rank)
        # opcjonalne raporty
        kf = os.path.join(self.output_dir, "keyword_frequency.csv")
        rf = os.path.join(self.output_dir, "reference_frequency.csv")
        if os.path.exists(kf):
            colab_files.download(kf)
        if os.path.exists(rf):
            colab_files.download(rf)

    # -------------------- publiczny layout --------------------

    def display(self):
        if not _HAVE_WIDGETS:
            display(_nice_html("To UI wymaga <b>ipywidgets</b>. W Colab wpisz: <code>pip install ipywidgets</code> i zrestartuj runtime."))
            return

        header = self.lbl_title

        # sekcja pliku
        file_box = W.VBox([
            W.HBox([self.btn_upload, self.dd_sep]),
            self.out_file
        ])

        # sekcja konfiguracji
        weights_box = W.VBox([self.sl_w_sem, self.sl_w_kw, self.sl_w_ref, self.sl_w_cit, self.lbl_wsum])
        params_box = W.VBox([
            self.dd_method,
            W.HBox([self.sl_topk_kw, self.sl_topk_ref]),
            W.HBox([self.sl_pen_kw, self.sl_pen_ref]),
            W.HTML("<hr>"),
            W.HTML("<b>Wagi kryteriów</b>"),
            weights_box,
            W.HTML("<hr>"),
            W.HTML("<b>Bonusy dla L‑Scoring+</b>"),
            W.HBox([self.tf_bonus_start, self.tf_bonus_full, self.tf_bonus_cap]),
            self.cb_save_freq,
            self.adv_box
        ])

        action_box = W.VBox([self.btn_run, self.btn_download, self.out_run])

        ui = W.VBox([header, W.HTML("<h4>1) Wgraj dane</h4>"), file_box,
                     W.HTML("<h4>2) Ustawienia scoringu</h4>"), params_box,
                     W.HTML("<h4>3) Uruchom i pobierz wyniki</h4>"), action_box])

        display(ui)


# -------------------------- API publiczne -------------------------------

def run() -> None:
    """
    Uruchom interfejs Colab dla EmbedSLR.
    Wywołanie: 
        from embedslr.colab_app import run
        run()
    """
    app = _ColabApp()
    app.display()
