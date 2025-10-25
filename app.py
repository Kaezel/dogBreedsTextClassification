# Text2Breed — Landing → BreedFinder (Top-4, gambar lokal)
# ---------------------------------------------------------
# - Landing page (hero + CTA)
# - BreedFinder (textarea + tombol kirim + hasil di atas form)
# - Model & index hanya diload saat halaman BreedFinder

import base64, re, unicodedata, json, html, torch, numpy as np, streamlit as st
from huggingface_hub import hf_hub_download
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -------------------------
# Paths & constants
# -------------------------
REPO_ID = "Kaezel/dogBreedsTextClassification"  # <- punyamu

HERE = Path(__file__).parent.resolve()
ASSETS_DIR = HERE / "assets"
LOGO = ASSETS_DIR / "logo_new_nobg.png"
DOGIMG_DIR = ASSETS_DIR / "dogImg"

TOP_K = 4

ACCENT  = "#d4af37"
TXT     = "#eaeaea"
BG_NAV  = "#0f1117"
BG_CARD = "#151922"

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Text2Breed",
    page_icon=str(LOGO) if LOGO.exists() else "🐾",
    layout="wide",
)

# -------------------------
# Session state
# -------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "landing"   # "landing" | "finder"

# -------------------------
# Utilities
# -------------------------
def _img_b64(p: Path) -> str:
    try:
        return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception:
        return ""

def _find_model_dir(base: Path) -> Path:
    # cari folder yg berisi 'modules.json' (format SentenceTransformer.save)
    for p in base.rglob("modules.json"):
        return p.parent
    # fallback: ada config.json + pytorch_model.bin
    for p in base.rglob("pytorch_model.bin"):
        return p.parent
    raise FileNotFoundError("Folder model (modules.json / pytorch_model.bin) tidak ditemukan.")

def _find_index_dir(base: Path) -> Path:
    # cari prototypes.npy + labels.txt
    protos = list(base.rglob("prototypes.npy"))
    labels = list(base.rglob("labels.txt"))
    if not protos or not labels:
        raise FileNotFoundError("index (prototypes.npy/labels.txt) tidak ditemukan di repo HF.")
    # pilih yang parentnya sama bila ada beberapa
    for p in protos:
        cand = p.parent
        if (cand / "labels.txt").exists():
            return cand
    return protos[0].parent

def _read_temperature(base: Path):
    # bisa di root atau di subfolder; baca pertama yang ketemu
    for p in [base / "temperature.json", *base.rglob("temperature.json")]:
        if p.exists():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                T = float(j.get("T")) if "T" in j else None
                scale = float(j.get("scale")) if "scale" in j else None
                return T, scale
            except Exception:
                pass
    return None, None

def render_explainer_tip():
    tip = ("Kemiripan −1..1 (semakin besar semakin mirip). "
           "Probabilitas 0–100% adalah softmax((scale × kemiripan)/T) "
           "— versi terkalibrasi untuk membandingkan kandidat.")
    tip = html.escape(tip)
    st.markdown(
        f"""
        <div class="t2b-footnote">
          Cara baca skor
          <span class="t2b-help" data-tip="{tip}">i</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def navbar():
    logo_src = f"data:image/png;base64,{_img_b64(LOGO)}" if LOGO.exists() else None
    st.markdown(
        f"""
        <style>
          header[data-testid="stHeader"] {{ display:none; }}
          .block-container {{ padding-top: 90px; }}
          .t2b-nav {{
            position: fixed; top:0; left:0; right:0; height:64px;
            background:{BG_NAV}; border-bottom:1px solid rgba(255,255,255,.08);
            display:flex; align-items:center; z-index:1000;
          }}
          .t2b-wrap {{ max-width:1200px; margin:0 20px; width:100%;
                       display:flex; align-items:center; gap:12px; padding:0 16px; }}
          .t2b-title {{ color:{ACCENT}; font-weight:800; font-size:26px; letter-spacing:.3px; }}
          .t2b-logo {{ height:80px; width:auto; }}
          .t2b-spacer {{ margin-left:auto; }}
        </style>
        <div class="t2b-nav">
          <div class="t2b-wrap">
            {'<img class="t2b-logo" src="'+logo_src+'" alt="logo"/>' if logo_src else '<span style="font-size:24px">🐾</span>'}
            <div class="t2b-title">Text2Breed</div>
            <div class="t2b-spacer"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def hero_landing():
    st.markdown(f"""
    <style>
      /* hero container */
      div[data-testid="stVerticalBlock"]:has(> .t2b-hero-marker) {{
          min-height: 70vh;
          display: flex; align-items: center; justify-content: center;
      }}
      .t2b-hero-title {{ color:{ACCENT}; font-size: 32px; font-weight: 800; margin: 18px 0 8px; }}
      .t2b-hero-sub   {{ color:{TXT}; opacity:.9; line-height:1.6; font-size:18px; max-width:720px; }}

      /* H1 brand — pakai !important agar tak ketimpa tema Streamlit */
      .t2b-brand-title {{
          color:{ACCENT} !important;
          font-weight:800; font-size:44px; margin:0; line-height:1;
      }}
    </style>
    """, unsafe_allow_html=True)

    # === HERO container (semua isi di dalam container ini) ===
    with st.container():
        st.markdown('<div class="t2b-hero-marker"></div>', unsafe_allow_html=True)

        # konten hero dipusatkan
        center = st.columns([4, 6, 4])[1]

        with center:
            # Baris brand: logo (150px) + title sejajar vertikal tengah
            row = st.columns([2, 8], gap="small", vertical_alignment="center")
            with row[0]:
                if LOGO.exists():
                    st.image(str(LOGO), width=150)   # <-- sesuai permintaan
                else:
                    st.write("🐾")
            with row[1]:
                st.markdown(
                    f"<h1 class='t2b-brand-title'>Text2Breed</h1>",
                    unsafe_allow_html=True
                )

            st.markdown('<div class="t2b-hero-title">Dari Teks ke Ras</div>', unsafe_allow_html=True)
            st.markdown('<div class="t2b-hero-sub">Tanpa foto pun bisa. Cukup tulis ciri fisik atau kepribadiannya, kami bantu cari rasnya.</div>', unsafe_allow_html=True)
            st.write("")
            go = st.button("Temukan Ras", key="cta_btn")

    if go:
        st.session_state.stage = "finder"
        st.rerun()

# ----- card CSS (shared) -----
st.markdown(
    f"""
    <style>
      .t2b-card {{
        background:{BG_CARD}; border:1px solid rgba(255,255,255,0.08);
        border-radius:14px; padding:16px; box-shadow:0 8px 24px rgba(0,0,0,0.25);
      }}
      .t2b-card img {{
        width:100%; height:375px; object-fit:cover;
        border-radius:12px; border:1px solid rgba(255,255,255,.06);
        margin-bottom:10px;
      }}
      .t2b-name {{ color:{ACCENT}; font-weight:800; font-size:26px; margin:6px 0 2px; text-align:center; }}
      .t2b-score {{ opacity:.9; text-align:center; font-size:14px; }}
      .t2b-center {{ max-width: 1000px; margin: 0 auto; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
      /* ...CSS kamu yang sudah ada... */

      /* Tooltip icon */
      .t2b-help {{
        position: relative; display:inline-flex; align-items:center; justify-content:center;
        width:18px; height:18px; border-radius:50%;
        background: rgba(255,255,255,.12); color:{TXT};
        font-size:12px; line-height:18px; margin-left:6px; cursor:default;
        border:1px solid rgba(255,255,255,.15);
      }}
      .t2b-help:hover::after {{
        content: attr(data-tip);
        position:absolute; bottom:140%; left:50%; transform:translateX(-50%);
        background:#111; color:#fff; padding:8px 10px; border-radius:8px;
        width:280px; font-size:12px; line-height:1.4; white-space:normal;
        box-shadow:0 8px 24px rgba(0,0,0,.35); z-index:9999;
      }}
      .t2b-help:hover::before {{
        content:""; position:absolute; bottom:128%; left:50%; transform:translateX(-50%);
        border:6px solid transparent; border-top-color:#111;
      }}
      .t2b-footnote {{
        margin-top:8px; opacity:.85; font-size:12px; display:flex; gap:6px; align-items:center;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- image helpers & render card -----
def _candidate_stems(name: str):
    s = name.strip()
    s1 = s.replace("/", "-").replace("\\", "-").replace(":", "-")
    s2 = re.sub(r"[()\\[\\]{}]", "", s1)
    s3 = s2.replace("’", "").replace("'", "")
    s4 = unicodedata.normalize("NFKD", s3).encode("ascii", "ignore").decode("ascii")
    variants = {s, s1, s2, s3, s4, s4.replace(" ", "_"), s4.replace(" ", "-"),
                s3.replace(" ", "_"), s3.replace(" ", "-")}
    return [v for v in variants if v]

def find_breed_image(name: str):
    exts = ["jpg", "jpeg", "png", "webp"]
    for stem in _candidate_stems(name):
        for ext in exts:
            p = DOGIMG_DIR / f"{stem}.{ext}"
            if p.exists():
                return p
    return None

def _img_src_for(name: str) -> str:
    p = find_breed_image(name)
    if p is None:
        return f"https://placehold.co/800x450?text={name.replace(' ', '+')}"
    ext = p.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else ("image/png" if ext == ".png" else "image/webp")
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def render_card(name: str, score: float, prob: float | None):
    src = _img_src_for(name)
    prob_html = f'<div class="t2b-score">Probabilitas: {prob*100:.1f}%</div>' if prob is not None else ''
    tip_txt = ("Kemiripan −1..1. Probabilitas = softmax((scale × kemiripan)/T).")
    st.markdown(
        f"""
        <div class="t2b-card">
          <img src="{src}" alt="{name}" />
          <div class="t2b-name">{name}</div>
          {prob_html}
          <div class="t2b-score">Skor kemiripan: {score:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----- model & inference (lazy load) -----
@st.cache_resource
def load_assets():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load model dari subfolder repo (bukan root)
    model = SentenceTransformer(
        REPO_ID,
        subfolder="mpnet_proto_ce_v1",
        device=device
    )

    # 2) Unduh index per file dari subfolder index
    labels_fp = hf_hub_download(
        REPO_ID, filename="labels.txt", subfolder="index_proto_ce_v1"
    )
    protos_fp = hf_hub_download(
        REPO_ID, filename="prototypes.npy", subfolder="index_proto_ce_v1"
    )

    # 3) Baca index
    with open(labels_fp, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]

    proto_np = np.load(protos_fp)
    if proto_np.dtype != np.float32:
        proto_np = proto_np.astype(np.float32)
    proto = torch.from_numpy(proto_np).to(model.device)
    proto = torch.nn.functional.normalize(proto, p=2, dim=1)

    # 4) Kalibrasi (opsional)
    T, scale = None, None
    try:
        temp_fp = hf_hub_download(REPO_ID, filename="temperature.json")
        with open(temp_fp, "r", encoding="utf-8") as f:
            tj = json.load(f)
        T = float(tj.get("T")) if "T" in tj else None
        scale = float(tj.get("scale")) if "scale" in tj else None  # boleh tidak ada
    except Exception:
        pass  # tidak apa-apa, fallback ke skor kemiripan saja

    # Safety: pastikan sinkron
    if proto.shape[0] != len(labels):
        raise RuntimeError(
            f"Mismatch index: prototypes={proto.shape[0]} vs labels={len(labels)}"
        )

    return model, proto, labels, T, scale

def predict_top4(model, proto, labels, text: str, T=None, scale=None):
    if not text.strip():
        return []
    with torch.inference_mode():
        e = model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
        sims = (e @ proto.T)[0]
        k = min(TOP_K, sims.shape[-1])
        topv, topi = torch.topk(sims, k=k)

        probs = None
        if (T is not None) and (scale is not None):
            logits = (scale * sims) / float(T)
            probs = torch.softmax(logits, dim=0)
            top_probs = probs[topi]

            idxs = topi.detach().cpu().tolist()
            vals = topv.detach().cpu().tolist()
            prbs = top_probs.detach().cpu().tolist()
            return [(labels[j], float(s), float(p)) for j, s, p in zip(idxs, vals, prbs)]
        else:
            idxs = topi.detach().cpu().tolist()
            vals = topv.detach().cpu().tolist()
            return [(labels[j], float(s), None) for j, s in zip(idxs, vals)]

# -------------------------
# Router
# -------------------------
if st.session_state.stage == "landing":
    hero_landing()
else:
    navbar()
    # BreedFinder page
    with st.spinner("Memuat model & index…"):
        model, proto, labels, T, scale = load_assets()

    if proto.shape[0] != len(labels):
        st.error(f"Jumlah prototipe ({proto.shape[0]}) ≠ jumlah label ({len(labels)}). Cek artifacts/index_ce_v1.")
        st.stop()

    # satu div terpusat
    with st.container(gap="medium"):
        st.markdown('<div class="t2b-center">', unsafe_allow_html=True)

        results_box = st.empty()  # hasil di atas form

        # FORM
        with st.form("t2b_form", clear_on_submit=False):
            col_text, col_btn = st.columns([12, 1], gap="small", vertical_alignment="center")
            with col_text:
                q = st.text_area(
                    "Deskripsi",
                    height=140,
                    placeholder=("Contoh: Anjing kecil, bulu keriting rapat tinggi 25-30cm berat 10kg, hipoalergenik, perlu grooming rutin "
                                 "(Mohon input Bahasa Indonesia yang baku)"),
                    label_visibility="collapsed",
                )
            with col_btn:
                submitted = st.form_submit_button(" ", icon=":material/send:", use_container_width=True)

        # Validasi + Prediksi + Render
        if submitted:
            txt = (q or "").strip()
            words = [w for w in txt.split() if w.strip()]
            if not txt:
                st.warning("Mohon input deskripsi", icon="⚠️")
            elif len(words) < 10:
                st.warning("Mohon input minimal 10 kata!", icon="⚠️")
            else:
                with st.spinner("Menghitung kemiripan…"):
                    preds = predict_top4(model, proto, labels, txt, T=T, scale=scale)

                with results_box.container():
                    if not preds:
                        st.info("Tidak ada prediksi.")
                    else:
                        # Main card center
                        c1, c2, c3 = st.columns([1, 1, 1], gap="large")
                        with c2:
                            name, score, prob = preds[0]
                            render_card(name, score, prob)

                        # Alternatif
                        if len(preds) > 1:
                            st.write("")
                            st.markdown("**Top-3 prediksi lainnya:**")
                            g1, g2, g3 = st.columns(3, gap="large")
                            for item, col in zip(preds[1:4], [g1, g2, g3]):
                                name, score, prob = item
                                with col:
                                    render_card(name, score, prob)
                            render_explainer_tip()

        st.markdown('</div>', unsafe_allow_html=True)
