# Text2Breed — Landing → BreedFinder (Top-4, gambar lokal)
# ---------------------------------------------------------
# - Landing page (hero + CTA)
# - BreedFinder (textarea + tombol kirim + hasil di atas form)
# - Model & index hanya diload saat halaman BreedFinder

import base64, re, unicodedata, json, html, torch, numpy as np, streamlit as st
from huggingface_hub import snapshot_download
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -------------------------
# Paths & constants
# -------------------------
REPO_ID = "Kaezel/dogBreedsTextClassification2"  # <- punyamu

HERE = Path(__file__).parent.resolve()
ASSETS_DIR = HERE / "assets"
LOGO = ASSETS_DIR / "logo_new_nobg.png"
LOGO_UNTAR = ASSETS_DIR / "logo_untar.png"
BG = ASSETS_DIR / "bg.png"
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
    st.session_state.stage = "landing"   # "landing" | "about" | "finder"

# -------------------------
# Utilities
# -------------------------
def _img_b64(p: Path) -> str:
    try:
        return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception:
        return ""

def set_background_local(image_path: str, overlay_opacity: float = 0.55):
    ext = Path(image_path).suffix.lower()
    mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(f"""
    <style>
    /* Latar belakang utama aplikasi */
    [data-testid="stAppViewContainer"] {{
      background: linear-gradient(rgba(0,0,0,{overlay_opacity}), rgba(0,0,0,{overlay_opacity})),
                  url("data:{mime};base64,{data}") center/cover no-repeat fixed;
    }}
    """, unsafe_allow_html=True)

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
      .t2b-hero-title {{ color:{ACCENT}; font-size: 40px; font-weight: 800; margin: 18px 0 8px; }}
      .t2b-hero-sub   {{ color:{TXT}; opacity:.9; line-height:1.6; font-size:28px; max-width:720px; }}

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
        center = st.columns([3, 6, 3])[1]

        with center:
            # Baris brand: logo (150px) + title sejajar vertikal tengah
            row = st.columns([3, 8], gap="small", vertical_alignment="center")
            with row[0]:
                if LOGO.exists():
                    st.image(str(LOGO), width=400)   # <-- sesuai permintaan
                else:
                    st.write("🐾")
            with row[1]:
                st.markdown(
                    f"<h1 class='t2b-brand-title'>Text2Breed</h1>",
                    unsafe_allow_html=True
                )

            st.markdown('<div class="t2b-hero-title">Dari Teks ke Ras Anjing</div>', unsafe_allow_html=True)
            st.markdown('<div class="t2b-hero-sub">Silahkan tuliskan ciri fisik atau kepribadian dari anjing yang Anda inginkan, kami akan bantu menemukan rasnya.</div>', unsafe_allow_html=True)
            st.write("")

    centerBtn = st.columns([3, 6, 3])[1]
    with centerBtn:
        rowBtn = st.columns([2, 10], gap="small", vertical_alignment="center")
        with rowBtn[0]:
            if st.button("Temukan Ras", key="cta_find"):
                go("finder")
        with rowBtn[1]:
            if st.button("Tentang Kami", key="cta_about"):
                go("about")


# === [ADD] About section (mirip kartu besar dengan heading & paragraf) ===
# [REPLACE] about_section() lama
def about_section():
    st.markdown(f"""
    <style>
      .t2b-about {{
        background:{BG_CARD};
        border-top:4px solid {ACCENT};
        border-radius:16px; padding:30px; margin-top:18px;
        box-shadow:0 8px 24px rgba(0,0,0,.12);
      }}
      .t2b-about-row{{
        display:flex; align-items:center; gap:28px;
      }}
      .t2b-about-left{{ flex:6; }}
      .t2b-about-left p{{ color:{TXT}; opacity:.92; font-size:22px; line-height:1.7; margin:0 0 10px 0; }}
      .t2b-about-left h2{{ color:{ACCENT}; margin:0 0 12px 0; font-weight:800; }}
      .t2b-about-right{{ flex:5; display:flex; justify-content:center; background: white; border-radius:16px; padding:30px; }}
      .t2b-about-right img{{ max-width:100%; height:auto; border-radius:12px; }}
      @media (max-width: 768px){{
        .t2b-about-row{{ flex-direction:column-reverse; }}
      }}
    </style>
    """, unsafe_allow_html=True)

    # siapkan src logo (base64) agar bisa disematkan di HTML
    logo_src = f"data:image/png;base64,{_img_b64(LOGO_UNTAR)}" if LOGO_UNTAR.exists() else None

    st.markdown(f"""
    <div class="t2b-about">
      <div class="t2b-about-row">
        <div class="t2b-about-left">
          <h2>Tentang Text2Breed</h2>
          <p><b>Text2Breed</b> adalah platform untuk membantu menemukan ras anjing
          yang paling mendekati berdasarkan <i>deskripsi naratif</i> pengguna. Platform ini dikembangkan oleh
          Jafier Andreas, mahasiswa Teknik Informatika Universitas Tarumanagara, sebagai bagian dari tugas akhir.
          Proyek ini dibimbing oleh Ibu Ir. Jeanny Pragantha, M.Eng. dan Bapak Henoch Juli Christanto, S.Kom., M.Kom.</p>
          <p>Label mencakup <b>332</b> ras anjing yang diakui oleh organisasi anjing internasional yaitu
          <i>Fédération Cynologique Internationale</i>.</p>
        </div>
        <div class="t2b-about-right">
          {('<img src="'+logo_src+'" alt="Universitas Tarumanagara" />') if logo_src else '🐾'}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def go(stage: str):
    st.session_state.stage = stage
    st.rerun()

# ===== [ADD] Konstanta ambang & pesan panduan =====
from types import SimpleNamespace

CONF = SimpleNamespace(
    MIN_SIM=0.52,      # top-1 cosine similarity minimal
    MIN_PROB=0.15,     # top-1 probability minimal (butuh T & scale)
)

GUIDE_GENERAL = (
    "Deskripsi tampak belum spesifik. Mohon tambahkan detail seperti tinggi/berat (cm/kg), "
    "tekstur & warna bulu, bentuk telinga/moncong, dan temperamen."
)

GUIDE_AMBIG = (
    "Beberapa ras memiliki kemiripan yang hampir setara. Mohon tambahkan ciri pembeda yang lebih khas, "
    "misalnya bentuk telinga (tegak/menjuntai/setengah), bentuk moncong (pendek/panjang), "
    "panjang & tekstur bulu (pendek/panjang/kasar/keriting), pola warna spesifik "
    "(merle/brindle/sable/tricolor, mask/blaze/white markings), ukuran dewasa (tinggi cm, berat kg), "
    "ciri ekor (panjang/keriting/melengkung/carriage), tujuan/asal ras (gembala, penjaga, pemburu, companion), "
    "atau tingkat energi & kebutuhan latihan."
)

# ===== [ADD] Gate low-confidence yang konsisten =====
def gate_prediction(sims: torch.Tensor, topv: torch.Tensor, topi: torch.Tensor,
                    T: float | None, scale: float | None) -> dict:
    """
    Mengembalikan diagnostic:
      - low_conf: bool -> perlu tampilkan peringatan
      - reasons: list[str] -> alasan yang memicu
      - top1_sim, top1_prob, margin, entropy (angka-angka ringkas)
    """
    top1_sim = float(topv[0])
    top1_prob = None

    # Prob/entropy hanya jika kalibrasi tersedia
    if (T is not None) and (scale is not None):
        logits = (scale * sims) / float(T)
        probs  = torch.softmax(logits, dim=0)
        top1_prob = float(probs[topi[0]])

    # Cek syarat low-confidence
    cause, msg = None, None
    if top1_sim < CONF.MIN_SIM:
        cause, msg = "low_sim", GUIDE_GENERAL
    elif (top1_prob is not None) and (top1_prob < CONF.MIN_PROB):
        cause, msg = "ambiguous", GUIDE_AMBIG
    
    return {
        "low_conf": cause is not None,
        "cause": cause,
        "msg": msg,
        "top1_sim": top1_sim,
        "top1_prob": top1_prob,
    }

def looks_like_gibberish(txt: str) -> bool:
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", txt).lower()
    tokens = [w for w in t.split() if len(w) >= 3]
    if not tokens:
        return True
    # 1) cukup vokal di token
    if sum(1 for w in tokens if re.search(r"[aiueo]", w)) / len(tokens) < 0.5:
        return True
    # 2) keyboard-mash umum
    if re.search(r"\b(?:asd|asdf|qwe|qwer|zxc|jkl|lkj|dfg)\w*\b", t):
        return True
    # 3) pengulangan huruf ekstrem
    if re.search(r"(.)\1\1\1+", t):
        return True
    return False

def validate_text(txt: str) -> tuple[bool, str]:
    if looks_like_gibberish(txt):
        return False, "Maaf, deskripsi tampak tidak memiliki makna. Mohon tuliskan ciri fisik/temperamen yang jelas."
    return True, ""

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
      .t2b-name {{ color:{ACCENT}; font-weight:800; font-size:28px; margin:6px 0 2px; text-align:center; }}
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
        margin-top:8px; opacity:.85; font-size:14px; display:flex; gap:6px; align-items:center;
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
if st.session_state.get("stage") in ("landing","about"):
    set_background_local("assets/bg3.jpeg", overlay_opacity=0.3)
else:
    set_background_local("assets/bg4.png", overlay_opacity=0.8)
@st.cache_resource
def load_assets():
    # ambil hanya yang dibutuhkan
    local_root = snapshot_download(
        REPO_ID,
        allow_patterns=[
            "mpnet_proto_ce_v1/*",
            "index_proto_ce_v1/*",
            "temperature.json"
        ],
        ignore_patterns=["*.md", "*.txt~", "*.log"]
    )
    local_root = Path(local_root)

    model_dir = local_root / "mpnet_proto_ce_v1"       # <— penting: arahkan ke subfolder model
    index_dir = local_root / "index_proto_ce_v1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(str(model_dir), device=device)

    # index
    P = np.load(index_dir / "prototypes.npy")
    if P.dtype != np.float32:
        P = P.astype(np.float32)
    proto = torch.from_numpy(P).to(model.device)
    proto = torch.nn.functional.normalize(proto, p=2, dim=1)

    labels = (index_dir / "labels.txt").read_text(encoding="utf-8").splitlines()
    labels = [ln.strip() for ln in labels if ln.strip()]

    # temperature
    T = scale = None
    tj = (local_root / "temperature.json")
    if tj.exists():
        j = json.loads(tj.read_text(encoding="utf-8"))
        T = float(j.get("T")) if "T" in j else None
        scale = float(j.get("scale")) if "scale" in j else None

    return model, proto, labels, T, scale

# ===== [UPDATE] predict_top4 -> kembalikan prediksi + diagnostic =====
def predict_top4(model, proto, labels, text: str, T=None, scale=None):
    if not text.strip():
        return [], None
    with torch.inference_mode():
        e = model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
        sims = (e @ proto.T)[0]
        k = min(TOP_K, sims.shape[-1])
        topv, topi = torch.topk(sims, k=k)

        diag = gate_prediction(sims, topv, topi, T, scale)

        if (T is not None) and (scale is not None):
            logits    = (scale * sims) / float(T)
            probs_all = torch.softmax(logits, dim=0)
            top_probs = probs_all[topi]
            preds = [(labels[j], float(s), float(p))
                     for j, s, p in zip(topi.tolist(), topv.tolist(), top_probs.tolist())]
        else:
            preds = [(labels[j], float(s), None) for j, s in zip(topi.tolist(), topv.tolist())]

        return preds, diag

# -------------------------
# Router
# -------------------------
if st.session_state.stage == "landing":
    hero_landing()
elif st.session_state.stage == "about":   # [ADD]
    navbar()
    if st.button("Kembali"):
        go("landing")
    about_section()

else:
    st.markdown("""
    <style>
    div[data-testid="stLayoutWrapper"]:first-child {
    background-color: #13161c;     /* dasar kartu */
    border: 1px solid #2b313c;     /* garis */
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,.35);
    }
                
    .stForm {
    border: 0px;
    padding: 0px;
    }
                
    /* 1) “Berdasarkan deskripsi …” di dalam st.success */
    [data-testid="stAlert"] > div {
    background: transparent !important;
    }
    div.stAlert p { font-size: 20px; line-height: 1.5;}  /* naikin semua alert text */
                
    .t2b-successAlert{
    background: #16a34a;              /* hijau solid 100% opacity */
    color: #ffffff;                    /* teks putih */
    border: 1px solid #11803b;         /* garis tipis */
    border-radius: 12px;               /* rounded corners */
    padding: 14px 18px;                /* ruang dalam */
    font-size: 20px;
    font-weight: 700;                  /* judul/teks tebal */
    box-shadow: 0 4px 14px rgba(0,0,0,.25);
    margin: 8px 0 50px;                /* jarak atas/bawah */
    }

    /* 2–3) Probabilitas & Skor kemiripan (pakai class di render_card) */
    .t2b-prob, .t2b-score { font-size: 16px; line-height: 1.35; }

    /* 4) Subjudul “Top-3 prediksi lainnya:” */
    .t2b-subheading { font-size: 20px; font-weight: 700; margin: 8px 0 6px; }

    /* 5) Label input (kalau pakai label custom + sembunyikan label bawaan) */
    .t2b-input-label { font-size: 20px; font-weight: 700; margin-bottom: 6px; }

    /* 6) Teks dan Placeholder di text_area */
    div[data-testid="stTextArea"] textarea {
        border: 2px solid gray;
        font-size: 20px !important;
        line-height: 1.5 !important;
    }
                
    div[data-testid="stTextArea"] textarea::placeholder { font-size: 20px; opacity: .85; }
    </style>
    """, unsafe_allow_html=True)

    navbar()
    if st.button("Kembali"):
        go("landing")
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
                st.markdown(
                    '<div class="t2b-input-label">Silahkan tuliskan ciri fisik atau kepribadian dari anjing yang Anda inginkan</div>',
                    unsafe_allow_html=True
                )
                q = st.text_area(
                    "",
                    label_visibility="collapsed",
                    height=140,
                    placeholder=("Contoh: Anjing kecil, bulu keriting rapat tinggi 25-30cm berat 10kg, hipoalergenik, perlu grooming rutin "
                                    "(Mohon input Bahasa Indonesia yang baku)"),
                )
                err_box = st.empty()
            with col_btn:
                submitted = st.form_submit_button("", icon=":material/send:", use_container_width=True)

        # Validasi + Prediksi + Render
        if submitted:
            txt = (q or "").strip()
            words = [w for w in txt.split() if w.strip()]

            if not txt:
                err_box.error("Warning: Mohon input deskripsi!")
                st.stop()
            elif len(words) < 5:
                err_box.error("Warning: Mohon input minimal 5 kata!")
                st.stop()

            ok, msg = validate_text(txt)
            if not ok:
                err_box.warning(msg, icon="⚠️")
                st.stop()

            err_box.empty()
            with st.spinner("Menghitung kemiripan…"):
                preds, diag = predict_top4(model, proto, labels, txt, T=T, scale=scale)

            with results_box.container():
                if not preds:
                    st.info("Tidak ada prediksi.")
                else:
                    if diag and diag["low_conf"]:
                        with st.container():
                            st.warning(f"🤔 Prediksi kurang yakin: {diag['msg']}")

                    st.markdown(
                        '<div class="t2b-successAlert">'
                        'Success: Berdasarkan deskripsi yang Anda masukkan, didapatkan ras-ras berikut yang mendekati deskripsi Anda.'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    # Main card center
                    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
                    with c2:
                        st.markdown('<div class="t2b-subheading">Top-1 prediksi:</div>', unsafe_allow_html=True)
                        name, score, prob = preds[0]
                        render_card(name, score, prob)

                    # Alternatif
                    if len(preds) > 1:
                        st.write("")
                        st.markdown('<div class="t2b-subheading">Top-3 prediksi lainnya:</div>', unsafe_allow_html=True)
                        g1, g2, g3 = st.columns(3, gap="large")
                        for item, col in zip(preds[1:4], [g1, g2, g3]):
                            name, score, prob = item
                            with col:
                                render_card(name, score, prob)
                        render_explainer_tip()

        st.markdown('</div>', unsafe_allow_html=True)
