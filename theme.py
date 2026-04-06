"""
MetaTune Theme System
=====================
Centralised CSS injection for light / dark mode.
All design tokens and Streamlit overrides live here.
"""

import streamlit as st


# ─── Font preconnect (call BEFORE inject_theme_css) ──────────────────────────
def inject_font_preconnect():
    """Emit <link rel="preconnect"> so Google Fonts begin loading immediately."""
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        unsafe_allow_html=True,
    )


# ─── Phase header helper ────────────────────────────────────────────────────
def render_phase_header(phase_num: str, phase_name: str, subtitle: str,
                        accent_color: str = "var(--dna-green)"):
    """Render a consistent phase section header with gradient accent line."""
    st.markdown(f"""
    <div style="margin: 24px 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border);">
      <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-family: var(--font-mono); font-size: 10px; color: var(--text-dim); letter-spacing: 0.3em;">{phase_num}</span>
        <div style="flex: 1; height: 1px; background: linear-gradient(90deg, {accent_color}, transparent);"></div>
      </div>
      <h2 style="font-family: var(--font-display); font-size: 32px; color: var(--text-primary); margin: 4px 0; letter-spacing: 0.08em;">{phase_name}</h2>
      <p style="font-family: var(--font-mono); font-size: 11px; color: var(--text-secondary); letter-spacing: 0.1em;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ─── DNA metric card helper ─────────────────────────────────────────────────
def render_dna_metric(label: str, value, accent: str = "var(--bio-cyan)"):
    """Return HTML for a single chamfer-cut DNA metric card."""
    return f"""
    <div style="
      background: var(--panel);
      border: 1px solid var(--border);
      clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 0 100%);
      padding: 12px;
      position: relative;
      overflow: hidden;
    ">
      <div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: {accent};"></div>
      <p style="font-family: var(--font-mono); font-size: 9px; color: var(--text-dim); letter-spacing: 0.2em; text-transform: uppercase; margin: 4px 0 6px;">{label}</p>
      <p style="font-family: var(--font-display); font-size: 22px; color: var(--text-primary); margin: 0;">{value}</p>
    </div>
    """


# ─── Main CSS injection ─────────────────────────────────────────────────────
def inject_theme_css(theme: str = "dark"):
    """Inject the complete MetaTune stylesheet with theme-aware design tokens.

    Call this ONCE at the top of the render pass, after inject_font_preconnect().
    """
    is_dark = theme == "dark"

    # ── Token values ──
    void            = "#03040A"  if is_dark else "#F8FAFC"
    deep            = "#080B14"  if is_dark else "#F1F5F9"
    surface         = "#0D1220"  if is_dark else "#FFFFFF"
    panel           = "#111827"  if is_dark else "#F8FAFC"
    border          = "#1a2540"  if is_dark else "#CBD5E1"
    dna_green       = "#00FF88"  if is_dark else "#059669"
    neural_amber    = "#FFB800"  if is_dark else "#D97706"
    quantum_magenta = "#FF006E"  if is_dark else "#DB2777"
    bio_cyan        = "#00D4FF"  if is_dark else "#0284C7"
    evolution_purple = "#9B5DE5" if is_dark else "#7C3AED"
    text_primary    = "#E8EEF4"  if is_dark else "#0F172A"
    text_secondary  = "#7A8BA0"  if is_dark else "#475569"
    text_dim        = "#3D4F66"  if is_dark else "#94A3B8"

    # ── Build the :root tokens block (needs f-string escaping) ──
    tokens = f""":root {{
  --void: {void};
  --deep: {deep};
  --surface: {surface};
  --panel: {panel};
  --border: {border};
  --dna-green: {dna_green};
  --neural-amber: {neural_amber};
  --quantum-magenta: {quantum_magenta};
  --bio-cyan: {bio_cyan};
  --evolution-purple: {evolution_purple};
  --text-primary: {text_primary};
  --text-secondary: {text_secondary};
  --text-dim: {text_dim};
  --font-display: 'Bebas Neue', sans-serif;
  --font-tech: 'Chakra Petch', sans-serif;
  --font-mono: 'Share Tech Mono', monospace;
}}"""

    # ── All CSS rules (plain string — no interpolation needed) ──
    rules = """
/* ═══════════════════════════════════════
   FORCE STREAMLIT CONTAINERS
═══════════════════════════════════════ */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
  background-color: var(--void) !important;
  color: var(--text-primary) !important;
}

/* ═══════════════════════════════════════
   BASE OVERRIDES
═══════════════════════════════════════ */
.stApp { background-color: var(--void) !important; color: var(--text-primary); font-family: var(--font-tech); }
.main .block-container { padding-top: 2rem; max-width: 100%; }

/* ═══════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════ */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ═══════════════════════════════════════
   METRIC CARDS
═══════════════════════════════════════ */
div[data-testid="stMetric"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-left: 3px solid var(--dna-green) !important;
  border-radius: 0 !important;
  padding: 16px 20px !important;
  clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
  transition: border-left-color 0.3s, transform 0.2s !important;
  font-family: var(--font-mono) !important;
}
div[data-testid="stMetric"]:hover {
  border-left-color: var(--bio-cyan) !important;
  transform: translateY(-4px) scale(1.02) !important;
  filter: drop-shadow(0 8px 16px rgba(0, 255, 136, 0.1)) !important;
}
div[data-testid="stMetricLabel"] {
  font-family: var(--font-mono) !important;
  font-size: 9px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  color: var(--text-dim) !important;
}
div[data-testid="stMetricValue"] {
  font-family: var(--font-display) !important;
  font-size: 28px !important;
  color: var(--text-primary) !important;
}

/* ═══════════════════════════════════════
   HEADERS
═══════════════════════════════════════ */
h1 { font-family: var(--font-display) !important; font-size: 56px !important; letter-spacing: 4px !important; color: var(--text-primary) !important; }
h2 { font-family: var(--font-display) !important; font-size: 36px !important; letter-spacing: 3px !important; color: var(--dna-green) !important; }
h3 { font-family: var(--font-tech) !important; font-weight: 600 !important; letter-spacing: 2px !important; color: var(--text-secondary) !important; }

/* ═══════════════════════════════════════
   ALL TEXT INHERITS THEME
═══════════════════════════════════════ */
.stMarkdown, .stText, p, h1, h2, h3, h4, label {
  color: var(--text-primary) !important;
  font-family: var(--font-tech) !important;
}

/* ═══════════════════════════════════════
   BUTTONS — CINEMATIC
═══════════════════════════════════════ */
.stButton > button {
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  background: linear-gradient(90deg, var(--quantum-magenta) 0%, #9B00FF 100%) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 0 !important;
  clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%) !important;
  padding: 14px 32px !important;
  transition: all 0.3s !important;
  animation: ignitePulse 3s ease-in-out infinite !important;
  width: 100% !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  filter: drop-shadow(0 0 20px rgba(255, 0, 110, 0.5)) !important;
}
.stButton > button:active {
  transform: translateY(2px) scale(0.98) !important;
  filter: drop-shadow(0 0 5px rgba(255, 0, 110, 0.8)) !important;
}

/* ═══════════════════════════════════════
   DOWNLOAD BUTTON
═══════════════════════════════════════ */
.stDownloadButton > button {
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  background: transparent !important;
  color: var(--dna-green) !important;
  border: 1px solid var(--dna-green) !important;
  border-radius: 0 !important;
  transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1) !important;
}
.stDownloadButton > button:hover {
  background: rgba(0,255,136,0.08) !important;
  transform: translateY(-2px) !important;
  filter: drop-shadow(0 0 10px rgba(0,255,136,0.3)) !important;
}
.stDownloadButton > button:active {
  transform: translateY(2px) scale(0.98) !important;
  filter: drop-shadow(0 0 5px rgba(0,255,136,0.8)) !important;
}

/* ═══════════════════════════════════════
   FILE UPLOADER
═══════════════════════════════════════ */
[data-testid="stFileUploader"] {
  background: var(--panel) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 0 !important;
  padding: 16px !important;
  font-family: var(--font-mono) !important;
  transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--dna-green) !important; }

/* ═══════════════════════════════════════
   SELECTBOX
═══════════════════════════════════════ */
[data-testid="stSelectbox"] > div {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  letter-spacing: 1px !important;
}

/* ═══════════════════════════════════════
   INFO / WARNING / SUCCESS BANNERS
═══════════════════════════════════════ */
[data-testid="stAlert"] {
  border-radius: 0 !important;
  border: none !important;
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 1px !important;
}
.stSuccess { border-left: 3px solid var(--dna-green) !important; background: rgba(0,255,136,0.06) !important; }
.stWarning { border-left: 3px solid var(--neural-amber) !important; background: rgba(255,184,0,0.06) !important; }
.stInfo    { border-left: 3px solid var(--bio-cyan) !important; background: rgba(0,212,255,0.06) !important; }
.stError   { border-left: 3px solid var(--quantum-magenta) !important; background: rgba(255,0,110,0.06) !important; }

/* ═══════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--deep) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div:first-child {
  background-color: var(--deep) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-tech) !important; }
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapseButton"] *, .material-symbols-rounded, [data-testid="stIconMaterial"], [data-testid="stSidebarNav"] * { font-family: "Material Symbols Rounded", sans-serif !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  font-family: var(--font-display) !important;
  letter-spacing: 3px !important;
}

/* ═══════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════ */
[data-testid="stProgressBar"] > div {
  background: var(--border) !important;
  border-radius: 0 !important;
  height: 3px !important;
}
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--dna-green), var(--bio-cyan)) !important;
  border-radius: 0 !important;
  filter: drop-shadow(0 0 6px rgba(0,255,136,0.5)) !important;
}

/* ═══════════════════════════════════════
   SPINNER
═══════════════════════════════════════ */
[data-testid="stSpinner"] { font-family: var(--font-mono) !important; font-size: 11px !important; letter-spacing: 2px !important; color: var(--dna-green) !important; }

/* ═══════════════════════════════════════
   EXPANDER
═══════════════════════════════════════ */
[data-testid="stExpander"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 !important;
}
[data-testid="stExpander"] summary { font-family: var(--font-mono) !important; letter-spacing: 2px !important; }

/* ═══════════════════════════════════════
   INPUT TEXT
═══════════════════════════════════════ */
[data-testid="stTextInput"] input {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 !important;
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
}
[data-testid="stTextInput"] input:focus { border-color: var(--dna-green) !important; filter: drop-shadow(0 0 6px rgba(0,255,136,0.2)) !important; }

/* ═══════════════════════════════════════
   ANIMATIONS (filter:drop-shadow path — GPU composited)
═══════════════════════════════════════ */
@keyframes ignitePulse {
  0%, 100% { filter: drop-shadow(0 0 5px rgba(255,0,110,0.3)); }
  50% { filter: drop-shadow(0 0 15px rgba(255,0,110,0.7)) drop-shadow(0 0 30px rgba(155,0,255,0.3)); }
}
@keyframes neuralPulse {
  0%, 100% { filter: drop-shadow(0 0 3px var(--dna-green)) drop-shadow(0 0 8px var(--dna-green)); }
  50% { filter: drop-shadow(0 0 10px var(--dna-green)) drop-shadow(0 0 25px var(--dna-green)); }
}
@keyframes scanSweep {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}
@keyframes matrixFlicker {
  0%, 100% { opacity: 1; }
  33% { opacity: 0.4; }
  66% { opacity: 0.8; }
}
@keyframes slideUpFadeIn {
  from { transform: translateY(24px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}
@keyframes orbitalSpin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
@keyframes orbitalSpinReverse {
  from { transform: rotate(0deg); }
  to { transform: rotate(-360deg); }
}
@keyframes heartbeat {
  0%, 100% { transform: scale(1); }
  14% { transform: scale(1.3); }
  28% { transform: scale(1); }
  42% { transform: scale(1.3); }
}
@keyframes borderTrace {
  0% { background-position: 0% 0%; }
  100% { background-position: 200% 0%; }
}
@keyframes barFillAnim {
  from { width: 0; }
}
@keyframes dataPacketFlow {
  0% { left: -10px; opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { left: calc(100% + 10px); opacity: 0; }
}
@keyframes glitchText {
  0%, 90%, 100% { text-shadow: none; clip-path: none; }
  92% { text-shadow: -2px 0 var(--quantum-magenta), 2px 0 var(--bio-cyan); clip-path: inset(10% 0 85% 0); }
  94% { text-shadow: 2px 0 var(--neural-amber), -2px 0 var(--dna-green); clip-path: inset(50% 0 30% 0); }
  96% { clip-path: none; text-shadow: none; }
}
@keyframes numberRoll {
  from { transform: translateY(-20px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

/* ═══════════════════════════════════════
   HERO — CENTERED STATE
═══════════════════════════════════════ */
.hero--centered {
  min-height: 80vh !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  align-items: flex-start !important;
  padding: 0 48px !important;
  border-bottom: none !important;
  margin-bottom: 0 !important;
  animation: slideUpFadeIn 0.8s ease-out !important;
}

/* ═══════════════════════════════════════
   RESET BUTTON (sidebar — no glow)
═══════════════════════════════════════ */
.reset-btn-wrapper .stButton > button {
  background: transparent !important;
  color: var(--text-dim) !important;
  border: 1px solid var(--border) !important;
  animation: none !important;
  filter: none !important;
}
.reset-btn-wrapper .stButton > button:hover {
  border-color: var(--text-secondary) !important;
  color: var(--text-secondary) !important;
}

/* ═══════════════════════════════════════
   THEME TOGGLE BUTTON (sidebar — subtle)
═══════════════════════════════════════ */
.theme-toggle-wrapper .stButton > button {
  background: var(--panel) !important;
  border: 1px solid var(--bio-cyan) !important;
  color: var(--bio-cyan) !important;
  font-size: 11px !important;
  animation: none !important;
  filter: none !important;
  padding: 8px 16px !important;
}
.theme-toggle-wrapper .stButton > button:hover {
  border-color: var(--dna-green) !important;
  color: var(--dna-green) !important;
}

/* ═══════════════════════════════════════
   ERROR CARD — DESIGN TOKENS (P0 FIX)
═══════════════════════════════════════ */
.metatune-error-card {
  background: color-mix(in srgb, var(--quantum-magenta) 12%, var(--surface));
  border: 1px solid var(--quantum-magenta);
  clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 0 100%);
  border-radius: 0;
  padding: 16px;
  font-family: var(--font-mono);
  color: var(--text-primary);
}

/* ═══════════════════════════════════════
   RESPONSIVE METRIC GRID (P1 FIX)
═══════════════════════════════════════ */
.metatune-metric-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 8px;
}
@media (max-width: 1200px) {
  .metatune-metric-grid { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 768px) {
  .metatune-metric-grid { grid-template-columns: repeat(2, 1fr); }
}

/* ═══════════════════════════════════════
   CTA BUTTON GLOW (P1 FIX — GPU path)
═══════════════════════════════════════ */
.metatune-cta-btn {
  filter: drop-shadow(0 0 8px var(--quantum-magenta));
  transition: filter 0.3s ease;
}
.metatune-cta-btn:hover {
  filter: drop-shadow(0 0 16px var(--quantum-magenta));
}

/* ═══════════════════════════════════════
   PLOTLY CHART BACKGROUNDS
═══════════════════════════════════════ */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
  background: var(--panel) !important;
}

/* ═══════════════════════════════════════
   DIVIDERS
═══════════════════════════════════════ */
hr { border-color: var(--border) !important; }

/* ═══════════════════════════════════════
   CODE BLOCKS
═══════════════════════════════════════ */
code, pre {
  background: var(--deep) !important;
  color: var(--dna-green) !important;
  font-family: var(--font-mono) !important;
  border: 1px solid var(--border) !important;
}

/* ═══════════════════════════════════════
   ACCESSIBILITY
═══════════════════════════════════════ */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
"""

    # ── Light-mode specific overrides ──
    light_overrides = "" if is_dark else """
/* ═══════════════════════════════════════
   LIGHT MODE OVERRIDES
═══════════════════════════════════════ */
/* Reduce / remove dark-mode glow effects */
.stApp [style*="text-shadow"] { text-shadow: none !important; }
.stApp [style*="box-shadow"] { box-shadow: none !important; }

/* Accent text needs higher contrast on white */
h2 { color: var(--dna-green) !important; font-weight: 700 !important; }

/* Status indicator dots — remove glow */
.stApp [style*="box-shadow: 0 0 8px"] { box-shadow: none !important; }
"""

    full_css = (
        "<style>\n"
        "@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue"
        "&family=Chakra+Petch:wght@300;400;600;700"
        "&family=Share+Tech+Mono&display=swap');\n"
        + tokens + "\n"
        + rules
        + light_overrides
        + "\n</style>"
    )
    st.markdown(full_css, unsafe_allow_html=True)


def get_plotly_theme_colors(theme: str = "dark"):
    """Return Plotly layout color overrides matching the active theme."""
    is_dark = theme == "dark"
    return {
        "paper_bgcolor": "rgba(0,0,0,0)" if is_dark else "#FFFFFF",
        "plot_bgcolor": "rgba(8,11,20,0.8)" if is_dark else "#F8FAFC",
        "font_color": "#3D4F66" if is_dark else "#475569",
        "grid_color": "rgba(26,37,64,0.5)" if is_dark else "rgba(203,213,225,0.5)",
    }
