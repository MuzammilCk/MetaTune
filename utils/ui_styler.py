import streamlit as st

def setup_tactical_style():
    """Injects the Tactical Cyberpunk CSS into Streamlit"""
    st.markdown("""
        <style>
        /* 1. IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;900&family=JetBrains+Mono:wght@400;700&display=swap');
        
        /* 2. THEME VARIABLES */
        :root {
            --color-nightBlack: #050505;
            --color-nightDark: #0a0a0a;
            --color-electricBlue: #3b82f6;
            --color-deepBlue: #1e3a8a;
            --font-main: 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }
        /* 3. BASE BACKGROUND */
        .stApp {
            background-color: var(--color-nightBlack);
            color: white;
            font-family: var(--font-main);
        }
        
        /* 4. CUSTOM HEADER & SIDEBAR ADAPTATION */
        header[data-testid="stHeader"] { background-color: transparent !important; }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--color-nightDark) !important;
            border-right: 1px solid #333;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
        
        /* Sidebar Content */
        section[data-testid="stSidebar"] .stMarkdown, 
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p {
            color: #ccc !important;
            font-family: var(--font-mono) !important;
        }
        
        /* Sidebar Headers Highlighting */
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
             color: white !important;
             text-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
        }
        section[data-testid="stSidebar"] hr { border-color: #333 !important; }
        
        /* 5. CUSTOM WIDGET STYLING */
        
        /* Text Input */
        div[data-testid="stTextInput"] input {
            background-color: var(--color-nightDark) !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 8px !important;
            font-family: var(--font-mono) !important;
            text-transform: uppercase;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: var(--color-electricBlue) !important;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.5) !important;
        }
        
        /* Buttons - Skewed & Glowing */
        div[data-testid="stButton"] button {
            background-color: var(--color-electricBlue) !important;
            color: black !important;
            font-weight: 900 !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            transform: skewX(-10deg);
            border: none !important;
            transition: all 0.2s ease;
        }
        div[data-testid="stButton"] button:hover {
            box-shadow: 0 0 15px var(--color-electricBlue);
            color: white !important;
        }
        div[data-testid="stButton"] button p { transform: skewX(10deg); }
        /* 6. File Uploader */
        div[data-testid="stFileUploader"] {
            border: 1px dashed #333;
            border-radius: 8px;
            padding: 20px;
            background-color: var(--color-nightDark);
        }
        div[data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] p {
            font-family: var(--font-mono);
            color: gray;
        }
        /* 7. Metrics (Cards) */
        div[data-testid="stMetric"] {
            background-color: #111;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid var(--color-electricBlue);
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 1px solid #222;
        }
        div[data-testid="stMetric"] label {
            color: gray;
            font-family: var(--font-mono);
            font-size: 0.8rem;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: white;
            font-family: var(--font-main);
            font-weight: 900;
        }
        /* 8. Alerts (Neon Style) */
        div[data-testid="stAlert"] {
            background-color: var(--color-nightDark);
            border: 1px solid #333;
            border-radius: 4px;
            color: white;
        }
        div[data-testid="stAlert"][data-testid*="info"] { border-left: 4px solid var(--color-electricBlue); }
        div[data-testid="stAlert"][data-testid*="success"] { border-left: 4px solid #10B981; }
        div[data-testid="stAlert"][data-testid*="warning"] { border-left: 4px solid #F59E0B; }
        div[data-testid="stAlert"][data-testid*="error"] { border-left: 4px solid #EF4444; }
        /* 9. Progress Bar */
        div[data-testid="stProgress"] > div > div > div {
            background-color: var(--color-electricBlue);
            box-shadow: 0 0 10px var(--color-electricBlue);
        }
        
        /* 10. Background Ambience */
        .stApp::before {
            content: "";
            position: fixed;
            top: -20%; left: -10%;
            width: 50%; height: 50%;
            background: rgba(30, 58, 138, 0.2);
            filter: blur(100px);
            border-radius: 50%;
            z-index: 0;
            pointer-events: none;
        }
        </style>
    """, unsafe_allow_html=True)

def render_tactical_hero(main_text="METATUNE", sub_text="DASHBOARD", subtitle="Tactical Repository Reconnaissance"):
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1 style="
                font-family: 'Inter', sans-serif;
                font-weight: 900;
                font-size: 4rem;
                font-style: italic;
                transform: skewX(-6deg);
                margin: 0;
            ">
                <span style="color: white; text-shadow: 0 0 10px rgba(255,255,255,0.2);">{main_text}</span>
                <span style="color: #3b82f6; margin: 0 10px;">//</span>
                <span style="
                    background: linear-gradient(to right, #3b82f6, white);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    filter: drop-shadow(0 0 15px rgba(59,130,246,0.6));
                    padding-right: 15px;
                ">{sub_text}</span>
            </h1>
            <p style="
                color: rgba(59, 130, 246, 0.6);
                font-family: 'JetBrains Mono', monospace;
                letter-spacing: 0.2em;
                text-transform: uppercase;
                margin-top: 1rem;
            ">
                {subtitle}
            </p>
        </div>
    """, unsafe_allow_html=True)
