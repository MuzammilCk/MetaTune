import uuid
import re

with open('app_wandb.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add import uuid
if 'import uuid' not in content:
    content = content.replace('import textwrap', 'import textwrap\nimport uuid')

# 2. Add defaults
defaults_block = """
    # ── INITIALIZE SESSION STATE ──
    _defaults = {
        'session_id': str(uuid.uuid4()),
        'temp_path': None,
        'system_status': 'idle',
        'file_encoding': 'utf-8',
        'study': None,
        'params': None,
        'ready_to_train': False,
    }
    for _k, _v in _defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    idle_html = '''
    <div style="font-family:var(--font-mono); font-size:10px; letter-spacing:2px; color:var(--dna-green); padding:10px 0; display:flex; align-items:center; gap:8px;">
      <span style="width:6px; height:6px; background:var(--dna-green); border-radius:50%; display:inline-block; box-shadow:0 0 8px var(--dna-green); animation: heartbeat 2s infinite;"></span>
      SYSTEM IDLE — AWAITING DATA
    </div>
    '''
    training_html = '''
    <div style="font-family:var(--font-mono);font-size:9px;letter-spacing:2px;color:var(--neural-amber);padding:8px 0;display:flex;align-items:center;gap:8px;">
      <span style="width:6px;height:6px;background:var(--neural-amber);border-radius:50%;box-shadow:0 0 8px var(--neural-amber);display:inline-block;animation:heartbeat 1.5s infinite;"></span>
      TRAINING IN PROGRESS...
    </div>
    '''
    error_html = '''
    <div style="font-family:var(--font-mono); font-size:10px; letter-spacing:2px; color:var(--quantum-magenta); padding:10px 0; display:flex; align-items:center; gap:8px;">
      <span style="width:6px; height:6px; background:var(--quantum-magenta); border-radius:50%; display:inline-block; box-shadow:0 0 8px var(--quantum-magenta);"></span>
      SYSTEM ERROR — CHECK LOGS
    </div>
    '''
"""
if '_defaults = {' not in content:
    content = content.replace('with st.sidebar:', 'with st.sidebar:' + defaults_block)

# 3. Handle File Uploader and Target Column
# We replace the whole block from "uploaded_file = " to before "SYSTEM STATUS"
old_upload_block = """    # ── KEEP ORIGINAL WIDGET (DO NOT REMOVE) ──
    uploaded_file = st.file_uploader("DROP CSV DATASET", type=['csv'])

    # ── KEEP ORIGINAL WIDGET (DO NOT REMOVE) ──
    target_col = st.text_input("TARGET COLUMN (optional)", help="Leave empty for auto-detection")"""

new_upload_block = """    # ── KEEP ORIGINAL WIDGET (DO NOT REMOVE) ──
    uploaded_file = st.file_uploader("DROP CSV DATASET", type=['csv'])

    if uploaded_file is not None:
        if uploaded_file.size > 200 * 1024 * 1024:
            st.markdown(f'<div class="stAlert stError" style="border-left: 3px solid var(--quantum-magenta);">FILE TOO LARGE ({uploaded_file.size / (1024*1024):.1f}MB). MAXIMUM SIZE IS 200MB.</div>', unsafe_allow_html=True)
            uploaded_file = None
        else:
            st.session_state['temp_path'] = f"temp_{st.session_state['session_id']}.csv"
            try:
                with open(st.session_state['temp_path'], "wb") as f: f.write(uploaded_file.getbuffer())
                try:
                    pd.read_csv(st.session_state['temp_path'], nrows=5, encoding='utf-8')
                    st.session_state['file_encoding'] = 'utf-8'
                except UnicodeDecodeError:
                    pd.read_csv(st.session_state['temp_path'], nrows=5, encoding='latin-1')
                    st.session_state['file_encoding'] = 'latin-1'
                st.markdown(f"<div style='font-family:var(--font-mono);font-size:9px;color:var(--text-dim);'>ENCODING: {st.session_state['file_encoding'].upper()} DETECTED</div>", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="stAlert stError" style="border-left: 3px solid var(--quantum-magenta);">COULD NOT PROCESS FILE: {str(e)}</div>', unsafe_allow_html=True)
                uploaded_file = None

    if uploaded_file is None:
        target_col = st.text_input("TARGET COLUMN (optional)", placeholder="UPLOAD FILE FIRST", disabled=True)
        target_col = None
    else:
        st.markdown('''<div style="font-family:var(--font-mono); font-size:9px; letter-spacing:4px; color:var(--text-dim); text-transform:uppercase; margin-bottom:12px; margin-top:24px;">◈ TARGET COLUMN</div>''', unsafe_allow_html=True)
        columns = pd.read_csv(st.session_state['temp_path'], nrows=0, encoding=st.session_state['file_encoding']).columns.tolist()
        options = ["⟳ AUTO-DETECT (last column)"] + columns
        selected = st.selectbox("Select Target", options=options, label_visibility="collapsed")
        target_col = None if selected == "⟳ AUTO-DETECT (last column)" else selected

        if target_col is not None and target_col not in columns:
            st.markdown('<div class="stAlert stError" style="border-left: 3px solid var(--quantum-magenta);">COLUMN NOT FOUND IN DATASET</div>', unsafe_allow_html=True)
            target_col = None
            st.markdown('<div style="font-family:var(--font-mono);font-size:9px;color:var(--text-dim);margin-top:4px;">REVERTING TO AUTO-DETECT</div>', unsafe_allow_html=True)"""

content = content.replace(old_upload_block, new_upload_block)


# 4. Status Indicator update
old_status_block = """    # ── STATUS INDICATOR (keep original reference) ──
    status_indicator = st.empty()
    status_indicator.markdown(\"""
    <div style="
      font-family:var(--font-mono); font-size:10px; letter-spacing:2px;
      color:var(--dna-green); padding:10px 0;
      display:flex; align-items:center; gap:8px;
    ">
      <span style="
        width:6px; height:6px; background:var(--dna-green);
        border-radius:50%; display:inline-block;
        box-shadow:0 0 8px var(--dna-green);
        animation: heartbeat 2s infinite;
      "></span>
      SYSTEM IDLE — AWAITING DATA
    </div>
    \""", unsafe_allow_html=True)"""

new_status_block = """    # ── STATUS INDICATOR (keep original reference) ──
    status_indicator = st.empty()
    if st.session_state['system_status'] == 'idle':
        status_indicator.markdown(idle_html, unsafe_allow_html=True)
    elif st.session_state['system_status'] == 'training':
        status_indicator.markdown(training_html, unsafe_allow_html=True)
    elif st.session_state['system_status'] == 'error':
        status_indicator.markdown(error_html, unsafe_allow_html=True)

    # ── RESET BUTTON ──
    st.markdown('''
    <div style="margin-top:24px; border-top:1px solid var(--border); padding-top:16px;"></div>
    <style>
    .reset-btn-wrapper .stButton > button {
      background: transparent !important;
      color: var(--text-dim) !important;
      border: 1px solid var(--border) !important;
      animation: none !important;
      box-shadow: none !important;
    }
    .reset-btn-wrapper .stButton > button:hover {
      border-color: var(--text-secondary) !important;
      color: var(--text-secondary) !important;
    }
    </style>
    <div class="reset-btn-wrapper">
    ''', unsafe_allow_html=True)
    if st.button("◈ RESET SESSION"):
        for key in ['temp_path', 'session_id', 'system_status', 'file_encoding', 'study', 'ready_to_train', 'params']:
            st.session_state.pop(key, None)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)"""

content = content.replace(old_status_block, new_status_block)

# 5. Remove "with open(...) f.write(...)" in the main track since we did it above
# We do this using regex because spacing can be tricky
content = re.sub(r'if uploaded_file:\n\s+# Save temp file\n\s+with open\("temp\.csv", "wb"\) as f: f\.write\(uploaded_file\.getbuffer\(\)\)\n\s+# 1\. ANALYSIS ROW', 'if uploaded_file:\n    # 1. ANALYSIS ROW', content)

# 6. Replace string "temp.csv" with st.session_state['temp_path']
content = content.replace('"temp.csv"', "st.session_state['temp_path']")
content = content.replace("'temp.csv'", "st.session_state['temp_path']")

# 7. Try-Except for Main Block
if 'st.session_state[\'system_status\'] = \'training\'' not in content:
    # First we need to find "if start_btn:" and replace it with try block setup
    old_start_btn = "    if start_btn:\n"
    new_start_btn = """    if start_btn:
        st.session_state['system_status'] = 'training'
        try:
"""
    parts = content.split("    if start_btn:\n")
    if len(parts) == 2:
        top_part = parts[0]
        bottom_part = parts[1]
        
        # We need to indent every line in bottom_part by 4 spaces
        indented_bottom_lines = []
        for line in bottom_part.splitlines(True):
            if line.strip(): # if not empty
                indented_bottom_lines.append("    " + line)
            else:
                indented_bottom_lines.append(line)
        
        indented_bottom_lines.append(f'''
        except Exception as e:
            st.session_state['system_status'] = 'error'
            status_indicator.markdown(error_html, unsafe_allow_html=True)
''')
        
        content = top_part + new_start_btn + "".join(indented_bottom_lines)

# One more fix: we replaced 'temp.csv' with st.session_state['temp_path'], but inside the script it is global. 
# It evaluates safely because we made sure it is set.

with open('app_wandb.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("done")
