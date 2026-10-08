"""Shared look for both pages: palette tokens, CSS, and small HTML pieces.

Colours follow the dataviz palette, validated for a dark surface."""

import html

import streamlit as st

PAGE = "#0d0d0d"
SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
ACCENT = "#3987e5"  # the score bar of a retrieved chunk

_CSS = f"""
<style>
.stApp {{
  background: radial-gradient(1100px 480px at 12% -8%, rgba(57,135,229,.20), transparent 60%),
              radial-gradient(900px 420px at 95% 0%, rgba(25,158,112,.12), transparent 55%), {PAGE};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid rgba(255,255,255,.08); }}

.hero-title {{
  font-size: 2.5rem; font-weight: 750; letter-spacing: -0.02em; margin: 0; line-height: 1.1;
  background: linear-gradient(90deg, #6da7ec 0%, #3987e5 40%, #199e70 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  -webkit-text-fill-color: transparent;
}}
.hero-sub {{ color: {INK_2}; margin: .7rem 0 1.4rem 0; font-size: 1.02rem; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {SURFACE}; border: 1px solid rgba(255,255,255,.10) !important; border-radius: 16px;
}}
button[data-baseweb="tab"] {{ font-weight: 600; }}

.section-title {{ color: {INK}; font-weight: 650; font-size: 1.05rem; margin: .2rem 0 .1rem 0; }}
.section-note {{ color: {MUTED}; font-size: .85rem; margin-bottom: .4rem; }}

.hit {{
  background: {PAGE}; border: 1px solid rgba(255,255,255,.08); border-radius: 12px;
  padding: 10px 12px; margin-bottom: 10px;
}}
.hit-top {{ display: flex; align-items: center; gap: 8px; }}
.rank {{
  background: {GRID}; color: {INK}; border-radius: 6px; font-weight: 700; font-size: .78rem;
  min-width: 24px; text-align: center; padding: 1px 6px;
}}
.sim {{ color: {INK}; font-weight: 650; font-size: .92rem; font-variant-numeric: tabular-nums; }}
.bar {{ flex: 1; height: 6px; background: {GRID}; border-radius: 3px; overflow: hidden; }}
.bar > span {{ display: block; height: 100%; border-radius: 3px; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 6px 0; }}
.badge {{ font-size: .7rem; font-weight: 600; padding: 1px 8px; border-radius: 999px; border: 1px solid; }}
.badge.text {{ color: #6da7ec; border-color: rgba(109,167,236,.45); background: rgba(57,135,229,.10); }}
.badge.table {{ color: #e8a24b; border-color: rgba(232,162,75,.45); background: rgba(217,89,38,.12); }}
.badge.picture {{ color: #e58bb0; border-color: rgba(213,81,129,.45); background: rgba(213,81,129,.12); }}
.hit img.picture {{ display: block; max-width: 100%; border-radius: 8px; margin: 6px 0 8px 0; }}
.badge.page {{ color: {INK_2}; border-color: rgba(255,255,255,.18); }}
.badge.agree {{ color: #4cc9a0; border-color: rgba(76,201,160,.45); background: rgba(25,158,112,.12); }}
.heading {{ color: {INK_2}; font-size: .78rem; margin-bottom: 4px; }}
.snippet {{ color: {INK_2}; font-size: .82rem; line-height: 1.4; }}
.hit details summary {{ color: {MUTED}; font-size: .76rem; cursor: pointer; margin-top: 6px; }}
.hit details pre {{
  white-space: pre-wrap; color: {INK_2}; font-size: .78rem; background: {SURFACE};
  border-radius: 8px; padding: 8px; margin-top: 6px;
}}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero-title">{html.escape(title)}</div><p class="hero-sub">{html.escape(subtitle)}</p>',
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>'
        + (f'<div class="section-note">{html.escape(note)}</div>' if note else ""),
        unsafe_allow_html=True,
    )
