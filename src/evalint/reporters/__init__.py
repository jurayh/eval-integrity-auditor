"""evalint reporters: terminal, HTML (self-contained), JSON."""
from .html import render_html
from .terminal import render_terminal

__all__ = ["render_html", "render_terminal"]
