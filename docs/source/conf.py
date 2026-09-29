import tomllib
from pathlib import Path

project = "Pasteur"
author = "Krv Labs"
copyright = "2026, Krv Labs"

_root = Path(__file__).resolve().parent.parent.parent
with (_root / "Cargo.toml").open("rb") as file:
    release = tomllib.load(file)["workspace"]["package"]["version"]

extensions = [
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
]

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Pasteur Documentation"
html_css_files = ["custom.css"]
pygments_style = "github-dark"
pygments_dark_style = "github-dark"

SANS = '"Geist","Geist Fallback",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif'
MONO = '"Geist Mono","SF Mono",ui-monospace,Menlo,Consolas,monospace'

_light_vars = {
    "color-brand-primary": "#bf6209",
    "color-brand-content": "#bf6209",
    "color-background-primary": "#ffffff",
    "color-background-secondary": "#f7f7f8",
    "color-background-border": "#ededed",
    "color-foreground-primary": "#0a0a0a",
    "color-foreground-secondary": "#525252",
    "color-sidebar-background": "#ffffff",
    "color-sidebar-background-border": "#ededed",
    "color-sidebar-item-background--current": "#fdf3e7",
    "color-code-background": "#0d1117",
    "color-code-foreground": "#e6edf3",
    "color-inline-code-background": "#f7f7f8",
    "font-stack": SANS,
    "font-stack--monospace": MONO,
    "docs-bg": "#ffffff",
    "docs-soft": "#f7f7f8",
    "docs-line": "#ededed",
    "docs-line-strong": "#e2e2e2",
    "docs-ink": "#0a0a0a",
    "docs-ink-2": "#525252",
    "docs-ink-3": "#8f8f8f",
    "docs-accent": "#d9730d",
    "docs-accent-ink": "#bf6209",
    "docs-shadow": "rgba(16,24,40,0.08)",
}

_dark_vars = {
    "color-brand-primary": "#f0a868",
    "color-brand-content": "#f0a868",
    "color-background-primary": "#0a0a0a",
    "color-background-secondary": "#141414",
    "color-background-border": "#262626",
    "color-foreground-primary": "#ededed",
    "color-foreground-secondary": "#a3a3a3",
    "color-sidebar-background": "#0a0a0a",
    "color-sidebar-background-border": "#1f1f1f",
    "color-sidebar-item-background--current": "#2a1d0e",
    "color-code-background": "#0d1117",
    "color-code-foreground": "#e6edf3",
    "color-inline-code-background": "#1c1c1c",
    "font-stack": SANS,
    "font-stack--monospace": MONO,
    "docs-bg": "#0a0a0a",
    "docs-soft": "#141414",
    "docs-line": "#262626",
    "docs-line-strong": "#333333",
    "docs-ink": "#ededed",
    "docs-ink-2": "#a3a3a3",
    "docs-ink-3": "#737373",
    "docs-accent": "#f0a868",
    "docs-accent-ink": "#f0a868",
    "docs-shadow": "rgba(0,0,0,0.55)",
}

html_theme_options = {
    "light_logo": "pasteur-logo.svg",
    "dark_logo": "pasteur-logo-dark.svg",
    "sidebar_hide_name": False,
    "light_css_variables": _light_vars,
    "dark_css_variables": _dark_vars,
    "source_repository": "https://github.com/Krv-Labs/pasteur",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "Krv Labs",
            "url": "https://krv.ai",
            "html": "Built by Krv Labs →",
            "class": "docs-footer-krv",
        },
    ],
}
