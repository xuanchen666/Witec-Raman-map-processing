"""Self-contained HTML pipeline report tying together stage summaries, plots, and CSVs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd

from .paths import _prepare_export_subdir, _sanitize_export_stem

_HTML_STYLE = """
body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.3rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #ccc; padding-bottom: 0.2rem; }
table { border-collapse: collapse; margin: 0.75rem 0 1.5rem 0; font-size: 0.85rem; }
th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
th { background-color: #f2f2f2; }
tr:nth-child(even) { background-color: #fafafa; }
ul.link-list { list-style: none; padding-left: 0; }
ul.link-list li { margin: 0.3rem 0; }
.meta { color: #666; font-size: 0.9rem; }
"""


def _relative_href(target: Path, output_dir: Path) -> str:
    """Return a POSIX-style relative link from `output_dir` to `target`."""
    return Path(os.path.relpath(target, start=output_dir)).as_posix()


def _render_table_section(title: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return f"<h2>{title}</h2><p><em>No data.</em></p>"
    return f"<h2>{title}</h2>{df.to_html(index=False, na_rep='—')}"


def _render_link_section(title: str, links: Mapping[str, Path], output_dir: Path) -> str:
    if not links:
        return f"<h2>{title}</h2><p><em>No files.</em></p>"
    items = "\n".join(
        f'<li><a href="{_relative_href(Path(path), output_dir)}">{label}</a>'
        f' <span class="meta">({Path(path)})</span></li>'
        for label, path in links.items()
    )
    return f'<h2>{title}</h2><ul class="link-list">{items}</ul>'


def build_pipeline_report(
    output_dir: Path,
    sample_name: str,
    stage_tables: Mapping[str, pd.DataFrame],
    plot_links: Mapping[str, Path],
    csv_links: Mapping[str, Path],
) -> Path:
    """Write one self-contained HTML report summarizing the full processing pipeline."""
    output_dir = Path(output_dir)
    report_dir = _prepare_export_subdir(output_dir, "report")
    report_path = report_dir / f"{_sanitize_export_stem(sample_name)}_pipeline_report.html"

    sections = [
        _render_table_section(title, df) for title, df in stage_tables.items()
    ]
    sections.append(_render_link_section("Stage 6 Plot Files", plot_links, report_dir))
    sections.append(_render_link_section("CSV Data Locations", csv_links, report_dir))

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<title>{sample_name} pipeline report</title><style>{_HTML_STYLE}</style></head><body>"
        f"<h1>{sample_name} — Raman Processing Pipeline Report</h1>"
        f'<p class="meta">Generated {generated_at}</p>'
        + "".join(sections)
        + "</body></html>"
    )

    report_path.write_text(html, encoding="utf-8")
    return report_path
