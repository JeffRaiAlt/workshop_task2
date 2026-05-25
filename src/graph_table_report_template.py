from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(enabled_extensions=("html",)),
)


def render_report_template(template_name: str, **context: object) -> str:
    """Рендерит отдельный HTML-шаблон, автоматически экранируя текстовые значения."""
    return TEMPLATE_ENV.get_template(template_name).render(**context)
