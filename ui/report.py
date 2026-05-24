import pandas as pd


def build_markdown_report(matches: pd.DataFrame) -> str:
    if matches.empty:
        return "Совпадения не найдены."

    lines = [
        "# Inference report",
        "",
        f"Найдено совпадений: {len(matches)}",
        "",
    ]

    if "score" in matches.columns:
        lines.extend(
            [
                f"Максимальный score: {matches['score'].max():.4f}",
                f"Средний score: {matches['score'].mean():.4f}",
                "",
            ]
        )

    return "\n".join(lines)
