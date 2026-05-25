import pandas as pd


def build_markdown_report(matches: pd.DataFrame, notes: str = "", mode: str = "table") -> str:
    lines = ["# 📊 Inference Report", ""]

    if notes:
        lines += ["## 📋 Параметры запуска", "", notes, ""]

    if matches.empty:
        lines.append("_Совпадения не найдены._")
        return "\n".join(lines)

    lines += [f"## 🔗 Найдено совпадений: {len(matches)}", ""]

    if "score" in matches.columns:
        lines += [
            f"- Максимальный score: **{matches['score'].max():.4f}**",
            f"- Средний score: **{matches['score'].mean():.4f}**",
            f"- Минимальный score: **{matches['score'].min():.4f}**",
            "",
        ]

    lines += ["## 📄 Пример результатов (первые 20 пар)", ""]
    lines.append(matches.head(20).to_markdown(index=False))

    return "\n".join(lines)
