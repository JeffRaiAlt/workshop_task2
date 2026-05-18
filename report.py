import pandas as pd


def build_markdown_report(df: pd.DataFrame, notes: str, mode: str) -> str:
    title = "# Автоматический отчет по данным\n"
    mode_section = f"**Режим обработки:** {'графовый (source/target/weight)' if mode == 'graph' else 'обычный (табличный)'}\n"
    shape_section = f"- Количество строк: **{len(df)}**\n- Количество колонок: **{df.shape[1]}**\n"
    dtypes_md = df.dtypes.reset_index()
    dtypes_md.columns = ['column', 'dtype']
    dtypes_table = dtypes_md.to_markdown(index=False)
    head_table = df.head(20).to_markdown(index=False)
    report = '\n'.join([
        title,
        mode_section,
        '## Общая информация',
        shape_section,
        '### Примечания по предобработке',
        notes if notes else '- Нет дополнительных примечаний.',
        '',
        '## Типы колонок',
        dtypes_table,
        '',
        '## Пример данных',
        head_table,
    ])
    return report
