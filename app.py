import streamlit as st

from services.inference_service import run_inference, artifacts_status
from ui.components import render_file_uploader, render_inference_controls
from ui.report import build_markdown_report


def main() -> None:
    st.set_page_config(
        page_title="Entity Matching Inference",
        layout="wide",
    )

    st.title("Entity Matching Inference")

    status = artifacts_status()

    with st.expander("Artifacts status"):
        st.json(status)

    input_df = render_file_uploader()

    if input_df is None:
        st.info("Загрузите CSV-файл для запуска инференса.")
        return

    st.subheader("Input preview")
    st.dataframe(input_df.head(50), use_container_width=True)

    mode, score_threshold = render_inference_controls()

    if st.button("Run inference", type="primary"):
        with st.spinner("Running inference..."):
            try:
                matches = run_inference(
                    input_df=input_df,
                    mode=mode,
                    score_threshold=score_threshold,
                )
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return

        st.subheader("Matches")
        st.dataframe(matches, use_container_width=True)

        st.subheader("Report")
        st.markdown(build_markdown_report(matches))


if __name__ == "__main__":
    main()