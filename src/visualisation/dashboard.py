"""Streamlit dashboard for interactive exploration of market intelligence results.

Run:
    streamlit run src/visualisation/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load processed data for the dashboard.

    Returns:
        Tuple of (games_df, niche_df, quality_report).
    """
    games_df = pd.read_json(PROCESSED_DIR / "games.json", lines=True)
    niche_path = RESULTS_DIR / "tables" / "top_niches.csv"
    niche_df = pd.read_csv(niche_path) if niche_path.exists() else pd.DataFrame()
    report_path = PROCESSED_DIR / "data_quality_report.json"
    quality_report = {}
    if report_path.exists():
        with open(report_path) as f:
            quality_report = json.load(f)
    return games_df, niche_df, quality_report


def main() -> None:
    """Streamlit app entry point."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(page_title="Steam Market Intelligence", layout="wide")
    st.title("Steam Game Market Intelligence Engine")
    st.markdown("Identifying underserved game market niches with quantified revenue estimates.")

    games_df, niche_df, quality_report = load_data()

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    if "genres" in games_df.columns:
        all_genres = sorted(set(g for genres in games_df["genres"].dropna() for g in genres))
        selected_genres = st.sidebar.multiselect("Genres", all_genres)
        if selected_genres:
            games_df = games_df[
                games_df["genres"].apply(
                    lambda g: any(sg in g for sg in selected_genres)
                    if isinstance(g, list) else False
                )
            ]

    # --- Overview tab ---
    tab_overview, tab_niches, tab_price, tab_quality = st.tabs(
        ["Overview", "Market Niches", "Price Analysis", "Data Quality"]
    )

    with tab_overview:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Games", f"{len(games_df):,}")
        col2.metric("Users Collected", f"{quality_report.get('users_total', 'N/A'):,}")
        col3.metric("RAWG Match Rate", f"{quality_report.get('rawg_match_rate', 0):.1%}")

        if "estimated_revenue" in games_df.columns:
            st.subheader("Revenue Distribution")
            st.bar_chart(games_df["estimated_revenue"].clip(upper=games_df["estimated_revenue"].quantile(0.95)))

    with tab_niches:
        st.subheader("Top Market Opportunities")
        if not niche_df.empty:
            st.dataframe(niche_df.head(25), use_container_width=True)

            # Interactive bubble chart
            fig_path = RESULTS_DIR / "figures" / "niche_bubble_chart.html"
            if fig_path.exists():
                st.components.v1.html(fig_path.read_text(), height=650, scrolling=True)
        else:
            st.info("Run the market gap analysis to populate this tab.")

    with tab_price:
        st.subheader("Price Sensitivity by Genre")
        segments_path = RESULTS_DIR / "tables" / "price_segments.csv"
        if segments_path.exists():
            segments = pd.read_csv(segments_path)
            st.dataframe(segments, use_container_width=True)
        else:
            st.info("Run the price analysis to populate this tab.")

    with tab_quality:
        st.subheader("Data Quality Report")
        if quality_report:
            st.json(quality_report)
        else:
            st.info("Run the data pipeline to generate the quality report.")


if __name__ == "__main__":
    main()
