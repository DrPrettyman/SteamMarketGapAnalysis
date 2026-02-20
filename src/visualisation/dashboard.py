"""Streamlit dashboard for interactive exploration of market intelligence results.

Run:
    streamlit run src/visualisation/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px

PROCESSED_DIR = Path("data/processed")
SAMPLE_DIR = Path("data/sample")
RESULTS_DIR = Path("results")


def _resolve(primary: Path, fallback: Path) -> Path | None:
    """Return primary path if it exists, else fallback, else None."""
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return None


def _read_json(path: Path | None, default=None):
    """Read a JSON file, returning *default* if path is None."""
    if path is None:
        return default if default is not None else {}
    with open(path) as f:
        return json.load(f)


def load_data() -> dict:
    """Load data for the dashboard.

    Prefers full results from data/processed/ and results/tables/.
    Falls back to data/sample/ when full data is unavailable (e.g. on
    Streamlit Community Cloud where only committed files exist).
    """
    data = {}

    # Games
    games_path = _resolve(
        PROCESSED_DIR / "games.json",
        SAMPLE_DIR / "games_sample.json",
    )
    data["games"] = pd.read_json(games_path, lines=True) if games_path else pd.DataFrame()

    # Niches
    niche_path = _resolve(
        RESULTS_DIR / "tables" / "top_niches.csv",
        SAMPLE_DIR / "top_niches.csv",
    )
    data["niches"] = pd.read_csv(niche_path) if niche_path else pd.DataFrame()

    all_niches_path = RESULTS_DIR / "tables" / "all_niches_scored.csv"
    data["all_niches"] = pd.read_csv(all_niches_path) if all_niches_path.exists() else pd.DataFrame()

    # Quality report
    report_path = _resolve(
        PROCESSED_DIR / "data_quality_report.json",
        SAMPLE_DIR / "data_quality_report.json",
    )
    data["quality"] = _read_json(report_path, {})

    # Recommender results
    rec_path = _resolve(
        RESULTS_DIR / "tables" / "recommender_results.json",
        SAMPLE_DIR / "recommender_results.json",
    )
    data["recommender"] = _read_json(rec_path, {})

    # Revenue estimates
    revenue_path = _resolve(
        RESULTS_DIR / "tables" / "revenue_estimates.json",
        SAMPLE_DIR / "revenue_estimates.json",
    )
    data["revenues"] = _read_json(revenue_path, [])

    # Price segments
    segments_path = _resolve(
        RESULTS_DIR / "tables" / "price_segments.csv",
        SAMPLE_DIR / "price_segments.csv",
    )
    data["price_segments"] = pd.read_csv(segments_path) if segments_path else pd.DataFrame()

    return data


def main() -> None:
    """Streamlit app entry point."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(page_title="Steam Market Intelligence", layout="wide")
    st.title("Steam Game Market Intelligence Engine")
    st.markdown(
        "Identifying underserved game market niches with quantified revenue estimates. "
        "Multi-source pipeline: Steam Web API + SteamSpy + RAWG."
    )

    data = load_data()
    games_df = data["games"]

    # --- Sidebar filters ---
    selected_genres = []
    st.sidebar.header("Filters")
    if "genres" in games_df.columns and not games_df.empty:
        all_genres = sorted(set(
            g for genres in games_df["genres"].dropna() for g in genres
            if isinstance(genres, list)
        ))
        selected_genres = st.sidebar.multiselect(
            "Filter by Genre/Tag",
            all_genres,
            help="Filters Overview (games) and Market Niches (niche names)",
        )
        if selected_genres:
            games_df = games_df[
                games_df["genres"].apply(
                    lambda g: any(sg in g for sg in selected_genres)
                    if isinstance(g, list) else False
                )
            ]

    # --- Tabs ---
    tab_overview, tab_niches, tab_recs, tab_price, tab_quality = st.tabs([
        "Overview", "Market Niches", "Recommender", "Price Analysis", "Data Quality",
    ])

    # ── Overview ──────────────────────────────────────────────────────────

    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Games", f"{len(games_df):,}")
        col2.metric("Users Collected", f"{data['quality'].get('users_total', 'N/A'):,}")
        col3.metric("RAWG Match Rate", f"{data['quality'].get('rawg_match_rate', 0):.1%}")
        col4.metric(
            "Median Games/User",
            f"{data['quality'].get('median_games_per_user', 'N/A')}",
        )

        if "estimated_revenue" in games_df.columns:
            st.subheader("Revenue Distribution")
            clipped = games_df["estimated_revenue"].dropna()
            clipped = clipped[clipped > 0].clip(upper=clipped.quantile(0.95))
            fig = px.histogram(
                clipped, nbins=50, labels={"value": "Estimated Revenue ($)"},
                title="Revenue distribution (clipped at 95th percentile)",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Market Niches ─────────────────────────────────────────────────────

    with tab_niches:
        st.subheader("Top Market Opportunities")
        niche_df = data["niches"]
        all_niche_df = data["all_niches"]

        if not niche_df.empty:
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                min_games = st.slider(
                    "Minimum games in niche", 5, 200,
                    value=10, step=5,
                )
            with col2:
                min_revenue = st.number_input(
                    "Minimum median revenue ($)", value=0, step=1000,
                )

            # Apply filters to all niches if available
            display_niches = all_niche_df if not all_niche_df.empty else niche_df

            # Handle both column naming conventions:
            #   top_niches.csv uses "# Games", "Median Revenue ($)"
            #   all_niches_scored.csv uses "supply", "median_revenue"
            games_col = next(
                (c for c in ["# Games", "supply"] if c in display_niches.columns),
                None,
            )
            revenue_col = next(
                (c for c in ["Median Revenue ($)", "median_revenue"] if c in display_niches.columns),
                None,
            )
            if games_col:
                display_niches = display_niches[display_niches[games_col] >= min_games]
            if revenue_col:
                display_niches = display_niches[display_niches[revenue_col] >= min_revenue]

            # Apply genre/tag sidebar filter to niche names
            if selected_genres and "niche" in display_niches.columns:
                display_niches = display_niches[
                    display_niches["niche"].apply(
                        lambda n: any(g.lower() in n.lower() for g in selected_genres)
                    )
                ]

            st.dataframe(display_niches.head(25), use_container_width=True, hide_index=True)

            # Revenue estimates
            if data["revenues"]:
                st.subheader("New-Entrant Revenue Estimates")
                rev_df = pd.DataFrame(data["revenues"])
                display_rev_cols = ["niche", "revenue_low", "revenue_mid", "revenue_high", "avg_price"]
                avail = [c for c in display_rev_cols if c in rev_df.columns]
                st.dataframe(rev_df[avail], use_container_width=True, hide_index=True)

            # Bubble chart
            fig_path = RESULTS_DIR / "figures" / "niche_bubble_chart.html"
            if fig_path.exists():
                st.subheader("Niche Landscape")
                st.components.v1.html(fig_path.read_text(), height=650, scrolling=True)
        else:
            st.info("Run the market gap analysis to populate this tab.")

    # ── Recommender ───────────────────────────────────────────────────────

    with tab_recs:
        st.subheader("Recommendation Engine Performance")
        rec = data["recommender"]

        if rec:
            # Metrics comparison table
            st.markdown("#### Model Comparison")
            models = {}
            for model_name in ["popularity_baseline", "collaborative_filtering",
                               "hybrid_all", "hybrid_warm", "hybrid_cold"]:
                if model_name in rec:
                    models[model_name.replace("_", " ").title()] = rec[model_name]

            if models:
                metrics_df = pd.DataFrame(models).T
                metrics_df.index.name = "Model"
                st.dataframe(
                    metrics_df.style.format("{:.4f}"),
                    use_container_width=True,
                )

                # Bar chart of key metrics
                key_metrics = ["precision@10", "recall@10", "ndcg@10"]
                avail_metrics = [m for m in key_metrics if m in metrics_df.columns]
                if avail_metrics:
                    chart_data = metrics_df[avail_metrics].reset_index()
                    melted = chart_data.melt(id_vars="Model", var_name="Metric", value_name="Score")
                    fig = px.bar(
                        melted, x="Model", y="Score", color="Metric",
                        barmode="group", title="Recommendation Metrics @ K=10",
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # Revenue-weighted metrics
                rev_metrics = [m for m in metrics_df.columns if "revenue_weighted" in m]
                if rev_metrics:
                    st.markdown("#### Revenue-Weighted Hit Rate")
                    st.markdown(
                        "Measures whether the recommender surfaces *high-value* games, "
                        "not just any games."
                    )
                    rev_chart = metrics_df[rev_metrics].reset_index()
                    melted = rev_chart.melt(id_vars="Model", var_name="Metric", value_name="Score")
                    fig = px.bar(
                        melted, x="Model", y="Score", color="Metric",
                        barmode="group",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Metadata
            if "metadata" in rec:
                with st.expander("Training Details"):
                    st.json(rec["metadata"])
        else:
            st.info("Run the recommender pipeline to populate this tab.")

    # ── Price Analysis ────────────────────────────────────────────────────

    with tab_price:
        st.subheader("Price Sensitivity by Genre")
        if not data["price_segments"].empty:
            st.dataframe(data["price_segments"], use_container_width=True, hide_index=True)
        else:
            st.info("Run the price analysis to populate this tab.")

    # ── Data Quality ──────────────────────────────────────────────────────

    with tab_quality:
        st.subheader("Data Quality Report")
        report = data["quality"]
        if report:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Collection Summary")
                st.markdown(f"""
                - **Users crawled:** {report.get('users_total', 'N/A'):,}
                - **Games analysed:** {report.get('games_total', 'N/A'):,}
                - **Median games/user:** {report.get('median_games_per_user', 'N/A')}
                - **Mean games/user:** {report.get('mean_games_per_user', 'N/A')}
                """)
            with col2:
                st.markdown("#### Playtime Statistics")
                st.markdown(f"""
                - **Median playtime:** {report.get('playtime_median_hrs', 'N/A')} hrs
                - **Mean playtime:** {report.get('playtime_mean_hrs', 'N/A')} hrs
                - **Zero playtime:** {report.get('playtime_zero_pct', 0):.1%}
                - **RAWG match rate:** {report.get('rawg_match_rate', 0):.1%}
                """)

            # Missing data
            if "games_missing" in report:
                st.markdown("#### Field Missingness")
                missing_df = pd.DataFrame(
                    list(report["games_missing"].items()),
                    columns=["Field", "Missing %"],
                )
                missing_df["Missing %"] = missing_df["Missing %"].apply(lambda x: f"{x:.1%}")
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run the data pipeline to generate the quality report.")


if __name__ == "__main__":
    main()
