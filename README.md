# Steam Game Market Intelligence Engine

**One-line summary:** A multi-source data pipeline and recommendation system that identifies underserved game market niches with quantified revenue estimates.

## Problem Statement

Game publishers allocating development budgets need data-driven answers to: which genre/tag combinations have strong player demand but few competing titles, and what revenue can a new entrant realistically expect?

## Business Impact

*To be populated after data collection and analysis are complete.*

## Methodology

- Collected player behaviour data from Steam Web API (N users via friend-graph BFS crawling), market data from SteamSpy (M games), and metadata from RAWG API
- Built hybrid recommendation engine (ALS collaborative filtering + content-based fallback) evaluated on revenue-weighted hit rate
- Identified market gaps by scoring tag combinations on demand, engagement, satisfaction, and supply
- Modelled price-segment demand across genres

## Key Findings

*To be populated after analysis.*

## Action Items

*To be populated after analysis.*

## Tech Stack

Python (pandas, numpy, scikit-learn, implicit, matplotlib, seaborn, plotly), Steam Web API, SteamSpy API, RAWG API

## How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt` (or `conda env create -f environment.yml`)
3. Copy `config.example.yaml` to `config.yaml` and add your API keys:
   - **Steam Web API key:** Register at https://steamcommunity.com/dev/apikey (requires a Steam account)
   - **RAWG API key:** Sign up at https://rawg.io/apidocs (free tier: 20,000 requests/month)
   - SteamSpy requires no API key
4. Run the data collection pipeline:
   ```bash
   python -m src.collect                  # all stages
   python -m src.collect --stage steam    # Steam crawl only
   python -m src.collect --stage steamspy # SteamSpy enrichment
   python -m src.collect --stage rawg     # RAWG metadata
   python -m src.collect --stage clean    # Clean & merge
   ```
5. Explore the notebooks in order: `notebooks/01_*.ipynb` through `notebooks/05_*.ipynb`
6. A small sample dataset is included in `data/sample/` for running notebooks without API keys
