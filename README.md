# Steam Game Market Intelligence Engine

**One-line summary:** A multi-source data pipeline and recommendation system that identifies underserved game market niches with quantified revenue estimates.

## Problem Statement

Game publishers allocating development budgets need data-driven answers to: which genre/tag combinations have strong player demand but few competing titles, and what revenue can a new entrant realistically expect?

## Business Impact

- Identified **140,136 market niches** from 50,005 Steam games, ranked by a composite opportunity score
- The ALS recommendation engine surfaces high-value games **3x more effectively** than a popularity baseline (revenue-weighted HR@20: 14.8% vs 4.8%)
- Top underserved niches (e.g. "Multiplayer + Open World", "Adventure + Open World") show new-entrant revenue potential of $200K–$21M

## Methodology

- Collected player behaviour data from Steam Web API (10,000 users via friend-graph BFS crawling), market data from SteamSpy (50,005 games), and metadata from RAWG API (9,951 matches)
- Built hybrid recommendation engine (ALS collaborative filtering + content-based fallback) evaluated on precision@K, NDCG@K, and a novel revenue-weighted hit rate
- Identified market gaps by scoring 140K tag combinations on demand, engagement, satisfaction, and supply
- Modelled price sensitivity across 27 genres with log-linear regression and per-genre elasticity estimates

## Key Findings

**1. Market Gaps:** The highest-opportunity niches are in multiplayer/open-world and tactical/e-sports spaces. "Multiplayer + Open World" (690 existing games, 1.47B total players) and "Adventure + Open World" ($217K–$20M estimated new-entrant revenue, 2.9x recency trend) stand out.

**2. Recommendation Engine:** ALS collaborative filtering on implicit playtime feedback achieves NDCG@10 = 0.108 — a 77% improvement over the popularity baseline. Content-based features (genre, tag TF-IDF, price) are too generic to improve on CF alone; cold-start remains unsolved without richer content signals.

**3. Price Sensitivity:** The global price model (R² = 0.18) shows a positive price-ownership association (+0.74%/$1) — reflecting quality-price endogeneity, not a causal pricing effect. The most price-sensitive categories are Free-to-Play (-1.87%/$1) and Massively Multiplayer (-1.38%/$1).

## Action Items

1. Publishers targeting **Multiplayer + Open World** can expect $0–$21M revenue at a ~$13.50 price point, with a strong 2.0x recency trend indicating growing demand
2. **Adventure + Open World** shows the strongest recency signal (2.9x) with a healthy $14.41 average price — newer entrants are outperforming older titles
3. **Multiplayer + Shooter** has a 3.9x recency multiplier — the strongest growth signal in the dataset — but is already highly competitive (789 games)
4. Price analysis is observational: the positive price coefficient should not be interpreted as "charge more to sell more" — it reflects that higher-quality games command both higher prices and higher sales

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
5. Run the analysis pipeline:
   ```bash
   python -m src.train      # Train & evaluate recommender
   python -m src.analyse    # Market gap & price analysis
   python -m src.visualise  # Generate all charts
   ```
6. Explore the notebooks in order: `notebooks/01_*.ipynb` through `notebooks/05_*.ipynb`
7. A small sample dataset is included in `data/sample/` for running notebooks without API keys
