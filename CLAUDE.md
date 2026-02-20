# CLAUDE.md — Steam Market Intelligence Engine

## Project Overview

A multi-source data pipeline and recommendation system that identifies underserved game market niches with quantified revenue estimates. Built as a portfolio project for data scientist roles.

Three API sources (Steam Web API, SteamSpy, RAWG) → cleaning/merging → hybrid recommender → market gap analysis → price sensitivity modelling.

## Quick Reference

```bash
# Run data pipeline
.venv/bin/python -m src.collect                  # all stages
.venv/bin/python -m src.collect --stage steam     # BFS crawl only
.venv/bin/python -m src.collect --stage steamspy  # SteamSpy enrichment
.venv/bin/python -m src.collect --stage rawg      # RAWG metadata
.venv/bin/python -m src.collect --stage clean     # Clean & merge

# Streamlit dashboard
.venv/bin/streamlit run src/visualisation/dashboard.py
```

## Project Structure

```
src/
├── collect.py               # CLI orchestrator (entry point: python -m src.collect)
├── utils.py                 # Config loading, RateLimiter, DiskCache, logging setup
├── collectors/
│   ├── steam_api.py         # Steam Web API client + BFS friend-graph crawler
│   ├── steamspy_api.py      # SteamSpy client (owner estimates, price, reviews)
│   └── rawg_api.py          # RAWG client (genres, tags, platforms, Metacritic)
├── processing/
│   ├── clean.py             # Per-source cleaners (dedup, type casting, outlier flags)
│   ├── merge.py             # 3-source join on app_id + data quality report
│   └── features.py          # Genre one-hot, tag TF-IDF, interaction matrix, niche descriptors
├── models/
│   ├── recommender.py       # ALS collaborative + content-based cosine + hybrid blend
│   ├── market_gaps.py       # Niche opportunity scoring + revenue estimation
│   └── price_analysis.py    # Price segments, log-linear model, genre elasticities
├── evaluation/
│   ├── metrics.py           # P@K, Recall@K, NDCG@K, revenue-weighted hit rate
│   └── validation.py        # Leave-one-out split, cold-start split, popularity baseline
└── visualisation/
    ├── market_map.py        # Heatmaps, violin plots, scatter, time series (matplotlib/seaborn)
    ├── niche_explorer.py    # Bubble chart, opportunity dist, revenue ranges (plotly)
    └── dashboard.py         # Streamlit app

notebooks/                   # Run in order: 01 → 05
data/raw/                    # API responses + cache (gitignored, large)
data/processed/              # Cleaned CSV/JSON tables (gitignored, reproducible)
data/sample/                 # Small sample for running without API keys
results/figures/             # Exported PNG/HTML visualisations
results/tables/              # CSV summary tables (top niches, elasticities, etc.)
logs/                        # collect.log (gitignored)
```

## Environment

- **Python**: 3.11+ (uses `str | Path` union syntax)
- **Virtual env**: `.venv/` — activate with `source .venv/bin/activate`
- **Install**: `pip install -r requirements.txt` or `conda env create -f environment.yml`
- **Key libraries**: pandas, numpy, requests, scikit-learn, implicit (ALS), thefuzz, matplotlib, seaborn, plotly

## Code Conventions

- **Imports**: absolute from `src` package — `from src.collectors.steam_api import SteamAPIClient`
- **Docstrings**: Google-style with Args/Returns/Raises sections
- **Type hints**: throughout, using Python 3.11+ syntax (`str | None` not `Optional[str]`)
- **Logging**: every module uses `logger = logging.getLogger(__name__)`; logs go to console + `logs/collect.log`
- **Naming**: PascalCase classes, snake_case functions, UPPER_SNAKE constants, `_private` prefix
- **Error handling**: try/except with logging warnings, graceful `None` returns, never silent failures
- **Spelling**: use British English for user-facing text (visualisation, behaviour, colour) — matches existing codebase

## Architecture Patterns

- **Rate limiting + disk caching**: all API clients use `RateLimiter` and `DiskCache` from `src/utils.py`
- **Checkpoint/resume**: long-running crawls save progress to JSON every 100 users; can stop and restart freely
- **Incremental CSV writes**: `on_checkpoint` callback writes CSV at each checkpoint interval
- **Config-driven**: all API keys, rate limits, and paths in `config.yaml` (never committed; template at `config.example.yaml`)
- **API key safety**: keys are redacted in log output; `config.yaml` and `logs/` are gitignored

## Data Flow

1. **Steam crawl** → `data/raw/user_games.csv` (user_id × app_id × playtime)
2. **SteamSpy** → `data/raw/steamspy_details.json` (owners, price, reviews per game)
3. **RAWG** → `data/raw/rawg_metadata.json` (genres, tags, platforms, Metacritic)
4. **Clean stage** → `data/processed/games.csv`, `data/processed/user_games.csv`, `data/processed/data_quality_report.json`
5. **Notebooks 02-04** → `results/tables/` and `results/figures/`

## Development Notes

**Always update `planning/notes.md`** when making changes to the project. This is a running commentary tracking:
- Problems encountered and how they were solved
- Script changes and why they were needed
- Data collection milestones (what completed, key stats)
- Decisions and their rationale

Read `planning/notes.md` at the start of each session to pick up context.

## Important Notes

- **No test suite yet** — validation is done through notebooks. Tests would be a good addition.
- **No CI/CD or linter config** — code quality is manual.
- **API rate limits matter**: SteamSpy (4 req/s), RAWG free tier (1 req/s, 20K/month), Steam (~1.5 req/s safe). The collectors enforce these.
- **Checkpoint files can be large**: the Steam crawl checkpoint stores the full queue (can be millions of IDs). This is expected.
- **SteamSpy owner estimates**: ±20-30% uncertainty. Revenue = owners_mid × current_price. All estimates should state assumptions.
- The `planning/` directory contains the comprehensive project plan and the portfolio article that guides design decisions — consult these for the "why" behind architectural choices.
