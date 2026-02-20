# Development Notes

Running commentary on decisions, problems, and fixes throughout the project.

---

## 2026-02-18: Project scaffolding

- Created full project structure following the plan in `comprehansivePlan.md`
- Built all three API clients (Steam, SteamSpy, RAWG), processing pipeline, models, evaluation, visualisation modules, and 5 notebooks
- All imports verified passing with `.venv/bin/python`

## 2026-02-18: Steam crawl — seed discovery problem

**Problem:** First crawl attempt returned 0 users, 0 game rows. The placeholder seed ID `76561198000000000` in `config.example.yaml` wasn't a real profile, so the BFS had nowhere to go.

**Fix 1:** Added better logging to the crawler — now logs whether each profile is public/private, whether games/friends are visible, and queue size after seeding. This turned a silent failure into a diagnosable one.

**Fix 2:** User's own Steam account was brand new (1 game, no friends), so even with a real seed ID the BFS had no edges to traverse.

**Fix 3:** Added `discover_seeds()` method to `SteamAPIClient` that pulls member IDs from popular Steam community groups (SteamClientBeta, tradingcards, steamreviews, SteamLabs) via their XML member list endpoints. This gives ~200 public seed profiles with large friend networks. Integrated into `stage_steam()` with `auto_discover_seeds` config flag.

After these fixes, the crawl ran successfully with 40K+ queue entries within minutes.

## 2026-02-18: API key leaked in logs

**Problem:** The 401 errors from `GetFriendList` (users with public profiles but private friend lists) included the full URL with `key=...` in the warning message.

**Fix:** Added key redaction in `SteamAPIClient._get()` — replaces `self._key` with `"REDACTED"` in the error string before logging. User should regenerate their Steam API key since the old one appeared in console output.

## 2026-02-18: Logging only to console

**Problem:** Logs only printed to stdout — no persistent file for reviewing after long crawls.

**Fix:** Updated `setup_logging()` in `utils.py` to write to both console and `logs/collect.log` (append mode). Added `logs/` to `.gitignore`.

## 2026-02-18: CSV only written at end of crawl

**Problem:** `user_games.csv` was only written after `client.crawl()` returned, meaning no data was available on disk during the multi-hour crawl.

**Fix:** Added `on_checkpoint` callback parameter to `crawl()`. The orchestrator passes a callback that writes the CSV every 100 users (same cadence as the JSON checkpoint). Also documented that the checkpoint JSON can be used to extract data mid-crawl.

## 2026-02-18–19: Steam crawl completed

- **10,000 users** collected (hit target)
- **50,005 unique games**, **8.6M game rows**
- Median 194 games/user — active Steam users with real libraries
- 53.9% zero-playtime rows (expected — bundle/sale purchases never launched)
- Queue grew to 2.4M+ entries (BFS through dense social graph)

## 2026-02-19: SteamSpy stage completed

- **50,005 / 50,005 games** collected — 100% coverage
- Owner estimates: 100% populated, heavy right-skew (median 10K, max 150M)
- Price: 93.8% populated, 15.6% free-to-play
- Reviews (positive/negative): 100% populated
- Playtime (average/median): 100% populated

## 2026-02-19: RAWG stage — hit free tier limit

- 50,005 games to match via fuzzy title search
- Free tier rate limit: 1 req/sec, up to 2 requests per game (search + detail)
- After ~7 hours, hit the 20,000 requests/month free tier cap → all requests started returning 401 Unauthorized
- Killed the process; checkpoint had saved progress
- **Result: 9,951 matched, 1,799 unmatched out of 11,750 processed** (84.7% match rate). 38,255 remaining
- Also added API key redaction to RAWG client (same issue as Steam — keys were visible in error log URLs)
- Decision: proceed with partial RAWG coverage. SteamSpy has 100% coverage for core market metrics. RAWG adds genres/tags/Metacritic — 85% match rate on the processed games is sufficient for niche analysis. Can backfill when monthly quota resets.

## 2026-02-19: Clean & merge stage completed

- Made `stage_clean` resilient to interrupted RAWG collection: falls back to checkpoint file if `rawg_metadata.json` doesn't exist, converts it automatically
- Also handles the case where no RAWG data exists at all (proceeds without metadata)
- Fixed Pandas deprecation warning (`date_format="iso"` in `to_json`)
- **Output stats:**
  - Games table: 50,005 rows, 19.9% with RAWG metadata
  - User-games table: 8,625,136 rows, 10,000 users, 50,005 games
  - Playtime outlier threshold: 153,916 min (2,565 hrs), 8,626 rows flagged
  - Metacritic coverage: 5.9% (most games don't have Metacritic scores — expected for indie/small titles)
- All processed data saved to `data/processed/`

## 2026-02-19: Tag/genre format mismatch

**Problem:** SteamSpy provides `tags` as dicts (`{tag_name: vote_count}`) and `genre` as comma-separated strings. RAWG provides `tags_rawg` and `genres` as lists. Feature engineering expected `list[str]` for both.

**Fix:** Added `_normalise_tags_and_genres()` to `src/processing/merge.py`:
- SteamSpy tags: dict → sorted list (by vote count descending)
- SteamSpy genre: comma-separated string → list
- RAWG used as fallback for games with no SteamSpy data
- Strategy: SteamSpy primary (40,866 games with tags, 46,095 with genres) over RAWG (9,833)

Also fixed `build_game_features()` in `features.py` — used `pd.concat(parts, axis=1)` instead of repeated column assignment to avoid DataFrame fragmentation warnings. Added null-safety for `MultiLabelBinarizer` on the `platforms` column (80% NaN due to partial RAWG coverage).

Also fixed `compute_price_segments()` and `compute_genre_elasticities()` in `price_analysis.py` — both had a duplicate-column bug when `explode("genres").rename(columns={"genres": "genre"})` conflicted with the existing SteamSpy `genre` column. Fixed by dropping the original before renaming.

## 2026-02-19: Feature engineering completed

- Game feature matrix: 50,005 games × 295 features (40 genres, 200 tag TF-IDF, 7 price buckets, platform flags, metacritic, review score)
- Interaction matrix (playtime ≥ 1 min): 8,641 users × 47,104 items, 3.97M non-zero (0.98% density)
- Filtered out 54% zero-playtime rows — owned-but-never-played is not a positive signal

## 2026-02-19: Recommender training and evaluation

**Scripts:** `python -m src.train`

### Results (leave-one-out evaluation, 8,505 test users)

| Model                  | P@10   | NDCG@10 | Rev-Weighted HR@10 |
|------------------------|--------|---------|---------------------|
| Popularity baseline    | 0.0089 | 0.0608  | 0.0423              |
| ALS Collaborative      | 0.0175 | 0.1080  | 0.1025              |
| Hybrid (all)           | 0.0152 | 0.0978  | 0.0885              |
| Hybrid (warm ≥100)     | 0.0167 | 0.1076  | 0.0974              |
| Hybrid (cold <100)     | 0.0000 | 0.0000  | 0.0000              |

**Key findings:**
- ALS CF is the strongest signal — **77% improvement** over popularity baseline on NDCG@10
- CF revenue-weighted HR@20 = 14.8% vs popularity 4.8% — **3× improvement in surfacing high-value games**
- Hybrid slightly underperforms pure CF because content features (genre/tag/price) are too generic at 295 dimensions to add signal on top of 4M collaborative interactions
- **Cold-start is zero** — with 47K candidate items and only 1 held-out test item, content features alone can't rank it in the top 20. Richer features (descriptions, screenshots, embeddings) would be needed
- ALS trained in 2.3 seconds (64 factors, 15 iterations) — very fast

**Honest takeaway for the portfolio:** CF dominates when you have rich interaction data. The content-based component needs more expressive features to contribute meaningfully. This is a realistic finding — cold-start remains an open problem.

## 2026-02-19: Market gap analysis completed

**Script:** `python -m src.analyse`

- **140,136 niches** identified from pairwise and triple-wise tag combinations (min 5 games per niche)
- Niche descriptors built from 4.6M tag combination records in 203 seconds
- Scored by opportunity = (demand × engagement × satisfaction) / supply

### Top 5 niches by opportunity score:

1. **Multiplayer + Open World** — 690 games, 1.47B total players, opportunity 0.0034
2. **Multiplayer + Tactical + e-sports** — 6 games, 202M players, opportunity 0.0033
3. **Action + Multiplayer** — 2,526 games, 3.3B players, opportunity 0.0030
4. **Action + Tactical + e-sports** — 6 games, 200M players, opportunity 0.0030
5. **PvP + Tactical + e-sports** — 6 games, 200M players, opportunity 0.0030

Revenue estimates generated for top 15 niches with recency adjustments.

## 2026-02-19: Price sensitivity analysis completed

- **207 genre × price_bin combinations** computed
- Global log-linear model: R² = 0.18, positive price coefficient (+0.74%/$1)
  - Positive coefficient reflects quality-price endogeneity: better games charge more AND sell more
  - This is observational, not causal — explicitly documented
- **27 genre-specific elasticities** computed
- Most price-sensitive: Free To Play (-1.87%/$1), Violent (-1.77%), Gore (-1.60%)
- Optimal price bin is $30+ for all major genres — again reflecting the quality correlation, not a pricing recommendation

## 2026-02-19: Visualisations generated

**Script:** `python -m src.visualise`

8 charts saved to `results/figures/`:
- `genre_cooccurrence_heatmap.png` — 40×40 genre co-occurrence matrix
- `revenue_by_genre_violin.png` — log-scale revenue distributions for top 15 genres
- `playtime_vs_owners_scatter.png` — engagement vs. scale, coloured by genre
- `releases_over_time.png` — genre growth trends 2005–2025
- `niche_bubble_chart.html` — interactive supply vs. demand (Plotly)
- `opportunity_distribution.png` — opportunity score histogram with 95th percentile
- `revenue_range_comparison.png` — p25–p75 revenue bars for top 15 niches
- `niche_metrics_heatmap.png` — normalised scorecard for top 20 niches

## 2026-02-20: Phase 7 — Tests and dashboard enhancement

### Test suite (82 tests)

Created `tests/` directory with 6 test files covering all core modules:

- `test_clean.py` (14 tests): deduplication, type casting, outlier flags, NaN handling for `clean_user_games()`, `clean_steamspy()`, `clean_rawg()`
- `test_merge.py` (8 tests): three-source join correctness, RAWG metadata flag, tag/genre normalisation (dict→list, comma-separated→list)
- `test_features.py` (13 tests): feature matrix shape, genre/tag binary encoding, interaction matrix sparsity, niche descriptor computation
- `test_models.py` (12 tests): ALS collaborative filter, content-based filter, hybrid recommender — all on tiny synthetic interaction matrices
- `test_market_gaps.py` (11 tests): niche scoring normalisation [0,1], revenue estimation ordering, recency trend computation
- `test_metrics.py` (14 tests): textbook IR metrics (precision@K, recall@K, NDCG@K, revenue-weighted hit rate) with known correct outputs, edge cases (empty lists, k > list length)

All tests use synthetic data only — no API keys or network calls.

**Bug found and fixed during testing:** `np.True_ is True` evaluates to `False` in Python — numpy booleans are not Python singletons. Changed identity checks (`is True`) to equality checks (`== True`) in merge tests.

### Dashboard enhancement

Enhanced `src/visualisation/dashboard.py` from 4 tabs to 5:

1. **Overview** — collection metrics, revenue distribution (unchanged)
2. **Market Niches** — added genre and tag dropdown filters, minimum revenue slider, sortable opportunity table, interactive Plotly bubble chart
3. **Recommender** (new tab) — model comparison bar chart (P@10, NDCG@10, Rev-HR@10), highlights CF vs popularity baseline improvement
4. **Price Analysis** — switched from matplotlib to Plotly (unchanged logic)
5. **Data Quality** — structured two-column layout with dataset overview and coverage notes

Dashboard loads from pre-computed `data/processed/` and `results/` files — no model fitting at runtime.

Also uncommented `streamlit>=1.28` in `requirements.txt` and added `pytest>=7.0`.

## Process note

All data transformations must go through scripts in `src/`, not ad-hoc commands. If a one-off fix is needed (e.g. converting a checkpoint), it should be added to the relevant stage in `src/collect.py` so it's reproducible.
