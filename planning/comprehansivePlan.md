# Steam Game Market Intelligence Engine — Project Plan

## Problem Statement

> Which game genres, tag combinations, and price points represent the highest-revenue opportunities for publishers entering the Steam market — and can we build a recommendation engine that surfaces underserved niches backed by real player behavior data?

## Why This Project

This project is designed to be a portfolio centrepiece for data scientist roles (Revolut, Canonical, Clarity AI, and similar), following the principles in Bysani's *"The Data Portfolio That Actually Gets You Hired"*. Every design decision below maps back to that article's advice.

| Article Principle | How This Project Addresses It |
|---|---|
| *"Answer a business question, don't just make predictions"* | The core deliverable is a ranked list of market opportunities with revenue estimates — not a model accuracy score |
| *Real business problems > toy datasets* | The question "what should we build next?" is one every game publisher actually pays consultants to answer |
| *End-to-end pipeline: messy data from APIs, cleaning, analysis, models, communication* | Three distinct APIs (Steam, SteamSpy, RAWG), each with different auth, rate limits, and data quality issues |
| *Industry-specific analysis* | Directly relevant to gaming/entertainment/tech companies; demonstrates domain knowledge |
| *No Kaggle beginner datasets* | Data is self-collected via APIs — no CSV download. The market gap analysis framing has no existing Kaggle competition |
| *Quantified business impact with stated assumptions* | Revenue estimates derived from real ownership × price data, with explicit assumption documentation |
| *Professional code quality: functions, docstrings, organised structure* | Modular Python package, not a monolithic notebook |
| *3-5 quality projects; this should be one of them* | Scoped as one anchor project that demonstrates breadth (engineering, ML, analysis, communication) |

---

## Data Sources & Pipeline

### Source 1: Steam Web API (Player Behavior)

**What it provides:** User-game ownership and playtime (minutes) for public profiles; friend lists for graph traversal.

**Endpoints:**
- `IPlayerService/GetOwnedGames` — games owned + playtime per user
- `ISteamUser/GetFriendList` — friend graph for BFS/DFS crawling
- `ISteamUser/GetPlayerSummaries` — profile visibility check (skip private profiles)

**Collection strategy:**
1. Start from 5-10 seed Steam IDs (your own account, public community hub users)
2. BFS through friend graphs, collecting owned games for each public profile
3. Target: **5,000-10,000 users** with public profiles (realistic over 3-5 days of polite crawling)
4. Rate limiting: ~100K requests/day is safe; implement exponential backoff
5. Store raw JSON, then flatten to a `user_id | app_id | playtime_forever` table

**Data quality issues to document:**
- Selection bias: friend-graph crawling over-represents socially connected users
- Survivorship bias: only public profiles are visible
- Playtime ≠ enjoyment (idle hours, AFK farming)
- Document these honestly in the README — this is a strength, not a weakness

### Source 2: SteamSpy API (Market Data)

**What it provides:** Estimated owners, average/median playtime, price, CCU (concurrent users) per game.

**Endpoints:**
- `/api.php?request=appdetails&appid=X` — per-game stats
- `/api.php?request=all` — bulk summary (limited fields)

**Collection strategy:**
1. First pull the bulk `/all` endpoint for a complete game list
2. Then pull detailed stats for every game that appears in your user dataset
3. Rate limit: 4 requests/second max
4. Target: detailed data for **all games in your user dataset** (likely 10,000-30,000 unique games)

**Key fields:** `owners` (range string like "200,000 .. 500,000"), `average_forever`, `median_forever`, `price` (cents), `positive`, `negative`

**Data quality issues to document:**
- Owner counts are estimates with known error margins (±20-30% for smaller titles)
- Price is current price, not historical — doesn't account for sales/bundles
- State these assumptions when computing revenue estimates

### Source 3: RAWG API (Game Metadata)

**What it provides:** Genres, tags, platforms, developers/publishers, Metacritic scores, ESRB ratings, release dates, screenshots.

**Endpoints:**
- `/api/games/{id}` — full game details
- `/api/games` — search/list with filters

**Collection strategy:**
1. For each unique game in your Steam dataset, look up by name or Steam app ID
2. Free tier: 20,000 requests/month — cache aggressively, run over multiple days if needed
3. Fuzzy match game titles between Steam and RAWG (they won't always align perfectly — document your matching logic)

**Key fields:** `genres`, `tags`, `platforms`, `metacritic`, `released`, `developers`, `publishers`, `ratings_count`

**Data quality issues to document:**
- Not all Steam games exist in RAWG — document your match rate
- Tag/genre taxonomies differ slightly between sources — document your mapping

### Pipeline Architecture

```
steam_market_intel/
├── README.md                          # < 500 words, business-first (see below)
├── requirements.txt
├── environment.yml
├── config.example.yaml                # API key placeholders — NEVER commit real keys
│
├── src/
│   ├── __init__.py
│   ├── collectors/
│   │   ├── steam_api.py               # Steam Web API client + BFS crawler
│   │   ├── steamspy_api.py            # SteamSpy client with rate limiting
│   │   └── rawg_api.py               # RAWG client with fuzzy matching
│   ├── processing/
│   │   ├── clean.py                   # Deduplication, type casting, null handling
│   │   ├── merge.py                   # Join three sources on app_id
│   │   └── features.py               # Feature engineering for models
│   ├── models/
│   │   ├── recommender.py            # Hybrid collaborative + content-based
│   │   ├── market_gaps.py            # Niche identification + revenue estimation
│   │   └── price_analysis.py         # Price-segment demand modelling
│   ├── evaluation/
│   │   ├── metrics.py                # Standard (P@K, NDCG) + revenue-weighted
│   │   └── validation.py             # Train/test split strategy for implicit feedback
│   └── visualisation/
│       ├── market_map.py             # Genre/tag landscape visualisation
│       ├── niche_explorer.py         # Interactive niche opportunity charts
│       └── dashboard.py              # Streamlit app (optional)
│
├── notebooks/
│   ├── 01_data_collection_log.ipynb  # Walkthrough of collection process + issues
│   ├── 02_eda.ipynb                  # Exploratory analysis with insights
│   ├── 03_recommender.ipynb          # Model development + evaluation
│   ├── 04_market_analysis.ipynb      # Niche identification + revenue estimates
│   └── 05_findings.ipynb             # Executive summary with key visuals
│
├── data/
│   ├── raw/                          # .gitignore'd — instructions to reproduce
│   ├── processed/                    # Cleaned, merged dataset
│   └── sample/                       # Small sample for reviewers to run code
│
└── results/
    ├── figures/                       # Exported key visualisations (PNG/SVG)
    └── tables/                        # Summary statistics, top niches, etc.
```

This structure follows the article's guidance on organised code: *"Functions with docstrings, descriptive variable naming, comments explaining reasoning, organised structure (avoid massive notebooks)."* The notebooks tell the analytical story; the `src/` modules contain reusable, tested code.

---

## Analysis Plan

### Phase 1: Data Collection & Cleaning (Week 1-2)

**Objective:** Build the merged dataset from three sources.

**Steps:**
1. Set up API clients with caching, rate limiting, and retry logic
2. Crawl Steam friend graph → collect user-game-playtime data
3. Enrich with SteamSpy market data and RAWG metadata
4. Clean and merge: handle missing values, map IDs across sources, deduplicate
5. Produce a clean `games` table (one row per game, all features) and a `user_games` table (one row per user-game pair)

**Documented deliverables:**
- Notebook `01` walking through collection decisions, problems encountered, and solutions
- Data quality report: match rates between sources, missingness by field, bias discussion

### Phase 2: Exploratory Data Analysis (Week 2-3)

**Objective:** Understand the Steam market landscape before modelling.

**Key questions to answer with visualisations:**
- What does the distribution of games by genre/tag look like? (long tail expected)
- How does playtime correlate with review scores and ownership?
- What's the relationship between price and ownership across genres?
- Which genres/tags are growing (recent releases) vs. stagnant?
- What does the "typical" successful indie game look like vs. the typical failure?

**Visualisation targets (the article flags "no visualisations" as a common mistake):**
- Genre/tag co-occurrence heatmap
- Revenue distribution by genre (violin plots or ridgeline)
- Scatter: median playtime vs. estimated owners (coloured by genre)
- Time series: game releases by genre over years (are certain niches growing?)
- Tag network graph: which tags cluster together in successful games?

### Phase 3: Recommendation Engine (Week 3-4)

**Objective:** Build a hybrid recommender that works for both established and cold-start games.

**Approach:**

*Collaborative Filtering (established games):*
- Implicit feedback matrix: users × games, values = playtime (normalised)
- Algorithm: ALS (Alternating Least Squares) via `implicit` library — well-suited for implicit feedback, scales to your dataset size
- Why ALS over neural approaches: transparency, explainability, and the article warns against unnecessary complexity / trendy library usage

*Content-Based (cold-start fallback):*
- Feature vector per game from RAWG metadata: genre one-hot, tag TF-IDF, price bucket, platform flags, Metacritic score
- Cosine similarity between game feature vectors
- For games with < N user interactions, fall back to content-based

*Hybrid combination:*
- Weighted blend: α × collaborative score + (1-α) × content score
- α scales with the number of interactions a game has (more data → trust collaborative more)

**Evaluation — this is where the project differentiates itself:**

*Standard metrics (expected by reviewers):*
- Precision@K, Recall@K, NDCG@K
- Leave-one-out evaluation on held-out user-game pairs

*Revenue-weighted metric (novel, business-relevant):*
- Instead of treating all hits equally, weight by game price × estimated ownership
- Rationale: recommending a $30 game that converts is more valuable than recommending a free game
- Formula: `Revenue-Weighted Hit Rate = Σ(hit_i × price_i × P(conversion_i)) / Σ(price_i × P(conversion_i))`
- This is the metric a publisher actually cares about

*What to emphasise in the README (per article's advice on precision-recall tradeoffs):*
- Don't report accuracy — it's meaningless for implicit feedback
- Show P@K vs R@K tradeoff curves
- Discuss the cold-start performance gap explicitly: "For games with >100 interactions, hybrid achieves P@10 of X. For cold-start games (<10 interactions), content-based achieves P@10 of Y, vs. Z for a popularity baseline."

### Phase 4: Market Gap Analysis (Week 4-5) — THE HEADLINE

**Objective:** Identify underserved niches with quantified revenue opportunity.

This is the part no existing portfolio project does, and it's the core business value.

**Method:**

1. **Define the tag-combination space.** Every game has multiple tags (e.g., "roguelike", "co-op", "narrative"). Generate all pairwise and triple-wise tag combinations that have ≥ N games (to avoid noise from ultra-rare combos).

2. **For each niche (tag combination), compute:**
   - `supply`: number of games in this niche
   - `demand_proxy`: total estimated owners across all games in the niche
   - `engagement`: median playtime-per-owner (how sticky are games here?)
   - `satisfaction`: median review score (positive / (positive + negative))
   - `median_revenue`: median (estimated owners × price) per game
   - `competition_intensity`: supply / demand_proxy (how crowded is it?)

3. **Score niches by opportunity.** The key insight: a high-opportunity niche has high demand, high engagement, high satisfaction, but low supply. Compute an opportunity score:
   ```
   opportunity = (demand_proxy × engagement × satisfaction) / supply
   ```
   Rank niches by this score. The top niches are your "underserved markets."

4. **Estimate revenue potential for a new entrant.** For each top niche:
   - Take the median revenue of existing games in that niche
   - Adjust for recency (are newer games in this niche doing better or worse?)
   - State assumption: "A new game entering this niche would likely achieve revenue between the 25th and 75th percentile of existing titles, or $X - $Y"
   - This follows the article's impact quantification template: *"If implemented at a company with [parameters], this model could [outcome]."*

5. **Produce a final ranked table:**
   | Rank | Niche (Tags) | # Games | Est. Total Owners | Median Playtime/Owner | Median Revenue | Opportunity Score | Est. Revenue Range (New Entrant) |
   |------|-------------|---------|-------------------|-----------------------|----------------|-------------------|-------------------------------|
   | 1 | roguelike + narrative + co-op | 12 | 2.1M | 48h | $890K | 8.4 | $340K - $1.2M |
   | 2 | ... | ... | ... | ... | ... | ... | ... |

   *(Numbers are illustrative — real numbers come from the data.)*

### Phase 5: Price Sensitivity Analysis (Week 5)

**Objective:** Model the relationship between price point and commercial success by genre.

**Approach:**
- Within each major genre, bin games by price ($0-5, $5-10, $10-15, $15-20, $20-30, $30+)
- For each bin: compute median ownership, median revenue, median review score
- Visualise as grouped bar charts or heatmaps
- Fit a simple log-linear model: `log(owners) ~ price + genre + review_score + platform_count`
- Report elasticity estimates by genre: "In the indie RPG segment, a $5 price increase is associated with a X% decrease in estimated ownership"

**Honest caveats (state explicitly — the article values honesty over inflated claims):**
- This is observational, not causal. Games are priced based on quality, so there's endogeneity.
- You cannot claim "lowering price by $5 will increase sales by X%"
- You can say: "Games in this genre priced at $10-15 tend to have higher total revenue than those at $20-25, suggesting this price range may be a sweet spot — though this could reflect quality differences rather than price sensitivity alone"
- If asked in an interview: describe the experiment you'd design to get causal estimates (randomised pricing, which Steam actually does with regional pricing)

### Phase 6: Communication & Polish (Week 5-6)

**Objective:** Package everything for a 3-minute hiring manager review.

---

## README Template (< 500 words)

Following the article's required sections:

```markdown
# Steam Game Market Intelligence Engine

**One-line summary:** A multi-source data pipeline and recommendation system
that identifies underserved game market niches with quantified revenue estimates.

## Problem Statement

Game publishers allocating development budgets need data-driven answers to:
which genre/tag combinations have strong player demand but few competing titles,
and what revenue can a new entrant realistically expect?

## Business Impact

Analysis of [N] games and [M] player profiles reveals [X] underserved niches
where median per-title revenue exceeds $[Y], with competition intensity below
the platform average. The top opportunity — [specific niche] — has [Z] estimated
total players across only [W] titles, suggesting significant unmet demand.

## Methodology

- Collected player behavior data from Steam Web API (N users via friend-graph
  crawling), market data from SteamSpy (M games), and metadata from RAWG API
- Built hybrid recommendation engine (ALS collaborative filtering + content-based
  fallback) evaluated on revenue-weighted hit rate
- Identified market gaps by scoring tag combinations on demand, engagement,
  satisfaction, and supply
- Modelled price-segment demand across genres

## Key Findings

- [Finding 1: top underserved niche with numbers]
- [Finding 2: recommendation engine cold-start improvement]
- [Finding 3: price sensitivity insight]
- [Finding 4: engagement vs. revenue disconnect in specific genre]

## Action Items

- Publishers targeting [niche X] can expect $A-$B revenue at $C price point
- The $10-15 price range maximises total revenue for indie [genre Y]
- [Specific recommendation 3]

## Tech Stack

Python (pandas, numpy, scikit-learn, implicit, matplotlib, seaborn, plotly),
Steam Web API, SteamSpy API, RAWG API, Streamlit (dashboard)

## How to Run

[Instructions referencing config.example.yaml and sample data]
```

---

## Resume Line

Following the article's template: *"Built [solution] to [problem]. Achieved [metrics]. Impact: [business outcome]. [Link]"*

> Built a multi-source market intelligence pipeline (Steam API + SteamSpy + RAWG) analysing [N] games and [M] player profiles to identify underserved game market niches. Developed a hybrid recommendation engine with revenue-weighted evaluation. Identified [X] high-opportunity niches with estimated per-title revenue of $[Y]-$[Z]. [GitHub link]

---

## Checklist (from the article)

### GitHub Requirements
- [ ] README under 500 words with all 5 required sections
- [ ] Commented, organised code with docstrings
- [ ] `requirements.txt` AND `environment.yml` included
- [ ] No API keys or passwords (use `config.example.yaml`)
- [ ] `.gitignore` covers `data/raw/`, `config.yaml`, `*.pyc`, etc.
- [ ] Sample data included so reviewers can run notebooks without API keys

### Project Quality
- [ ] Business context is the headline, not model architecture
- [ ] Mix of Python, SQL (if using SQLite for storage), visualisation, communication
- [ ] End-to-end: data collection → cleaning → analysis → models → recommendations
- [ ] No copied tutorials — the market gap analysis is original
- [ ] Assumptions stated explicitly alongside every revenue estimate

### Presentation Standards
- [ ] Professional visualisations (consistent colour scheme, labelled axes, no default matplotlib)
- [ ] Concise writing — the notebook narrative should be scannable
- [ ] All results quantified with confidence intervals or ranges where appropriate
- [ ] Specific, actionable recommendations (not "further research needed")
- [ ] Easy navigation: numbered notebooks, clear module structure

### Professional Polish
- [ ] Error-free writing (run through a spell checker)
- [ ] All links functional
- [ ] Code runs end-to-end on sample data without errors
- [ ] LinkedIn project description matches GitHub README

---

## Timeline

| Week | Phase | Key Deliverable |
|------|-------|----------------|
| 1 | API clients + data collection | Raw data from all 3 sources |
| 2 | Cleaning + merging + EDA starts | Merged dataset, initial visualisations |
| 3 | EDA completion + recommender | Notebook 02-03, trained recommender with evaluation |
| 4 | Market gap analysis | Ranked niche table with revenue estimates |
| 5 | Price analysis + polish | Notebook 04-05, all figures exported |
| 6 | README, dashboard (optional), final review | Publishable repository |

---

## Interview Talking Points

This project generates natural discussion topics for data science interviews:

- **Recommendation systems:** Implicit vs. explicit feedback, cold-start strategies, ALS vs. neural approaches, evaluation beyond accuracy
- **Causal inference:** Why the price analysis is observational, what experiment you'd design, difference-in-differences for sale events
- **A/B testing:** How you'd validate the recommender in production, metric selection, sample size for revenue-weighted tests
- **Business sense:** Translating model output into actionable strategy, communicating uncertainty to non-technical stakeholders
- **Data engineering:** Multi-source pipelines, API rate limiting, data quality tradeoffs, caching strategies
- **Statistics:** Selection bias in friend-graph sampling, confidence intervals on SteamSpy estimates, multiple comparisons when ranking niches