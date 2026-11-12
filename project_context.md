# 12-Hour AI Developer Sprint: HengSeng Sentiment Agent

**Instructions for the User:** 
Do NOT feed this entire document to the AI at once. Feed the "System Rules" first to set the behavior. Then, copy and paste each "Phase" prompt one by one. Wait for the AI to finish, test the code, commit the changes, and then move to the next phase.

---

## INITIALIZATION: Set the System Rules
*Copy and paste this into the main context or system prompt of your AI assistant:*

> "You are an expert Python Quantitative Developer and DevOps Engineer. We are building the 'HengSeng Sentiment Price-Prediction Agent' in a 12-hour sprint. 
> 
> **CRITICAL RULES FOR THIS SESSION:**
> 1. **Modular Code:** Write clean, modular, documented Python code (PEP 8 standard). Do not put everything in one file. Follow a strict `src/` directory structure.
> 2. **No Look-Ahead Bias:** When building the data merging and model, you must ensure tomorrow's data is NEVER used to predict tomorrow's price. 
> 3. **Git Commits:** After completing *every single step* or fixing a bug, you must output the exact `git add` and `git commit` commands using Conventional Commits (e.g., `feat(ingestion): add Apify scraper`). We need a dense, professional commit history.
> 4. **No Placeholders:** Write the actual code. Use `vaderSentiment` for fast, local sentiment scoring to save time. 
> Say 'ACKNOWLEDGED' if you understand the rules, and we will begin Phase 1."

---

## PROMPT SEQUENCE

### Phase 1: Scaffold & Config (Hour 1)
*Prompt for the AI:*
> "Phase 1: Scaffold the project. 
> 1. Create a `requirements.txt` containing: pandas, numpy, yfinance, apify-client, exa-py, scikit-learn, lightgbm, vaderSentiment, streamlit, plotly, pyyaml, python-dotenv.
> 2. Create a `.env.example` file with placeholder API keys for Apify and Exa.
> 3. Create a `config/assets.yaml` containing just 2 test assets for now to speed up dev: Tencent (0700.HK) and Samsung (005930.KS). Include their ticker, name, and search keywords.
> 4. Create the folder structure: `src/ingestion`, `src/processing`, `src/modeling`, `src/reporting`, `data/raw`, `data/processed`.
> Please generate the code and files, then provide the git commit commands."

### Phase 2: Exa News Ingestion (Hour 2)
*Prompt for the AI:*
> "Phase 2: Build `src/ingestion/exa_news.py`. 
> 1. Write a function `fetch_recent_news(ticker, company_name, days_back=10)` using `exa-py`.
> 2. The query should look for recent financial news, earnings, and market impact.
> 3. Return a Pandas DataFrame with columns: `date, ticker, title, url, summary, source`.
> 4. Add a `__main__` block that tests the function for '0700.HK' and saves the output to `data/raw/news_mock.csv`. 
> Provide the code and the git commit commands."

### Phase 3: Apify Social Ingestion (Hour 3-4)
*Prompt for the AI:*
> "Phase 3: Build `src/ingestion/apify_social.py`.
> 1. Write a function `scrape_social_mentions(ticker, keywords, days_back=10)`.
> 2. We will use the Apify X/Twitter scraper (actor ID placeholder from .env). 
> 3. Map the raw JSON response into a Pandas DataFrame with columns: `date, ticker, platform, text, likes, retweets, views, author_followers`.
> 4. Write a script to save this to `data/raw/social_mock.csv`.
> Provide the code and the git commit commands."

### Phase 4: Price Data & Sentiment Processing (Hour 5-6)
*Prompt for the AI:*
> "Phase 4: Build `src/processing/features.py` and `src/ingestion/price_data.py`.
> 1. Price: Use `yfinance` to pull daily Open, High, Low, Close, Volume for the last 200 days.
> 2. Sentiment: Write a function that uses `vaderSentiment` to score the text in our social and news dataframes (adding a `sentiment_score` column from -1 to 1).
> 3. Engagement Weighting: Write a function that calculates an `engagement_weight` = `log(1 + likes + retweets) * log(1 + author_followers)`. Then calculate the daily `engagement_weighted_sentiment` for each ticker.
> Provide the code and the git commit commands."

### Phase 5: The Master Merge & Target Creation (Hour 7-8)
*Prompt for the AI:*
> "Phase 5: Build `src/modeling/build_targets.py`. This is critical.
> 1. Merge the Daily Price data, Daily Aggregate Social Sentiment, and Daily News Sentiment on `date` and `ticker`.
> 2. Create technical features: RSI (14), 10-day volatility, 5-day return.
> 3. Create the Target: `target_next_return` = `(Close tomorrow - Close today) / Close today`. 
> 4. **CRITICAL:** Shift the target correctly. Row for Date N must contain features from Date N and the target return for Date N+1. Drop the last row since we don't know tomorrow's close yet.
> Provide the code and the git commit commands."

### Phase 6: Walk-Forward Model (Hour 9-10)
*Prompt for the AI:*
> "Phase 6: Build `src/modeling/train_model.py`.
> 1. Build a LightGBM Regressor to predict `target_next_return`.
> 2. Implement walk-forward validation (do NOT use standard train_test_split or random shuffling). Train on days 1 to T, test on T+1. Move window forward. Do this for the last 10 days of our dataset.
> 3. Calculate directional accuracy (did it predict up/down correctly?) and RMSE.
> 4. Output a CSV: `outputs/predictions.csv` with columns: `date, ticker, current_close, predicted_next_return, predicted_next_close, direction_signal`.
> Provide the code and the git commit commands."

### Phase 7: The 'Standout' Streamlit UI (Hour 11)
*Prompt for the AI:*
> "Phase 7: Build `src/reporting/dashboard.py` using Streamlit. Make it look like a high-end Bloomberg Terminal (dark mode by default).
> 1. Page Title: 'HengSeng Quantitative Sentiment Signals'.
> 2. Layout: A sidebar for selecting the ticker.
> 3. Top Row: 4 Metric Cards (Current Price, Predicted Price, Signal (Bullish/Bearish), Social Sentiment Score).
> 4. Main Chart: Use `plotly.graph_objects` to plot a Candlestick chart of the last 30 days, with a secondary Y-axis showing a bar chart of the daily `engagement_weighted_sentiment`.
> 5. Bottom Table: A clean dataframe showing recent news headlines and their sentiment scores.
> Provide the code and the git commit commands."

### Phase 8: Final JSON Signal & README (Hour 12)
*Prompt for the AI:*
> "Phase 8: Polish and Documentation.
> 1. Create a script `src/reporting/generate_signals.py` that takes the latest predictions and saves them as a structured JSON file in `outputs/signals.json` (as per the blueprint).
> 2. Write a highly professional `README.md`. Include a 'Quickstart' guide, an explanation of our 'Zero Look-Ahead Bias' architecture, and a 'Methodology' section explaining the engagement-weighted sentiment.
> Provide the code and the final git commit commands."