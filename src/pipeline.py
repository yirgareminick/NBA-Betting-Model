from datetime import date
from prefect import flow, task
from ingest.ingest_games import pull_games

# ── placeholder tasks flesh out later ─────────────────────────
@task
def ingest_raw(run_date: date):
    # for now: seasonal slice ending yesterday
    season_start = f"{run_date.year-1}-10-01"
    season_end   = run_date.isoformat()
    df = pull_games(season_start, season_end)
    print(df.head())
    return df

@task
def build_features(run_date: date): ...
@task
def train_model(run_date: date): ...
@task
def size_bets(run_date: date): ...
@task
def push_picks(run_date: date): ...

# ── master flow ──────────────────────────────────────────────────────
@flow(name="nba-betting-daily")
def main(run_date: date = date.today()):
    raw = ingest_raw(run_date)
    feats = build_features(raw)
    model_artifact = train_model(feats)
    bets = size_bets(model_artifact)
    push_picks(bets)

if __name__ == "__main__":
    main()