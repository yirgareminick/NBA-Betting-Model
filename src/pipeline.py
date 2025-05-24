from datetime import date
from prefect import flow, task

# ── placeholder tasks you’ll flesh out later ─────────────────────────
@task
def ingest_raw(run_date: date): ...
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