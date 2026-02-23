# Submission Checklist

## Before submission
- Confirm `results/full_report.md` is up to date.
- Confirm `results/main_dashboard.png` and `results/ablations_dashboard.png` exist.
- Confirm `results/metrics.csv` and `results/metrics_ablations.csv` exist.
- Confirm presentation notes are updated:
  - `docs/presentation_slides.md`
  - `docs/presentation_talk_track.md`

## Reproducibility checks
- Run `./run.sh setup`
- Run `./run.sh sweep`
- Run `./run.sh ablations`
- Run `./run.sh report`
- Run `./run.sh package`

## What to submit
- `results/submission_package.zip`
- Slides (exported from `docs/presentation_slides.md` or your slide tool)
- Presentation recording/live deck as required by course policy

## Final sanity checks
- Delta CI section appears in report.
- Failure mode breakdown appears in report.
- Figures referenced in slides match the latest outputs.
- No encoding artifacts in report/slides text.
