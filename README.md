# Understanding Cancellation Behavior: An Unsupervised Approach to User Segmentation in Ride-Hailing

Self-reported cancellation reasons are unreliable. A user who cancels "driver too far" 8 seconds after assignment is behaviorally very different from one who waits 10 minutes after the driver has already arrived. This project ignores the reason codes entirely and asks a simpler question: **what kind of canceler is this person, based on what they actually do?**

## What's in the notebook

Synthetic dataset of 2M orders across 300K users and 6 behavioral archetypes. The full analysis pipeline:

- Data synthesis (vectorized, no row loops — runs in ~20s)
- 13 behavioral features engineered on a 90-day retrospective window
- Algorithm selection: DBSCAN vs KMeans, elbow + silhouette analysis
- KMeans with k=6 (fit on 30K stratified sample, batch-predicted on full population)
- Cluster profiling, PCA visualization, radar charts
- ARI stability check across random seeds
- Risk tier mapping + business recommendations

## Archetypes

| Archetype | ~% of users | What they do |
|---|---|---|
| Good users | 65% | Normal usage, low cancel rate |
| Casual cancelers | 12% | Occasional cancels, no clear pattern |
| Serial cancelers | 10% | Quickly cancel after assignment, high volume |
| Baiters | 6% | Cancel after driver arrives |
| Fare hunters | 5% | Cancel then reorder, heavy promo usage |
| Suspected colluders | 2% | Concentrated orders with 1-2 specific drivers |

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook cancellation_unsupervised_ml.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/) — no additional installs needed, runs end-to-end in under 3 minutes on a free runtime.

## Notes

- All data is synthesized — no real user or trip data
- `RANDOM_STATE = 42` throughout for reproducibility
- Feature engineering deliberately avoids pandas `.rolling()` for speed on large DataFrames
