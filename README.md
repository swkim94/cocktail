# Cocktail Recommender

A small end-to-end data science project that turns [TheCocktailDB](https://www.thecocktaildb.com/) into:

1. A structured dataset of cocktails and their ingredients
2. A cosine-similarity-based recommender ("if you like X, try Y")
3. A PCA + clustering view that groups cocktails into flavor/spirit-based segments

Built with `pandas`, `NumPy`, `scikit-learn`, and `matplotlib`.

## What it does

**Data collection & structuring** — pulls cocktail records from TheCocktailDB's public API, removes duplicates, and reshapes each drink into a row of ingredients + measures.

**Vectorization** — converts each cocktail's ingredient list into a numeric vector across the full ingredient space (~416 unique ingredients), so cocktails become comparable points in a high-dimensional space.

**Recommendation (cosine similarity)** — L2-normalizes each cocktail vector and takes the dot product between all pairs (`normalize()` + `.dot()`), which is equivalent to cosine similarity. For a given input cocktail, the highest-scoring match (excluding itself) is returned as the recommendation, along with its similarity score.

**Dimensionality reduction & clustering** — reduces the normalized ingredient vectors to 3 components with PCA, then applies Spectral Clustering (`affinity='nearest_neighbors'`) to assign each cocktail to one of 5 groups. Groups are visualized as a 3D scatter plot and labeled after the fact by their dominant ingredient (e.g. "Vodka," "Gin," "Orange Juice").

**Visualization** — ingredient histograms, per-cocktail ingredient "fingerprints," 3D PCA cluster plots, and per-group ingredient profiles reprojected back into the original ingredient space.

## Files

| File | Description |
|---|---|
| `Q3a.py` | Initial data collection experiment — small-batch API pull, dataframe structuring, first pass at vectorization and grouping |
| `Q3b.py` | Vectorizes the full dataset, plots ingredient histograms, and groups cocktails using a nearest-neighbors heuristic |
| `Q3c.py` | Refines the vectorization/grouping pipeline; adds cached intermediate data (`cocktails3.pkl`) to avoid re-scraping |
| `Q3d.py` | Full pipeline: vectorization, top-10 ingredient chart, cosine-similarity recommender with example queries, PCA (3 components), and Spectral Clustering with labeled group visualizations |

`Q3d.py` is the most complete/final version of the pipeline.

## Example output

Input: `Cosmopolitan Martini`
Recommendation: `Cape Codder` (cosine similarity score ≈ 0.75)

## Notes

- Data is cached locally as pickled DataFrames (`cocktails.pkl`, `cocktails3.pkl`) to avoid repeatedly hitting the API.
- This was a personal/learning project built outside of work to practice `pandas`/`scikit-learn` workflows on a real, messy dataset — not built for production use.

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
requests
```
