# CSE 158 Assignment 2

Team members:
- Hajin Park
- Albert Ho
- Joey Kim
- Charlie Zhu

## Project Overview
A song-to-user recommendation system using the PDMX dataset. Given a new song, the system predicts which users would enjoy it and returns a ranked top-K list.

## Repository Structure

```
cse-158-assignment2/
├── 158 _ 258 2025 Assignment 2.pdf
├── README.md
├── environment.yml
├── results/
│   ├── diagrams/
│   ├── evaluation_results.csv
│   └── user_profiles.json
├── submission/
│   └── video_url.txt
├── video_script.md
├── workbook.ipynb
└── workbook.py
```

## Key Files

### workbook.py (Main Implementation)
Complete music recommendation system with 5 sections:

**Section 1: Task Definition & Data Loading**
- Loads PDMX.csv with 254,077 songs
- Problem: Song-to-user recommendation for cold-start songs
- Creates organized output directories (`results/`, `results/diagrams/`)

**Section 2: Exploratory Analysis**
- Feature selection: 4 musical features
  - `complexity` - Harmonic/melodic complexity (0-3)
  - `pitch_class_entropy` - Pitch variety/chromaticism
  - `scale_consistency` - Adherence to scale/key
  - `groove_consistency` - Rhythmic regularity
- MinMaxScaler normalization to [0,1] range
- Generates distribution and correlation visualizations

**Section 3: Synthetic Users & Modeling**
- 500 users across 10 archetypes (Classical Purist, Jazz Enthusiast, etc.)
- 7 Recommender models:
  1. `RandomRecommender` - Baseline random selection
  2. `PopularityRecommender` - Most active users
  3. `GenreRecommender` - Genre matching
  4. `ContentBasedRecommender` - Cosine similarity on 4-feature vectors
  5. `KNNItemRecommender` - k-NN with user aggregation (k=20)
  6. `MatrixFactorizationRecommender` - SVD with content-to-factor mapping (f=50)
  7. `HybridRecommender` - Weighted combination (0.3, 0.35, 0.35)

**Section 4: Evaluation**
- Metrics: Precision@K, Recall@K, MAP@K, NDCG@K, Hit Rate, Coverage
- Train/test split: 80/20 cold-start simulation
- Statistical significance testing with paired t-tests

**Section 5: Demo & Related Work**
- `demonstrate_recommendation()` function for live demos
- Discussion of prior work on PDMX and music recommendation

## Feature Engineering Analysis

### Selected Features (4)
| Feature | Description | Range | Rationale |
|---------|-------------|-------|-----------|
| complexity | Harmonic/melodic complexity | 0-3 | Captures musical sophistication |
| pitch_class_entropy | Pitch variety | 0-3.58 | Measures chromaticism |
| scale_consistency | Adherence to key | 0.58-1.0 | Indicates tonal clarity |
| groove_consistency | Rhythmic regularity | 0.38-1.0 | Measures beat stability |

### Tested but Excluded Features
| Feature | Issue | Impact |
|---------|-------|--------|
| n_tracks | High correlation with complexity (-0.54), extreme outliers (max=71 vs median=1) | Degraded performance |
| notes_per_bar | High correlation with n_tracks (0.67), outliers (max=4231 vs median=6.7) | Compressed normalization |
| song_length.bars | Moderate correlation with entropy (0.36), outliers (max=32329) | Noisy signal |

**Conclusion**: The 4 selected features provide the best balance of information content, low correlation, and well-behaved distributions. Adding the 3 extra features decreased MAP@10 by ~10% due to outlier compression.

## Model Performance (Final Results)

| Model | MAP@10 | NDCG@10 | Hit Rate | Coverage |
|-------|--------|---------|----------|----------|
| Random | 0.0795 | 0.0507 | 0.4427 | 1.000 |
| Popularity | 0.1386 | 0.0969 | 0.5995 | 0.100 |
| Genre-Based | 0.0691 | 0.0449 | 0.4328 | 0.428 |
| Content-Based | 0.0942 | 0.0596 | 0.4594 | 0.962 |
| k-NN (k=20) | 0.1062 | 0.0722 | 0.5045 | 1.000 |
| **Matrix Factorization** | **0.1427** | **0.1000** | **0.6003** | 0.256 |
| Hybrid | 0.1261 | 0.0867 | 0.5698 | 0.948 |

**Best Model**: Matrix Factorization (f=50)
- Highest MAP@10 (0.1427) and Hit Rate (60%)
- SVD with content-to-factor mapping handles cold-start effectively

## Dependencies
```
pandas numpy matplotlib seaborn scikit-learn scipy
```

## Running the Project
```bash
conda env create -f environment.yml
conda activate cse-158-assignment2
python workbook.py
```
Output will be saved to `results/` and `results/diagrams/`.

## Key Technical Decisions
1. **Synthetic Users**: PDMX lacks user IDs, so we generate 500 synthetic users with defined preferences
2. **Cold-Start Handling**: Content-to-factor mapping in MF enables new song recommendations
3. **Feature Selection**: 4 features chosen based on correlation analysis and normalization quality
4. **Hybrid Weights**: Content (0.3), k-NN (0.35), MF (0.35) - empirically tuned