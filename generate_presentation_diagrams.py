"""
Generate all presentation diagrams for CSE 158 Assignment 2.
Clean, professional diagrams suitable for academic presentation.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Setup
DIAGRAMS_DIR = 'results/diagrams'
os.makedirs(DIAGRAMS_DIR, exist_ok=True)

# Global style settings - clean and professional
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
})

# Color palette - limited to 4 main colors
COLORS = {
    'primary': '#2E5090',      # Dark blue
    'secondary': '#5A5A5A',    # Gray
    'accent': '#228B22',       # Green
    'highlight': '#CC4444',    # Red
    'light_bg': '#F5F5F5',     # Light gray background
    'white': '#FFFFFF',
}


def save_figure(fig, filename):
    """Save figure to diagrams directory."""
    path = os.path.join(DIAGRAMS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {path}")


def draw_box(ax, x, y, w, h, text, color=COLORS['primary'], text_color='white', fontsize=10):
    """Draw a simple rectangular box with centered text."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor='#333333', linewidth=1)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color)


def draw_arrow(ax, x1, y1, x2, y2, color='#333333'):
    """Draw a simple arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=color))


def create_cold_start_diagram():
    """Slide 4: Cold-start challenge diagram."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('The Cold-Start Problem', fontsize=14, fontweight='bold', pad=15)

    # New song box (left)
    draw_box(ax, 0.5, 2, 2.5, 1.5, 'NEW SONG\n(No history)', COLORS['secondary'])

    # Question mark area
    ax.text(4.5, 2.75, '?', fontsize=36, ha='center', va='center',
            fontweight='bold', color=COLORS['secondary'])

    # Arrow
    draw_arrow(ax, 3.2, 2.75, 6.8, 2.75)

    # Users (right)
    for i, y in enumerate([3.6, 2.8, 2.0]):
        label = f'User {i+1}' if i < 3 else '...'
        draw_box(ax, 7.5, y-0.25, 2.5, 0.5, label, COLORS['light_bg'], COLORS['secondary'], 9)

    # Labels
    ax.text(1.75, 1.2, 'No interaction data', ha='center', fontsize=9,
            color=COLORS['highlight'], fontweight='bold')
    ax.text(8.75, 4.3, 'Which users would enjoy it?', ha='center', fontsize=10, fontweight='bold')

    # Solution box
    draw_box(ax, 2.5, 0.3, 7, 0.6, 'SOLUTION: Use content features (no history needed)',
             COLORS['accent'], 'white', 10)

    save_figure(fig, 'cold_start_diagram.png')


def create_system_architecture():
    """Slide 5: System architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('System Architecture', fontsize=14, fontweight='bold', pad=15)

    # INPUT box
    draw_box(ax, 0.3, 2.2, 2.2, 1.6, 'INPUT\nNew Song\n+ Features', COLORS['secondary'])

    # Arrow to processing
    draw_arrow(ax, 2.6, 3, 3.4, 3)

    # Processing area - simple border
    proc_box = FancyBboxPatch((3.5, 0.5), 6.5, 5, boxstyle="round,pad=0.02",
                               facecolor=COLORS['light_bg'], edgecolor=COLORS['primary'], linewidth=1)
    ax.add_patch(proc_box)
    ax.text(6.75, 5.2, 'RECOMMENDER MODELS', ha='center', fontsize=11, fontweight='bold')

    # Model boxes - simplified
    models = ['Random', 'Popularity', 'Genre', 'Content', 'k-NN', 'Matrix Fact.']
    for i, name in enumerate(models):
        x = 3.8 + (i % 3) * 2.1
        y = 3.8 if i < 3 else 1.2
        color = COLORS['light_bg'] if i < 2 else COLORS['primary']
        text_col = COLORS['secondary'] if i < 2 else 'white'
        draw_box(ax, x, y, 1.9, 0.9, name, color, text_col, 9)

    # Hybrid box
    draw_box(ax, 5.5, 2.4, 2.5, 1, 'HYBRID\nCombination', COLORS['primary'])

    # Arrow to output
    draw_arrow(ax, 10.2, 3, 10.8, 3)

    # OUTPUT box
    draw_box(ax, 11, 2, 2.5, 2, 'OUTPUT\nTop-K Users', COLORS['accent'])

    save_figure(fig, 'system_architecture.png')


def create_dataset_statistics():
    """Slide 6: Dataset statistics infographic."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('PDMX Dataset Overview', fontsize=14, fontweight='bold', pad=15)

    # Main stats boxes - 4 key numbers
    stats = [
        ('254,077', 'Total Songs', 1.5),
        ('57', 'Feature Columns', 4.5),
        ('4', 'Selected Features', 7.5),
        ('14,182', 'Rated Songs', 10.5),
    ]

    for val, label, x in stats:
        draw_box(ax, x-1.1, 4, 2.2, 1.4, f'{val}\n{label}', COLORS['primary'], 'white', 11)

    # Feature list - simple table format
    ax.text(6, 3.2, 'Selected Musical Features', ha='center', fontsize=12, fontweight='bold')

    features = [
        ('complexity', '0-3', 'Harmonic/melodic complexity'),
        ('pitch_class_entropy', '0-3.58', 'Pitch variety'),
        ('scale_consistency', '0.58-1.0', 'Key adherence'),
        ('groove_consistency', '0.38-1.0', 'Rhythmic regularity'),
    ]

    for i, (name, range_val, desc) in enumerate(features):
        y = 2.5 - i*0.5
        ax.text(1.5, y, name, ha='left', fontsize=10, fontweight='bold')
        ax.text(5.5, y, f'[{range_val}]', ha='center', fontsize=9, color=COLORS['secondary'])
        ax.text(7, y, desc, ha='left', fontsize=9)

    save_figure(fig, 'dataset_statistics.png')


def create_feature_selection_table():
    """Slide 7: Feature selection analysis table."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    ax.set_title('Feature Selection Analysis', fontsize=14, fontweight='bold', pad=15)

    # Selected features table
    selected_data = [
        ['complexity', '0-3', 'Harmonic complexity', 'Low correlation', 'SELECTED'],
        ['pitch_class_entropy', '0-3.58', 'Pitch variety', 'Low correlation', 'SELECTED'],
        ['scale_consistency', '0.58-1.0', 'Key adherence', 'Low correlation', 'SELECTED'],
        ['groove_consistency', '0.38-1.0', 'Rhythmic regularity', 'Low correlation', 'SELECTED'],
    ]

    # Excluded features
    excluded_data = [
        ['n_tracks', 'max=71', 'High corr. (-0.54)', 'Extreme outliers', 'EXCLUDED'],
        ['notes_per_bar', 'max=4231', 'High corr. (0.67)', 'Extreme outliers', 'EXCLUDED'],
        ['song_length.bars', 'max=32329', 'Mod. corr. (0.36)', 'Noisy signal', 'EXCLUDED'],
    ]

    ax.text(0.5, 0.93, 'Selected Features (4)', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=COLORS['accent'])

    table1 = ax.table(cellText=selected_data,
                      colLabels=['Feature', 'Range', 'Description', 'Correlation', 'Status'],
                      loc='upper center', cellLoc='left',
                      bbox=[0.02, 0.55, 0.96, 0.35])
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    for i in range(5):
        table1[(0, i)].set_facecolor(COLORS['primary'])
        table1[(0, i)].set_text_props(color='white', fontweight='bold')
    for row in range(1, 5):
        for col in range(5):
            table1[(row, col)].set_facecolor('white' if row % 2 else COLORS['light_bg'])

    ax.text(0.5, 0.48, 'Excluded Features', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=COLORS['highlight'])

    table2 = ax.table(cellText=excluded_data,
                      colLabels=['Feature', 'Distribution', 'Issue', 'Impact', 'Status'],
                      loc='center', cellLoc='left',
                      bbox=[0.02, 0.15, 0.96, 0.28])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    for i in range(5):
        table2[(0, i)].set_facecolor(COLORS['secondary'])
        table2[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.text(0.5, 0.06, 'Adding excluded features decreased MAP@10 by ~10%',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='center', color=COLORS['highlight'])

    save_figure(fig, 'feature_selection_table.png')



def create_user_archetypes():
    """Slide 9: User archetypes visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_title('Synthetic User Archetypes (500 Users, 10 Types)',
                 fontsize=14, fontweight='bold', pad=15)

    archetypes = [
        ('Classical Purist', 0.8, 0.7, 0.9, 0.8, 'classical'),
        ('Jazz Enthusiast', 0.7, 0.85, 0.6, 0.9, 'jazz'),
        ('Pop Lover', 0.3, 0.5, 0.85, 0.95, 'pop'),
        ('Folk Fan', 0.4, 0.6, 0.9, 0.85, 'folk'),
        ('Rock Aficionado', 0.5, 0.65, 0.75, 0.9, 'rock'),
        ('Electronic Explorer', 0.6, 0.75, 0.7, 0.95, 'electronic'),
        ('Eclectic Listener', 0.5, 0.7, 0.75, 0.85, 'mixed'),
        ('Simple Melodies', 0.2, 0.4, 0.95, 0.9, 'folk/pop'),
        ('Complex Compositions', 0.9, 0.8, 0.7, 0.75, 'classical/jazz'),
        ('Rhythm Focused', 0.5, 0.6, 0.8, 0.98, 'electronic/rock'),
    ]

    table_data = [[name, f'{c:.1f}', f'{e:.1f}', f'{s:.1f}', f'{g:.1f}', genre]
                  for name, c, e, s, g, genre in archetypes]

    table = ax.table(cellText=table_data,
                     colLabels=['Archetype', 'Complexity', 'Entropy', 'Scale', 'Groove', 'Genre'],
                     loc='center', cellLoc='left',
                     bbox=[0.05, 0.1, 0.9, 0.8])

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Header styling
    for i in range(6):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternating row colors
    for row in range(1, 11):
        for col in range(6):
            table[(row, col)].set_facecolor('white' if row % 2 else COLORS['light_bg'])

    ax.text(0.5, 0.03, 'Each user has noise added (std=0.15) for diversity',
            transform=ax.transAxes, fontsize=9, ha='center')

    save_figure(fig, 'user_archetypes.png')


def create_model_overview():
    """Slide 10: Model overview table."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    ax.set_title('Recommender Models Overview', fontsize=14, fontweight='bold', pad=15)

    # Baselines
    baseline_data = [
        ['Random', 'Random user selection', 'None', 'Sanity check'],
        ['Popularity', 'Most active users', 'Activity counts', 'Upper baseline'],
        ['Genre-Based', 'Match genre to user prefs', 'Genre prefs', 'Simple content'],
    ]

    # Advanced models
    advanced_data = [
        ['Content-Based', 'Cosine similarity', '4 features', 'Cold-start friendly'],
        ['k-NN (k=20)', 'Similar songs, aggregate users', 'Similarity', 'Collaborative'],
        ['Matrix Factorization', 'SVD with content mapping', '50 factors', 'Best accuracy'],
        ['Hybrid', 'Weighted combination', '0.30+0.35+0.35', 'Best balance'],
    ]

    ax.text(0.5, 0.93, 'Baseline Models', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=COLORS['secondary'])

    table1 = ax.table(cellText=baseline_data,
                      colLabels=['Model', 'Method', 'Key Parameter', 'Purpose'],
                      loc='upper center', cellLoc='left',
                      bbox=[0.02, 0.62, 0.96, 0.28])
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    for i in range(4):
        table1[(0, i)].set_facecolor(COLORS['secondary'])
        table1[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.text(0.5, 0.55, 'Advanced Models', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=COLORS['accent'])

    table2 = ax.table(cellText=advanced_data,
                      colLabels=['Model', 'Method', 'Key Parameter', 'Strength'],
                      loc='center', cellLoc='left',
                      bbox=[0.02, 0.18, 0.96, 0.35])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    for i in range(4):
        table2[(0, i)].set_facecolor(COLORS['accent'])
        table2[(0, i)].set_text_props(color='white', fontweight='bold')

    save_figure(fig, 'model_overview.png')


def create_content_based_diagram():
    """Slide 11: Content-based filtering diagram."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Content-Based Filtering', fontsize=14, fontweight='bold', pad=15)

    # Step 1: Song features
    draw_box(ax, 0.3, 1.8, 2.8, 1.8, '1. Song Features\n[0.67, 0.75, 0.92, 0.97]',
             COLORS['secondary'], 'white', 10)

    # Arrow 1
    draw_arrow(ax, 3.3, 2.7, 4.0, 2.7)

    # Step 2: Cosine similarity
    draw_box(ax, 4.2, 1.5, 3.2, 2.4, '2. Cosine Similarity\ncos(song, user)\nFor each user',
             COLORS['primary'], 'white', 10)

    # Arrow 2
    draw_arrow(ax, 7.6, 2.7, 8.3, 2.7)

    # Step 3: Ranked output
    draw_box(ax, 8.5, 1.5, 3.2, 2.4, '3. Ranked Users\n1. user_0466\n2. user_0197\n3. user_0280',
             COLORS['accent'], 'white', 10)

    # Key insight
    ax.text(6, 0.5, 'No interaction history needed - works for new songs',
            ha='center', fontsize=10, fontweight='bold', color=COLORS['accent'])

    save_figure(fig, 'content_based_diagram.png')


def create_knn_diagram():
    """Slide 12: k-NN item-based diagram."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('k-NN Item-Based Recommendation', fontsize=14, fontweight='bold', pad=15)

    # Step 1: New song
    draw_box(ax, 0.3, 1.8, 2.5, 1.8, '1. New Song\nFeatures', COLORS['secondary'], 'white', 10)

    # Arrow 1
    draw_arrow(ax, 3.0, 2.7, 3.7, 2.7)

    # Step 2: Find k similar songs
    draw_box(ax, 3.9, 1.5, 3.2, 2.4, '2. Find k=20\nSimilar Songs\nSong_1: 0.95\nSong_2: 0.91',
             COLORS['primary'], 'white', 10)

    # Arrow 2
    draw_arrow(ax, 7.3, 2.7, 8.0, 2.7)

    # Step 3: Aggregate users
    draw_box(ax, 8.2, 1.5, 3.5, 2.4, '3. Aggregate Users\nuser_A: 1.86\nuser_B: 0.88\nuser_C: 1.83',
             COLORS['accent'], 'white', 10)

    # Key insight
    ax.text(6, 0.5, 'Leverages collaborative signals through similar songs',
            ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'])

    save_figure(fig, 'knn_diagram.png')


def create_svd_diagram():
    """Slide 13: Matrix Factorization/SVD diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Matrix Factorization (SVD)', fontsize=14, fontweight='bold', pad=15)

    # Training Phase label
    ax.text(1.5, 5.5, 'Training Phase:', fontsize=11, fontweight='bold')

    # User-Song Matrix
    draw_box(ax, 0.3, 4, 2.5, 1.2, 'User-Song Matrix\n500 x N', COLORS['secondary'], 'white', 9)

    draw_arrow(ax, 3.0, 4.6, 3.7, 4.6)
    ax.text(4.0, 4.6, 'SVD', fontsize=10, fontweight='bold')
    draw_arrow(ax, 4.5, 4.6, 5.2, 4.6)

    # Factors
    draw_box(ax, 5.4, 4, 2.2, 1.2, 'User Factors\n500 x 50', COLORS['primary'], 'white', 9)
    draw_box(ax, 8, 4, 2.2, 1.2, 'Song Factors\nN x 50', COLORS['accent'], 'white', 9)

    # Cold-start handling label
    ax.text(1.5, 3.0, 'Cold-Start Handling:', fontsize=11, fontweight='bold')

    # Content to factor mapping
    draw_box(ax, 0.3, 1.2, 2.5, 1.4, 'New Song\nFeatures', COLORS['secondary'], 'white', 9)

    draw_arrow(ax, 3.0, 1.9, 3.7, 1.9)
    ax.text(4.2, 2.5, 'Ridge Regression', fontsize=9, fontweight='bold')

    draw_box(ax, 4.0, 1.2, 2.2, 1.4, 'Predicted\nSong Factor', COLORS['accent'], 'white', 9)

    draw_arrow(ax, 6.4, 1.9, 7.1, 1.9)
    ax.text(7.5, 2.5, 'Dot Product', fontsize=9)

    draw_box(ax, 8, 1.2, 2.2, 1.4, 'User Scores\nRanked', COLORS['primary'], 'white', 9)

    # Key insight
    ax.text(6, 0.4, 'Learns latent representations; maps new songs via content features',
            ha='center', fontsize=10, fontweight='bold', color=COLORS['accent'])

    save_figure(fig, 'svd_diagram.png')


def create_hybrid_architecture():
    """Slide 14: Hybrid architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Hybrid Recommender Architecture', fontsize=14, fontweight='bold', pad=15)

    # Input song
    draw_box(ax, 0.3, 2, 1.8, 1.4, 'New Song\n+ Features', COLORS['secondary'], 'white', 9)

    # Three model branches
    models = [
        ('Content-Based', 3.8, 0.30),
        ('k-NN', 2.5, 0.35),
        ('Matrix Fact.', 1.2, 0.35),
    ]

    for name, y, weight in models:
        draw_arrow(ax, 2.3, 2.7, 3.0, y+0.35)
        draw_box(ax, 3.2, y, 2.2, 0.7, name, COLORS['primary'], 'white', 9)
        ax.text(5.8, y+0.35, f'w={weight}', fontsize=9)
        draw_arrow(ax, 6.4, y+0.35, 7.0, 2.7)

    # Combiner
    draw_box(ax, 7.2, 2, 2.2, 1.4, 'Weighted\nCombination', COLORS['primary'], 'white', 10)

    # Arrow to output
    draw_arrow(ax, 9.6, 2.7, 10.2, 2.7)

    # Output
    draw_box(ax, 10.4, 2, 1.5, 1.4, 'Ranked\nUsers', COLORS['accent'], 'white', 10)

    # Key insight
    ax.text(6, 0.4, 'Captures different aspects of user-song compatibility',
            ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'])

    save_figure(fig, 'hybrid_architecture.png')


def create_metrics_definitions():
    """Slide 15: Evaluation metrics definitions."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    ax.set_title('Evaluation Metrics', fontsize=14, fontweight='bold', pad=15)

    metrics = [
        ('Precision@K', 'Fraction of recommended users who are relevant',
         'Relevant and Recommended / K'),
        ('Recall@K', 'Fraction of relevant users that are recommended',
         'Relevant and Recommended / Relevant'),
        ('MAP@K', 'Mean Average Precision - rewards higher rankings',
         'Mean of AP scores across test songs'),
        ('NDCG@K', 'Normalized Discounted Cumulative Gain',
         'DCG@K / IDCG@K'),
        ('Hit Rate', 'Fraction of songs with at least one hit',
         'Songs with hits / Test songs'),
        ('Coverage', 'Fraction of users in recommendations',
         'Unique recommended / All users'),
    ]

    table_data = [[name, desc, formula] for name, desc, formula in metrics]

    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'Description', 'Formula'],
                     loc='center', cellLoc='left',
                     bbox=[0.02, 0.15, 0.96, 0.75],
                     colWidths=[0.15, 0.45, 0.40])

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, 7):
        for j in range(3):
            table[(i, j)].set_facecolor('white' if i % 2 else COLORS['light_bg'])

    ax.text(0.5, 0.06, 'Primary: MAP@K, NDCG@K  |  Secondary: Coverage, Hit Rate',
            transform=ax.transAxes, fontsize=10, ha='center', fontweight='bold')

    save_figure(fig, 'metrics_definitions.png')


def create_train_test_split():
    """Slide 16: Train/test split diagram."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Train/Test Split: Cold-Start Simulation', fontsize=14, fontweight='bold', pad=15)

    # Full dataset bar
    ax.text(6, 4.5, 'All Songs with Interactions', ha='center', fontsize=11, fontweight='bold')
    full_bar = FancyBboxPatch((1, 3.6), 10, 0.6, boxstyle="round,pad=0.02",
                               facecolor=COLORS['light_bg'], edgecolor='#333', linewidth=1)
    ax.add_patch(full_bar)

    # Train portion (80%)
    train_bar = FancyBboxPatch((1, 3.6), 8, 0.6, boxstyle="round,pad=0.02",
                                facecolor=COLORS['primary'], edgecolor='#333', linewidth=1)
    ax.add_patch(train_bar)
    ax.text(5, 3.9, '80% Training', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold')

    ax.text(10, 3.9, '20% Test', ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows pointing down
    draw_arrow(ax, 5, 3.4, 5, 2.8)
    draw_arrow(ax, 10, 3.4, 10, 2.8)

    # Training box
    draw_box(ax, 2, 1.5, 5.5, 1.2, 'Training Data\nBuild profiles, k-NN index, MF factors',
             COLORS['primary'], 'white', 9)

    # Test box
    draw_box(ax, 8.2, 1.5, 2.8, 1.2, 'Test Songs\nNo interactions',
             COLORS['highlight'], 'white', 9)

    # Key insight
    ax.text(6, 0.6, 'Test songs have ZERO training interactions - true cold-start evaluation',
            ha='center', fontsize=10, fontweight='bold', color=COLORS['highlight'])

    save_figure(fig, 'train_test_split.png')


def create_results_table():
    """Slide 17: Results table visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_title('Evaluation Results Summary', fontsize=14, fontweight='bold', pad=15)

    # Results data
    results_data = [
        ['Random', '0.0795', '0.0507', '0.4427', '1.000', ''],
        ['Popularity', '0.1386', '0.0969', '0.5995', '0.100', ''],
        ['Genre-Based', '0.0691', '0.0449', '0.4328', '0.428', ''],
        ['Content-Based', '0.0942', '0.0596', '0.4594', '0.962', ''],
        ['k-NN (k=20)', '0.1062', '0.0722', '0.5045', '1.000', ''],
        ['Matrix Factorization', '0.1427', '0.1000', '0.6003', '0.256', 'BEST ACCURACY'],
        ['Hybrid', '0.1261', '0.0867', '0.5698', '0.948', 'BEST BALANCE'],
    ]

    table = ax.table(cellText=results_data,
                     colLabels=['Model', 'MAP@10', 'NDCG@10', 'Hit Rate', 'Coverage', 'Note'],
                     loc='center', cellLoc='left',
                     bbox=[0.02, 0.2, 0.96, 0.7])

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Baseline rows (gray)
    for i in range(1, 4):
        for j in range(6):
            table[(i, j)].set_facecolor(COLORS['light_bg'])

    # Advanced models (white)
    for i in range(4, 6):
        for j in range(6):
            table[(i, j)].set_facecolor('white')

    # Best rows highlighted
    for j in range(6):
        table[(6, j)].set_facecolor('#E8F5E9')  # MF - light green
        table[(7, j)].set_facecolor('#E3F2FD')  # Hybrid - light blue

    ax.text(0.5, 0.1, 'Matrix Factorization: Best MAP@10 (0.1427) | Hybrid: Best coverage (94.8%)',
            transform=ax.transAxes, fontsize=10, ha='center', fontweight='bold')

    save_figure(fig, 'results_table.png')


def create_ttest_results():
    """Slide 19: Statistical significance results."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_title('Statistical Significance Testing', fontsize=14, fontweight='bold', pad=15)

    # T-test results (no emoji checkmarks)
    ttest_data = [
        ['Hybrid vs Random', '> 5.0', '< 0.001', 'Highly Significant'],
        ['Hybrid vs Content-Based', '> 2.5', '< 0.05', 'Significant'],
        ['Hybrid vs k-NN', '> 2.0', '< 0.05', 'Significant'],
        ['MF vs Random', '> 5.0', '< 0.001', 'Highly Significant'],
        ['MF vs Content-Based', '> 3.0', '< 0.01', 'Significant'],
    ]

    table = ax.table(cellText=ttest_data,
                     colLabels=['Comparison', 't-statistic', 'p-value', 'Conclusion'],
                     loc='center', cellLoc='left',
                     bbox=[0.1, 0.25, 0.8, 0.6])

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternating rows
    for i in range(1, 6):
        for j in range(4):
            table[(i, j)].set_facecolor('white' if i % 2 else COLORS['light_bg'])

    ax.text(0.5, 0.15, 'Paired t-tests on NDCG@10 scores - all improvements significant (p < 0.05)',
            transform=ax.transAxes, fontsize=9, ha='center')

    save_figure(fig, 'ttest_results.png')


# Main execution
if __name__ == '__main__':
    print("=" * 60)
    print("Generating Presentation Diagrams for CSE 158 Assignment 2")
    print("=" * 60)

    print("\n[1/14] Creating cold-start diagram...")
    create_cold_start_diagram()

    print("[2/14] Creating system architecture...")
    create_system_architecture()

    print("[3/14] Creating dataset statistics...")
    create_dataset_statistics()

    print("[4/14] Creating feature selection table...")
    create_feature_selection_table()

    print("[5/14] Creating user archetypes...")
    create_user_archetypes()

    print("[6/14] Creating model overview...")
    create_model_overview()

    print("[7/14] Creating content-based diagram...")
    create_content_based_diagram()

    print("[8/14] Creating k-NN diagram...")
    create_knn_diagram()

    print("[9/14] Creating SVD diagram...")
    create_svd_diagram()

    print("[10/14] Creating hybrid architecture...")
    create_hybrid_architecture()

    print("[11/14] Creating metrics definitions...")
    create_metrics_definitions()

    print("[12/14] Creating train/test split diagram...")
    create_train_test_split()

    print("[13/14] Creating results table...")
    create_results_table()

    print("[14/14] Creating t-test results...")
    create_ttest_results()

    print("\n" + "=" * 60)
    print("All diagrams generated successfully!")
    print(f"Output directory: {DIAGRAMS_DIR}/")
    print("=" * 60)