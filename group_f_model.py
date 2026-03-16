import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings, os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

# Use augmented CSV if available, otherwise original
if os.path.exists("classroom_temperature_augmented.csv"):
    file_path = "classroom_temperature_augmented.csv"
    print("Using augmented dataset.")
else:
    file_path = "classroom_temperature.csv"
    print("Using original dataset. Run augment_temperature.py first for better results.")

df = pd.read_csv(file_path)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop metadata and date — not useful features
df = df.drop(columns=['source', 'date'], errors='ignore')

# Drop constant columns — same value every row, teach the model nothing
# ac_on=0 always, class_ongoing=0 always, doors_open=1 always
# lights_on and season have near-zero variance in this dataset
df = df.drop(columns=['ac_on','class_ongoing','doors_open','lights_on','season'], errors='ignore')
print("Dropped: source, date, ac_on, class_ongoing, doors_open, lights_on, season")


# ─────────────────────────────────────────────
# STEP 2 — HANDLE MISSING VALUES
# ─────────────────────────────────────────────

# prev_avg_last_session: first row has no previous session, stored as '—'
df['prev_avg_last_session'] = pd.to_numeric(df['prev_avg_last_session'], errors='coerce')
df['prev_avg_last_session'] = df['prev_avg_last_session'].fillna(df['prev_avg_last_session'].mean())


# ─────────────────────────────────────────────
# STEP 3 — DERIVE MISSING COLUMNS + ENCODE
# ─────────────────────────────────────────────

# Derive time_of_day from hour if column is missing
# morning = 5-11, afternoon = 12-16, evening = 17-23
if 'time_of_day' not in df.columns:
    def hour_to_slot(h):
        if 5 <= h <= 11:  return 'morning'
        if 12 <= h <= 16: return 'afternoon'
        return 'evening'
    df['time_of_day'] = df['hour'].apply(hour_to_slot)
    print('  Derived time_of_day from hour column')

# Derive wind if missing — default to light (common in Benin City)
if 'wind' not in df.columns:
    df['wind'] = 'light'
    print('  wind column missing — defaulting to light')

# Derive rain if missing — default to no
if 'rain' not in df.columns:
    df['rain'] = 'no'
    print('  rain column missing — defaulting to no')

# Derive season if missing — default to dry (March readings)
if 'season' not in df.columns:
    df['season'] = 'dry'

# Only encode columns that actually exist
categorical_cols = ['day','time_of_day','out_weather','humidity','sunlight','wind','rain']
categorical_cols = [c for c in categorical_cols if c in df.columns]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f'  Encoded: {col}')

df = df.drop(columns=categorical_cols, errors='ignore')


# ─────────────────────────────────────────────
# STEP 4 — ENGINEER USEFUL FEATURES
# ─────────────────────────────────────────────

# Cyclical hour encoding — so the model knows 11pm and 1am are close together
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# How much hotter/cooler is it outside vs the last session
df['temp_diff'] = df['out_temp_c'] - df['prev_avg_last_session']

# NOTE: Room_L, Room_B, Room_H, Win_L, Win_B NOT added as features
# They are constants — same value every single row
# A constant column gives the model zero information
# Window_Ratio and People_Density also removed — just People / constant = redundant


# ─────────────────────────────────────────────
# STEP 5 — DEFINE FEATURES AND TARGETS
# ─────────────────────────────────────────────

grid_cols = [c for c in df.columns if c.lower().startswith('l') and len(c) == 4]
df['room_avg'] = df[grid_cols].mean(axis=1)   # average of all 16 points
target_cols = grid_cols + ['room_avg']        # 17 targets: 16 points + 1 average

# Build feature list dynamically — only include cols that actually exist
# This handles CSVs that are missing wind, rain, time_of_day etc.
all_possible_features = [
    'hour','hour_sin','hour_cos',
    'out_temp_c','temp_diff',
    'people','windows_open','fans_on',
    'prev_avg_last_session',
    'day_enc','time_of_day_enc','out_weather_enc',
    'humidity_enc','sunlight_enc','wind_enc','rain_enc',
]
feature_cols = [f for f in all_possible_features if f in df.columns]
print(f'  Using {len(feature_cols)} features: {feature_cols}')

X = df[feature_cols].copy()
y = df[target_cols].copy()

print(f"\nFeatures : {len(feature_cols)}")
print(f"Targets  : {len(target_cols)} (16 grid points + room_avg)")
print(f"X shape  : {X.shape}")


# ─────────────────────────────────────────────
# STEP 6 — SCALE FEATURES
# ─────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)


# ─────────────────────────────────────────────
# STEP 7 — SPLIT
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining : {len(X_train)} rows")
print(f"Testing  : {len(X_test)} rows")


# ─────────────────────────────────────────────
# STEP 8 — TRAIN ONE MODEL PER TARGET
# ─────────────────────────────────────────────

results = {}
models  = {}

print("\nTraining models...")
print(f"{'Target':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV R²':>8}")
print("─" * 48)

for target in target_cols:
    rf = RandomForestRegressor(
        n_estimators=200,     # 200 trees in the forest
        max_depth=10,         # prevents overfitting
        min_samples_split=5,  # needs at least 5 samples to split a node
        min_samples_leaf=2,   # each leaf must have at least 2 samples
        random_state=42,
        n_jobs=-1             # use all CPU cores
    )

    rf.fit(X_train, y_train[target])
    y_pred = rf.predict(X_test)

    mae  = mean_absolute_error(y_test[target], y_pred)
    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
    r2   = r2_score(y_test[target], y_pred)
    cv   = cross_val_score(rf, X_train, y_train[target], cv=3, scoring='r2').mean()

    results[target] = {
        'MAE':mae, 'RMSE':rmse, 'R2':r2, 'CV_R2':cv,
        'predictions':y_pred, 'actual':y_test[target]
    }
    models[target] = rf
    print(f"  {target:<10} {mae:>7.3f}°C {rmse:>7.3f}°C {r2:>7.3f} {cv:>7.3f}")

avg_mae = np.mean([results[k]['MAE'] for k in results])
avg_r2  = np.mean([results[k]['R2']  for k in results])
print("─" * 48)
print(f"  {'AVERAGE':<10} {avg_mae:>7.3f}°C {'':>8} {avg_r2:>7.3f}")


# ─────────────────────────────────────────────
# STEP 9 — FEATURE IMPORTANCE CHART
# ─────────────────────────────────────────────

importance_df = pd.DataFrame({
    'feature'   : feature_cols,
    'importance': models['room_avg'].feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 7))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(importance_df)))
plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.xlabel('Importance Score')
plt.title('Feature Importance — Room Average Temperature Prediction', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
for i, (_, row) in enumerate(importance_df.iterrows()):
    plt.text(row['importance'] + 0.002, i, f'{row["importance"]:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("\nSaved: feature_importance.png")

print("\nTop 5 most important features:")
print(importance_df.head(5).to_string(index=False))


# ─────────────────────────────────────────────
# STEP 10 — ACTUAL vs PREDICTED PLOT
# ─────────────────────────────────────────────

actual    = results['room_avg']['actual']
predicted = results['room_avg']['predictions']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(actual, predicted, alpha=0.7, color='steelblue', edgecolors='k', s=60)
axes[0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label='Perfect fit')
axes[0].set_xlabel('Actual Temperature (°C)')
axes[0].set_ylabel('Predicted Temperature (°C)')
axes[0].set_title('Room Average: Actual vs Predicted', fontweight='bold')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

residuals = actual - predicted
axes[1].scatter(predicted, residuals, alpha=0.7, color='seagreen', edgecolors='k', s=60)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Temperature (°C)')
axes[1].set_ylabel('Residuals (°C)')
axes[1].set_title('Residuals Plot\n(closer to 0 = better)', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
print("Saved: model_performance.png")


# ─────────────────────────────────────────────
# STEP 11 — SAVE EVERYTHING
# ─────────────────────────────────────────────

joblib.dump(models,   'temp_models.pkl')      # all 17 trained RF models
joblib.dump(scaler,   'temp_scaler.pkl')      # feature scaler
joblib.dump(encoders, 'temp_encoders.pkl')    # label encoders
joblib.dump(feature_cols, 'temp_features.pkl')  # feature column order

print("\nSaved:")
print("  temp_models.pkl    — 17 RandomForest models (16 points + room_avg)")
print("  temp_scaler.pkl    — StandardScaler for features")
print("  temp_encoders.pkl  — LabelEncoders for text columns")
print("  temp_features.pkl  — feature column order (needed for prediction)")
print("\nTraining complete.")