import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

INPUT_CSV  = "group_F_readings.csv"    # your original dataset
OUTPUT_CSV = "classroom_temperature_augmented.csv"  # output with augmented rows added

AUGMENTS_PER_ROW = 5   # each row produces 5 new variations → 33 × 5 = 165 new rows
                        # total after augmentation: 33 + 165 = 198 rows
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
print(f"Original dataset: {len(df)} rows")

# ── Normalise column names ────────────────────────────────────────────────
# Some CSVs use CamelCase (Out_Temp_C, Windows_Open) others use lowercase
# We standardise everything to lowercase so the script works with both
df.columns = df.columns.str.strip().str.lower()

# Map known alternate column names to our standard names
col_rename = {
    'out_temp_c'        : 'out_temp_c',     # already correct
    'temperature'       : 'out_temp_c',
    'windows_open'      : 'windows_open',
    'no_of_windows'     : 'windows_open',
    'fans_on'           : 'fans_on',
    'no_of_fans'        : 'fans_on',
    'prev_avg'          : 'prev_avg_last_session',
    'prev_avg_last_session': 'prev_avg_last_session',
    'out_weather'       : 'out_weather',
    'weather'           : 'out_weather',
}
df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})
print(f"Columns: {list(df.columns)}")

# Add source column if it does not exist
# (Group_F_readings.csv won't have it — treat all rows as REAL)
if 'source' not in df.columns:
    df['source'] = 'R'
    print("  No 'source' column found — treating all rows as REAL")

# Separate REAL and ESTIMATED rows
df_real = df[df['source'] == 'R'].copy()
df_est  = df[df['source'] == 'E'].copy()
print(f"  REAL rows     : {len(df_real)}")
print(f"  ESTIMATED rows: {len(df_est)}")

# Grid point columns — these are the temperature readings
grid_cols = [c for c in df.columns if c.lower().startswith('l') and len(c) == 4]
# → L1B1, L1B2 ... L4B4

# ─────────────────────────────────────────────
# STEP 2 — AUGMENTATION FUNCTIONS
# Each function takes a row and returns a slightly modified copy
# ─────────────────────────────────────────────

def nudge_temperatures(row, std=0.15):
    """
    Add tiny random noise to all 16 grid temperatures.
    std=0.15 means variations stay within ±0.3°C — realistic thermometer variation.
    """
    new = row.copy()
    for col in grid_cols:
        new[col] = round(float(row[col]) + np.random.normal(0, std), 1)
    return new


def nudge_people(row, max_change=3):
    """
    Vary people count by up to ±3 people.
    A class that had 15 people might have had 13 or 17 on a similar day.
    """
    new = row.copy()
    change = np.random.randint(-max_change, max_change + 1)
    new['people'] = max(0, int(row['people']) + change)
    # People affect temperature slightly — adjust grid temps proportionally
    temp_effect = change * 0.008   # each person adds ~0.008°C
    for col in grid_cols:
        new[col] = round(float(row[col]) + temp_effect, 1)
    return new


def nudge_windows(row):
    """
    Vary windows open by ±1.
    Window-side columns (B1) get a corresponding temp nudge.
    """
    new = row.copy()
    current = float(row['windows_open'])
    change = np.random.choice([-1, 0, 1])
    new_windows = max(0, min(6, current + change))
    new['windows_open'] = new_windows

    # Opening a window cools the window-side column (B1) slightly
    win_effect = -change * 0.1
    for col in grid_cols:
        if col.upper().endswith('B1'):   # window-side column
            new[col] = round(float(row[col]) + win_effect, 1)
    return new


def nudge_outdoor_temp(row, std=0.5):
    """
    Vary outdoor temperature by ±0.5°C.
    Outdoor temp directly affects all indoor temps slightly.
    """
    new = row.copy()
    delta = np.random.normal(0, std)
    new['out_temp_c'] = round(float(row['out_temp_c']) + delta, 1)
    # Small proportional effect on indoor temps
    indoor_effect = delta * 0.05   # 5% of outdoor change reaches indoors
    for col in grid_cols:
        new[col] = round(float(row[col]) + indoor_effect, 1)
    return new


def nudge_fans(row):
    """
    Randomly toggle fans (only if current value allows toggling).
    Fans reduce temperature spread — middle cols become more uniform.
    """
    new = row.copy()
    current_fans = int(row['fans_on'])

    # Only flip if it makes sense (don't go below 0 or above 2)
    if current_fans == 0:
        new_fans = np.random.choice([0, 1])
    elif current_fans == 2:
        new_fans = np.random.choice([1, 2])
    else:
        new_fans = np.random.choice([0, 1, 2])

    fan_change = new_fans - current_fans
    new['fans_on'] = new_fans

    # Fans reduce spread — wooden wall (B4) cools slightly when fans increase
    for col in grid_cols:
        if col.upper().endswith('B4'):   # wooden wall side — most affected by fans
            new[col] = round(float(row[col]) - fan_change * 0.08, 1)
        elif col.upper().endswith('B2') or col.upper().endswith('B3'):   # middle — slight effect
            new[col] = round(float(row[col]) - fan_change * 0.04, 1)
    return new


# All augmentation functions in one list
AUGMENTATION_POOL = [
    nudge_temperatures,
    nudge_people,
    nudge_windows,
    nudge_outdoor_temp,
    nudge_fans,
]


# ─────────────────────────────────────────────
# STEP 3 — GENERATE AUGMENTED ROWS
# ─────────────────────────────────────────────

augmented_rows = []

for _, row in df.iterrows():

    for i in range(AUGMENTS_PER_ROW):

        # Apply a randomly chosen augmentation function
        aug_fn = np.random.choice(AUGMENTATION_POOL)
        new_row = aug_fn(row)

        # Mark as augmented so we can tell them apart
        new_row['source'] = 'AUG'

        # Recalculate prev_avg for the augmented row
        # (use the average of the 16 grid temps in the augmented row)
        grid_vals = [float(new_row[c]) for c in grid_cols]
        if len(grid_vals) > 0:
            new_row['prev_avg_last_session'] = round(sum(grid_vals) / len(grid_vals), 2)

        augmented_rows.append(new_row)

df_aug = pd.DataFrame(augmented_rows)
print(f"\nAugmented rows generated: {len(df_aug)}")


# ─────────────────────────────────────────────
# STEP 4 — COMBINE AND SAVE
# ─────────────────────────────────────────────

df_final = pd.concat([df, df_aug], ignore_index=True)

# Shuffle so REAL, ESTIMATED and AUG rows are mixed
df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

df_final.to_csv(OUTPUT_CSV, index=False)

print(f"\nFinal dataset saved to '{OUTPUT_CSV}'")
print(f"  Original rows : {len(df)}")
print(f"  Augmented rows: {len(df_aug)}")
print(f"  TOTAL         : {len(df_final)}")
print(f"\nSource breakdown:")
print(df_final['source'].value_counts().to_string())