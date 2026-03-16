import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────
# LOAD SAVED MODEL FILES
# (Run train_temperature_model.py first to generate these)
# ─────────────────────────────────────────────

try:
    models       = joblib.load('temp_models.pkl')
    scaler       = joblib.load('temp_scaler.pkl')
    encoders     = joblib.load('temp_encoders.pkl')
    feature_cols = joblib.load('temp_features.pkl')
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run train_temperature_model.py first to generate the model files.")
    exit()


# ─────────────────────────────────────────────
# CLASSROOM HEATMAP FUNCTION
# Draws a colour-coded 4x4 grid of the classroom
# ─────────────────────────────────────────────

def draw_classroom_heatmap(grid_dict, title="Predicted Classroom Temperature (°C)", save_as=None):
    """
    grid_dict: { 'L1B1': 26.5, 'L1B2': 26.3, ... 'L4B4': 27.1 }
    Draws the classroom from front (board) to back (door).
    """
    # Arrange into 4x4 grid (rows=L1-L4, cols=B1-B4)
    # Normalise keys to lowercase so it works regardless of how model saved them
    gd = {k.lower(): v for k, v in grid_dict.items()}
    grid = np.array([
        [gd['l1b1'], gd['l1b2'], gd['l1b3'], gd['l1b4']],
        [gd['l2b1'], gd['l2b2'], gd['l2b3'], gd['l2b4']],
        [gd['l3b1'], gd['l3b2'], gd['l3b3'], gd['l3b4']],
        [gd['l4b1'], gd['l4b2'], gd['l4b3'], gd['l4b4']],
    ])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Colour map: blue=cool, yellow=warm, red=hot
    vmin = grid.min() - 0.5
    vmax = grid.max() + 0.5
    im = ax.imshow(grid, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='auto')

    # Label each cell with its point name and temperature
    row_labels = ['L1 (Front/Board)', 'L2 (Mid + Fan)', 'L3 (Mid + Fan)', 'L4 (Back/Door)']
    col_labels = ['B1 (Window)', 'B2', 'B3', 'B4 (Wooden Wall)']
    point_names = [
        ['L1B1','L1B2','L1B3','L1B4'],
        ['L2B1','L2B2','L2B3','L2B4'],
        ['L3B1','L3B2','L3B3','L3B4'],
        ['L4B1','L4B2','L4B3','L4B4'],
    ]  # display labels only — always uppercase for readability

    for r in range(4):
        for c in range(4):
            temp  = grid[r, c]
            point = point_names[r][c]
            # White text on dark cells, black on light
            norm_val = (temp - vmin) / (vmax - vmin)
            text_color = 'white' if norm_val > 0.6 else 'black'
            ax.text(c, r, f"{point}\n{temp:.1f}°C",
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color=text_color)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)

    # Classroom orientation labels
    ax.set_xlabel('← Window Side (Left Wall)          Wooden Wall (Right) →',
                  fontsize=10, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Add board and door labels
    ax.annotate('BOARD →', xy=(0, -0.08), xycoords='axes fraction',
                fontsize=9, color='navy', fontweight='bold')
    ax.annotate('← DOOR (D4 corner)', xy=(0.7, 1.02), xycoords='axes fraction',
                fontsize=9, color='darkred', fontweight='bold')

    plt.colorbar(im, ax=ax, label='Temperature (°C)', shrink=0.8)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved: {save_as}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_temperature(
    hour,           # int: actual hour e.g. 8, 13, 18
    out_temp_c,     # float: outdoor temperature in °C
    people,         # int: number of people in room
    windows_open,   # float: how many windows open (0-6, fractions allowed e.g. 3.33)
    fans_on,        # int: number of fans on (0, 1, or 2)
    prev_avg,       # float: average room temp from last session (or estimated)
    day,            # str: 'Monday', 'Tuesday', etc.
    time_of_day,    # str: 'morning', 'afternoon', 'evening'
    out_weather,    # str: 'cold', 'cool', 'warm', 'hot'
    humidity,       # str: 'low', 'medium', 'high'
    sunlight,       # str: 'no', 'partial', 'yes'
    wind,           # str: 'none', 'light', 'strong'
    rain,           # str: 'no', 'yes'
    show_heatmap=True,
    save_heatmap=None
):
    """
    Returns a dictionary of predicted temperatures for all 16 grid points + room average.

    Example usage:
        results = predict_temperature(
            hour=14, out_temp_c=35.0, people=30, windows_open=4, fans_on=2,
            prev_avg=26.5, day='Thursday', time_of_day='afternoon',
            out_weather='hot', humidity='low', sunlight='yes',
            wind='light', rain='no'
        )
    """

    # ── Encode categorical inputs ──────────────────────────────────────────
    try:
        day_enc       = encoders['day'].transform([day])[0]
        tod_enc       = encoders['time_of_day'].transform([time_of_day])[0]
        weather_enc   = encoders['out_weather'].transform([out_weather])[0]
        humidity_enc  = encoders['humidity'].transform([humidity])[0]
        sunlight_enc  = encoders['sunlight'].transform([sunlight])[0]
        wind_enc      = encoders['wind'].transform([wind])[0]
        rain_enc      = encoders['rain'].transform([rain])[0]
    except ValueError as e:
        print(f"\nEncoding error: {e}")
        print("Make sure text values exactly match what was in your training data.")
        print("  day        : Monday Tuesday Wednesday Thursday Friday Saturday Sunday")
        print("  time_of_day: morning afternoon evening")
        print("  out_weather: cold cool warm hot")
        print("  humidity   : low medium high")
        print("  sunlight   : no partial yes")
        print("  wind       : none light strong")
        print("  rain       : no yes")
        return None

    # ── Engineer features (same as training script) ────────────────────────
    hour_sin   = np.sin(2 * np.pi * hour / 24)
    hour_cos   = np.cos(2 * np.pi * hour / 24)
    temp_diff  = out_temp_c - prev_avg

    # ── Build input row in exact same column order as training ─────────────
    input_row = pd.DataFrame([{
        'hour'                  : hour,
        'hour_sin'              : hour_sin,
        'hour_cos'              : hour_cos,
        'out_temp_c'            : out_temp_c,
        'temp_diff'             : temp_diff,
        'people'                : people,
        'windows_open'          : windows_open,
        'fans_on'               : fans_on,
        'prev_avg_last_session' : prev_avg,
        'day_enc'               : day_enc,
        'time_of_day_enc'       : tod_enc,
        'out_weather_enc'       : weather_enc,
        'humidity_enc'          : humidity_enc,
        'sunlight_enc'          : sunlight_enc,
        'wind_enc'              : wind_enc,
        'rain_enc'              : rain_enc,
    }])[feature_cols]   # ensure column order matches training

    input_scaled = pd.DataFrame(scaler.transform(input_row), columns=feature_cols)

    # ── Run prediction for each target ────────────────────────────────────
    predictions = {}
    for target, model in models.items():
        predictions[target] = round(model.predict(input_scaled)[0], 2)

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "="*52)
    print(f"  TEMPERATURE PREDICTION")
    print(f"  {day} {time_of_day.upper()} | Hour {hour}:00")
    print(f"  Outside: {out_temp_c}°C ({out_weather}) | People: {people}")
    print(f"  Windows: {windows_open} open | Fans: {fans_on} on | Rain: {rain}")
    print("="*52)

    grid_cols_only = [k for k in predictions if k != 'room_avg']
    print(f"\n  {'Point':<8} {'Temp':>8}  {'Note'}")
    print("  " + "─"*42)
    # Use actual keys from predictions dict — handles both l1b1 and L1B1
    ordered_points = sorted(grid_cols_only, key=lambda x: x.lower())
    for point in ordered_points:
        temp = predictions[point]
        note = ""
        if point.lower() in ['l1b1','l1b4','l4b1','l4b4']:
            note = "← corner"
        if point.lower().endswith('b4'):
            note += " (wooden wall)"
        if point.lower().endswith('b1'):
            note += " (window side)"
        print(f"  {point.upper():<8} {temp:>6.2f}°C  {note}")

    print("  " + "─"*42)
    print(f"  {'Room Avg':<8} {predictions['room_avg']:>6.2f}°C")

    # Comfort label
    avg = predictions['room_avg']
    if avg < 24:   comfort = "COLD"
    elif avg < 27: comfort = "COMFORTABLE"
    elif avg < 30: comfort = "WARM"
    else:          comfort = "HOT"
    print(f"\n  Comfort level : {comfort}")

    hottest  = max(grid_cols_only, key=lambda k: predictions[k])
    coolest  = min(grid_cols_only, key=lambda k: predictions[k])
    print(f"  Hottest point : {hottest.upper()} = {predictions[hottest]:.2f}°C")
    print(f"  Coolest point : {coolest.upper()} = {predictions[coolest]:.2f}°C")
    print("="*52)

    # ── Draw classroom heatmap ─────────────────────────────────────────────
    if show_heatmap:
        grid_only = {k: v for k, v in predictions.items() if k != 'room_avg'}
        title = f"Predicted Temperatures — {day} {time_of_day.title()} ({hour}:00)\nOutside: {out_temp_c}°C | People: {people} | Windows: {windows_open}"
        draw_classroom_heatmap(grid_only, title=title, save_as=save_heatmap)

    return predictions


# ─────────────────────────────────────────────
# EXAMPLE PREDICTIONS
# Change these values to match the session you want to predict
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n── EXAMPLE 1: Thursday afternoon, hot day ──")
    predict_temperature(
        hour=14, out_temp_c=35.0, people=30,
        windows_open=4, fans_on=2, prev_avg=26.5,
        day='Thursday', time_of_day='afternoon',
        out_weather='hot', humidity='low',
        sunlight='yes', wind='light', rain='no',
        save_heatmap='heatmap_afternoon.png'
    )

    print("\n── EXAMPLE 2: Wednesday morning, rainy and cold ──")
    predict_temperature(
        hour=8, out_temp_c=22.0, people=10,
        windows_open=2.67, fans_on=2, prev_avg=25.2,
        day='Wednesday', time_of_day='morning',
        out_weather='cold', humidity='high',
        sunlight='no', wind='none', rain='yes',
        save_heatmap='heatmap_rainy_morning.png'
    )

    print("\n── EXAMPLE 3: What if 2 more windows are opened? ──")
    print("Before (2 windows open):")
    before = predict_temperature(
        hour=13, out_temp_c=33.0, people=20,
        windows_open=2, fans_on=2, prev_avg=26.0,
        day='Monday', time_of_day='afternoon',
        out_weather='hot', humidity='medium',
        sunlight='yes', wind='light', rain='no',
        show_heatmap=False
    )

    print("\nAfter (4 windows open):")
    after = predict_temperature(
        hour=13, out_temp_c=33.0, people=20,
        windows_open=4, fans_on=2, prev_avg=26.0,
        day='Monday', time_of_day='afternoon',
        out_weather='hot', humidity='medium',
        sunlight='yes', wind='light', rain='no',
        show_heatmap=False
    )

    if before and after:
        diff = after['room_avg'] - before['room_avg']
        print(f"\n  Opening 2 more windows changed room avg by: {diff:+.2f}°C")