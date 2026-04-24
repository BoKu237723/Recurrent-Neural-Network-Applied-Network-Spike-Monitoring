import random
import datetime

# Configuration
INTERVAL_HOURS = 2
INTERVALS_PER_DAY = 24 // INTERVAL_HOURS   # 12
DAYS = 14  # two weeks

# Base traffic patterns (in MB per 2-hour interval)
# Original 5-min peak: 280 MB → 2-hour peak: 280 * (120/5) = 280 * 24 = 6,720 MB
PEAK_2HOUR_MB = 6720      # ~3.36 GB/hour peak
OFF_PEAK_2HOUR_MB = 2880  # ~1.44 GB/hour
NIGHT_2HOUR_MB = 720      # ~0.36 GB/hour
WEEKEND_BASE_MB = 1440    # lower baseline for weekends (60 * 24)

# REDUCED NOISE: ±10% random variation + rare small spikes
def add_noise(value):
    # Smaller random variation (±10%)
    noise_factor = random.uniform(0.90, 1.10)
    # Occasional small spike (+0 to +1200 MB) - 10% of intervals
    if random.random() < 0.10:
        spike = random.randint(100, 1200)
    # Rare medium spike (+1200 to +2400 MB) - 2% of intervals
    elif random.random() < 0.02:
        spike = random.randint(1200, 2400)
    else:
        spike = 0
    final = int(value * noise_factor) + spike
    return max(120, final)  # never below 120 MB

def get_base_traffic(dt):
    hour = dt.hour
    weekday = dt.weekday()  # 0=Monday, 6=Sunday
    is_weekend = weekday >= 5

    if is_weekend:
        # weekend: lower overall, but still some daytime/night variation
        if 9 <= hour < 22:
            base = WEEKEND_BASE_MB + 1200  # +50 MB * 24
        else:
            base = WEEKEND_BASE_MB
    else:
        # work day
        if 9 <= hour < 17:   # peak work hours
            base = PEAK_2HOUR_MB
        elif 17 <= hour < 22: # evening – still moderate
            base = OFF_PEAK_2HOUR_MB
        else:                 # night (22 – 9)
            base = NIGHT_2HOUR_MB
            # slight bump around midnight (e.g., backups)
            if hour in [0, 1, 2]:
                base = int(base * 1.6)
            # early morning ramp-up (6-8 AM)
            if 6 <= hour <= 8:
                base = int(base * (1.0 + (hour - 5) * 0.3))
    return base

# Generate data
data = []
start_date = datetime.datetime(2025, 1, 6, 0, 0)  # arbitrary Monday start

for day in range(DAYS):
    for interval in range(INTERVALS_PER_DAY):
        current_time = start_date + datetime.timedelta(days=day, hours=interval * INTERVAL_HOURS)
        base_mb = get_base_traffic(current_time)
        final_mb = add_noise(base_mb)
        data.append(final_mb)

# Save to file
output_file = "network_traffic_2hour_2weeks.txt"
with open(output_file, "w") as f:
    for mb in data:
        f.write(f"{mb} ")

# Print some statistics
print(f"Generated {len(data)} data points")  # Should be 14 days * 12 = 168 points
print(f"Max value: {max(data)} MB")
print(f"Min value: {min(data)} MB")
print(f"Average: {sum(data)/len(data):.1f} MB")
print(f"Saved to {output_file}")