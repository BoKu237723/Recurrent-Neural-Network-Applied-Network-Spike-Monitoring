import random
import datetime

# Configuration
INTERVAL_MINUTES = 5
INTERVALS_PER_HOUR = 60 // INTERVAL_MINUTES  # 12
INTERVALS_PER_DAY = 24 * INTERVALS_PER_HOUR   # 288
DAYS = 14  # two weeks

# Base traffic patterns (in MB per 5‑min interval)
# Now adjusted to reach near 300 MB during peak times
PEAK_5MIN_MB = 280      # ~3.36 GB/hour (near your 300 MB target)
OFF_PEAK_5MIN_MB = 120  # ~1.44 GB/hour
NIGHT_5MIN_MB = 30      # ~0.36 GB/hour
WEEKEND_BASE_MB = 60    # lower baseline for weekends

# More aggressive noise: ±30% random variation + occasional large spikes
def add_noise(value):
    # Larger random variation (±30%)
    noise_factor = random.uniform(0.70, 1.30)
    # Occasional medium spike (+0 to +150 MB) - 15% of intervals
    if random.random() < 0.15:
        spike = random.randint(20, 150)
    # Rare large spike (+200 to +400 MB) - 3% of intervals
    elif random.random() < 0.03:
        spike = random.randint(200, 400)
    else:
        spike = 0
    final = int(value * noise_factor) + spike
    return max(5, final)  # never below 5 MB

def get_base_traffic(dt):
    hour = dt.hour
    weekday = dt.weekday()  # 0=Monday, 6=Sunday
    is_weekend = weekday >= 5

    if is_weekend:
        # weekend: lower overall, but still some daytime/night variation
        if 9 <= hour < 22:
            base = WEEKEND_BASE_MB + 50
        else:
            base = WEEKEND_BASE_MB
    else:
        # work day
        if 9 <= hour < 17:   # peak work hours
            base = PEAK_5MIN_MB
        elif 17 <= hour < 22: # evening – still moderate
            base = OFF_PEAK_5MIN_MB
        else:                 # night (22 – 9)
            base = NIGHT_5MIN_MB
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
        current_time = start_date + datetime.timedelta(days=day, minutes=interval * INTERVAL_MINUTES)
        base_mb = get_base_traffic(current_time)
        final_mb = add_noise(base_mb)
        data.append(final_mb)

# Save to file
output_file = "network_traffic_5min_2weeks.txt"
with open(output_file, "w") as f:
    for mb in data:
        f.write(f"{mb} ")

# Print some statistics
print(f"Generated {len(data)} data points")
print(f"Max value: {max(data)} MB")
print(f"Min value: {min(data)} MB")
print(f"Average: {sum(data)/len(data):.1f} MB")
print(f"Saved to {output_file}")