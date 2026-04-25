import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates

DAYS = 7
INTERVAL_HOURS = 2
INTERVALS_PER_DAY = 24 // INTERVAL_HOURS  # 12

# Read the data
with open("predicted_pattern_2hour.txt", "r") as f: # predicted_pattern_2hour.txt # network_traffic_2hour_2weeks.txt
    data = [int(x) for x in f.read().strip().split()]

# Create timeline for 2-hour intervals
start_date = datetime(2025, 1, 6, 0, 0)
timeline = [start_date + timedelta(hours=INTERVAL_HOURS * i) for i in range(len(data))]

# Create dot plot
fig, ax = plt.subplots(figsize=(16, 8))

# Create scatter plot with dots
ax.scatter(timeline, data, s=30, c='blue', alpha=0.7, marker='o')

# Customize the plot
ax.set_title('Network Traffic - Dot Plot (2-hour intervals over 2 weeks)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date and Time', fontsize=12)
ax.set_ylabel('Traffic (MB per 2 hours)', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

# Add horizontal lines for reference (updated for 2-hour values)
ax.axhline(y=6720, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Peak baseline (~6.7 GB/2hr)')
ax.axhline(y=2880, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Off-peak baseline (~2.9 GB/2hr)')
ax.axhline(y=720, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Night baseline (0.72 GB/2hr)')

# Add vertical lines for day boundaries (every 12 intervals = 1 day)
for day in range(1, DAYS):
    ax.axvline(timeline[day * INTERVALS_PER_DAY], color='gray', alpha=0.3, linewidth=0.5)

# Set y-axis limits
ax.set_ylim(0, max(data) + 1000)

# Add legend
ax.legend(loc='upper right', fontsize=10)

# Format x-axis to show day of week
ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))  # %A = full weekday name
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Tick every day
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Print summary
print("\n" + "=" * 70)
print("DOT PLOT VISUALIZATION SUMMARY")
print("=" * 70)
print(f"Total dots (data points): {len(data)}")
print(f"Time span: {timeline[0].strftime('%Y-%m-%d %H:%M')} to {timeline[-1].strftime('%Y-%m-%d %H:%M')}")
print(f"Each dot represents: {INTERVAL_HOURS} hours of network traffic")
print(f"Intervals per day: {INTERVALS_PER_DAY}")
print(f"X-axis shows: Every single day (Jan 6 through Jan 19)")
print(f"Expected daily pattern: Higher dots (~6-7 GB) during work hours, lower at night/weekends")