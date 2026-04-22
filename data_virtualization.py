import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

DAYS = 14

# Read the data
with open("predicted_pattern.txt", "r") as f: # predicted_pattern.txt # network_traffic_5min_2weeks.txt
    data = [int(x) for x in f.read().strip().split()]

# Create timeline
start_date = datetime(2025, 1, 6, 0, 0)
timeline = [start_date + timedelta(minutes=5*i) for i in range(len(data))]

# Create dot plot
fig, ax = plt.subplots(figsize=(16, 8))

# Create scatter plot with small dots
ax.scatter(timeline, data, s=8, c='blue', alpha=0.6, marker='o')

# Customize the plot
ax.set_title('Network Traffic - Dot Plot (5-minute intervals over 2 weeks)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date and Time', fontsize=12)
ax.set_ylabel('Traffic (MB per 5 minutes)', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

# Add horizontal lines for reference
ax.axhline(y=400, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Peak baseline (400 MB)')
ax.axhline(y=150, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Off-peak baseline (150 MB)')
ax.axhline(y=40, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Night baseline (40 MB)')

# Add vertical lines for day boundaries (every 288 intervals = 1 day)
for day in range(1, DAYS):
    ax.axvline(timeline[day*144], color='gray', alpha=0.2, linewidth=0.5) ## edit this based on training file

# Set y-axis limits
ax.set_ylim(0, max(data) + 50)

# Add legend
ax.legend(loc='upper right', fontsize=10)

# Format x-axis to show dates nicely
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Print summary
print("\n" + "=" * 70)
print("DOT PLOT VISUALIZATION SUMMARY")
print("=" * 70)
print(f"Total dots (data points): {len(data)}")
print(f"Time span: {timeline[0].strftime('%Y-%m-%d %H:%M')} to {timeline[-1].strftime('%Y-%m-%d %H:%M')}")
print(f"Each dot represents: 5 minutes of network traffic")
print(f"Dot density shows traffic patterns: dense clusters at 400 MB (peak hours), sparse at lower values")