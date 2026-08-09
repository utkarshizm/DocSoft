import matplotlib.pyplot as plt

# Your REAL latency data from the load test (in seconds)
latencies = [2.16, 1.51, 1.43, 1.53, 2.04, 1.45, 1.44, 2.17, 1.37, 1.56, 1.46, 1.44]
requests = list(range(1, len(latencies) + 1))

# Your calculated metrics
p50 = 1.48
p95 = 2.16

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(requests, latencies, marker='o', linestyle='-', color='#1f77b4', label='Request Latency')

# Add p50 and p95 horizontal lines
plt.axhline(y=p50, color='g', linestyle='--', label=f'p50 ({p50}s)')
plt.axhline(y=p95, color='r', linestyle='--', label=f'p95 ({p95}s)')

# Formatting
plt.title('DocSoft API - Baseline Query Latency (Sequential Load Test)', fontsize=14)
plt.xlabel('Request Number', fontsize=12)
plt.ylabel('Latency (seconds)', fontsize=12)
plt.xticks(requests)
plt.ylim(1.0, 2.5)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

# Save as a high-quality image
plt.tight_layout()
plt.savefig('latency_graph.png', dpi=300)
print("✅ Graph saved successfully as 'latency_graph.png'!")
