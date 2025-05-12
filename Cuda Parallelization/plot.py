import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("transformer_benchmark_results.csv")

# Extract columns
threads = data["Threads"]
execution_time = data["Parallel Time (s)"]
speedup = data["Speedup"]

# Plot Execution Time
plt.figure(figsize=(8, 5))
plt.plot(threads, execution_time, marker='o', linestyle='-', color='b', label='Execution Time')
plt.xlabel("Threads per Block")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time vs. Threads per Block")
plt.xscale("log", base=2)  # Log scale for better visualization
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.savefig("execution_time_plot.png")
plt.show()

# Plot Speedup
plt.figure(figsize=(8, 5))
plt.plot(threads, speedup, marker='s', linestyle='-', color='r', label='Speedup')
plt.xlabel("Threads per Block")
plt.ylabel("Speedup")
plt.title("Speedup vs. Threads per Block")
plt.xscale("log", base=2)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.savefig("speedup_plot.png")
plt.show()