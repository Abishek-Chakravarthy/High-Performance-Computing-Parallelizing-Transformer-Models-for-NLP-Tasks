import pandas as pd
import matplotlib.pyplot as plt

# Load performance data from CSV file
csv_file = "parallel_performance_results.csv"
df = pd.read_csv(csv_file)

# Extract columns
threads = df["Threads"]
time = df["Time"]
speedup = df["Speedup"]
parallel_fraction = df["ParallelFraction"]

# 1. Plot Speedup vs. Number of Threads
plt.figure(figsize=(10, 6))
plt.plot(threads, speedup, marker='o', linestyle='-', color='blue')
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs. Number of Threads")
plt.grid(True)
plt.savefig("speedup_vs_threads.png")
plt.show()

# 2. Plot Parallel Fraction vs. Number of Threads
plt.figure(figsize=(10, 6))
plt.plot(threads, parallel_fraction, marker='s', linestyle='-', color='red')
plt.xlabel("Number of Threads")
plt.ylabel("Parallel Fraction")
plt.title("Parallel Fraction vs. Number of Threads")
plt.grid(True)
plt.savefig("parallel_fraction_vs_threads.png")
plt.show()

# 3. Plot Execution Time vs. Number of Threads
plt.figure(figsize=(10, 6))
plt.plot(threads, time, marker='o', linestyle='-', color='green')
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time vs. Number of Threads")
plt.grid(True)
plt.savefig("execution_time_vs_threads.png")
plt.show()

# 4. Plot Efficiency vs. Number of Threads (Efficiency = Speedup / Threads)
efficiency = speedup / threads
plt.figure(figsize=(10, 6))
plt.plot(threads, efficiency, marker='o', linestyle='-', color='magenta')
plt.xlabel("Number of Threads")
plt.ylabel("Efficiency")
plt.title("Efficiency vs. Number of Threads")
plt.grid(True)
plt.savefig("efficiency_vs_threads.png")
plt.show()

print("Plots saved as:")
print("  speedup_vs_threads.png")
print("  parallel_fraction_vs_threads.png")
print("  execution_time_vs_threads.png")
print("  efficiency_vs_threads.png")
