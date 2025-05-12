# 🚀 High Performance Computing: Parallelizing Transformer Models for NLP Tasks

This project explores high-performance computing techniques to optimize a Transformer encoder model (inspired by BERT) using OpenMP, CUDA, and MPI. The goal is to accelerate compute-intensive components such as self-attention, matrix multiplication, GELU, and layer normalization. Profiling tools were used to guide optimization efforts.

---

## 📌 Overview

The Transformer encoder architecture is highly parallelizable due to its reliance on matrix operations and element-wise functions. This project evaluates three parallelization strategies:

- **OpenMP (multi-core CPU parallelism)**
- **CUDA (GPU parallelism)**
- **MPI (Distributed multi-process parallelism)**

Extensive benchmarking and profiling were performed to identify bottlenecks and measure the effectiveness of each method.

---

## 📁 Project Structure

```
├── src/
│   ├── baseline/           # Serial implementation
│   ├── openmp/             # OpenMP version
│   ├── mpi/                # MPI version
│   └── cuda/               # CUDA version
├── reports/                # Contains all detailed PDF reports
├── data/                   # Input/output samples
├── Makefile
└── README.md
```

---

## ⚙️ Requirements

- GCC with OpenMP support
- OpenMPI or MPICH (for MPI version)
- NVIDIA CUDA Toolkit
- Profiling tools: gprof, gcov, LIKWID

---

## 🧪 Profiling Summary

**Tools Used**:
- `gprof`: Identified that `matmul` takes ~93% of execution time.
- `gcov`: Line-level profiling showed hotspot loops in `matmul` and `self_attention`.
- `LIKWID`: Hardware metrics revealed suboptimal floating-point throughput and high memory usage.

### Key Bottlenecks:
- `matmul`: Main performance bottleneck (92.66% of total execution time)
- `self_attention`: 4–6% runtime; indirectly impacted by `matmul`

---

## 🧵 Parallelization Strategies

### 🔹 OpenMP

- Parallelized `matmul`, `self_attention`, and normalization
- Max Speedup: **2.67x at 8 threads**
- Parallelization Fraction: **0.63–0.78**
- Used `#pragma omp parallel for` and SIMD vectorization

### 🔸 CUDA

- Offloaded core Transformer components to GPU
- Optimized tiled matrix multiplication using shared memory
- Execution Time: **Reduced from 70s (CPU) to 1.69s (GPU)**
- Speedup: **~42x at 256 threads**
- Techniques: memory coalescing, thread block tuning, loop unrolling

### 🔷 MPI

- Parallelized matrix ops by distributing row blocks across processes
- Communication: `MPI_Bcast`, `MPI_Gatherv`, `MPI_Allreduce`
- Speedup: **17.68x using 8 processes**
- Focused on distributed memory scaling

---

## 📊 Results

| Version      | Speedup        | Notes                            |
|--------------|----------------|----------------------------------|
| Baseline     | 1.0x           | Serial C++ code                  |
| OpenMP       | 2.67x          | Optimal at 8 threads             |
| CUDA         | 42.0x          | At 256 threads (1.69s runtime)   |
| MPI          | 17.68x         | Using 8 processes (25s runtime)  |

---

## ▶️ How to Run

### Baseline
```bash
cd src/baseline
./bert_baseline input.txt
```

### OpenMP
```bash
cd src/openmp
export OMP_NUM_THREADS=8
./bert_openmp input.txt
```

### CUDA
```bash
cd src/cuda
./bert_cuda input.txt
```

### MPI
```bash
cd src/mpi
mpirun -np 8 ./bert_mpi input.txt
```

---

## 👤 Author

**Abishek Chakravarthy**  
Roll Number: CS22B2054  
IIITDM Kancheepuram, February 2025

---

## 📄 License

This project is licensed under the MIT License.
