# ♟️ Tournament-Grade Konane (Hawaiian Checkers) AI Engine

A highly optimized, multithreaded Artificial Intelligence engine written in modern C++ to play the board game **Konane** (Hawaiian Checkers). 

This agent uses a deeply optimized **Minimax algorithm with Alpha-Beta Pruning** and can evaluate millions of board states per second using bitwise arithmetic, Lazy SMP multithreading, and advanced search heuristics.

![C++](https://img.shields.io/badge/C++-17%2B-blue.svg)
![Optimization](https://img.shields.io/badge/Optimization--O3-success.svg)
![Multithreading](https://img.shields.io/badge/Multithreading-Lazy_SMP-orange.svg)

---

## 🚀 Key Features & Optimizations

This engine implements several state-of-the-art chess-engine techniques adapted for Konane:

### ⚡ Move Generation & Board Representation
*   **64-bit Bitboards:** The 8x8 game board is represented purely as 64-bit integers. Move generation and board evaluations are resolved in single CPU clock cycles using bitwise operations (`AND`, `OR`, `XOR`, Bitshifts) and hardware intrinsics (`__builtin_popcountll`, `__builtin_ctzll`).
*   **Zobrist Hashing:** Every board state generates a unique, incrementally updated 64-bit hash in practically zero time.

### 🧠 Search Algorithm
*   **Iterative Deepening & Aspiration Windows:** Searches shallow depths first to guarantee a move is ready before the time limit, then uses previous scores to narrow the Alpha-Beta window for faster deep searches.
*   **Late Move Reduction (LMR):** Aggressively prunes unpromising branches at shallower depths, vastly increasing the overall search depth.
*   **Move Ordering:** Utilizes a custom History Heuristic, Capture Distance scoring, and TT-Move prioritization to ensure the engine checks the strongest moves first, resulting in **60%+ Alpha-Beta cutoff rates**.

### 🧵 Multithreading & Memory
*   **Lazy SMP (Symmetric Multiprocessing):** Spawns a worker thread for every available CPU core. Threads share a global hash table but search randomized move branches to discover traps in parallel.
*   **Lock-Free Spinlocks:** Uses a custom SIMD-optimized spinlock (`_mm_pause`) with lock-striping to allow all CPU threads to write to the memory table simultaneously without OS-level mutex bottlenecks.
*   **CPU Affinity:** On Linux, physical OS threads are bound directly to specific CPU cores (`pthread_setaffinity_np`) to prevent L1/L2 cache misses.
*   **Shared Transposition Table (TT):** A massive 256MB cache stores previously evaluated board states to prevent redundant calculations.

### 🎯 Heuristics
*   **Mobility-Centric Evaluation:** In Konane, the last player to move wins. The engine's core evaluation genetically weights *Mobility* (having more available jumps than the opponent) heavily over static positioning.
*   **Piece Square Tables (PST):** Applies static positional weights to prioritize board corners and penalize vulnerable edge-adjacent squares.

---

## 🛠️ Building and Running

### Prerequisites
A modern C++ compiler (GCC or Clang) supporting C++17. 

### Compilation
Because this engine relies on multithreading and specific hardware instructions, compile with `-O3` (Max Optimization), `-pthread`, and `-march=native`:

```bash
g++ -O3 -pthread -march=native -std=c++17 main.cpp -o konane
