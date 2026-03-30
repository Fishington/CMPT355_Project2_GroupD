#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <thread>
#include <random>
#include <future>
#include <cstdint>
#include <cmath>
#include <atomic>
#include <memory>
#include <fstream>
#include <cctype>
#include <cstdlib>

// OS-Level Headers for Thread Pinning
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

// Hardware headers for Spinlock Pausing
#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#endif

// --- SETTINGS ---
const double THINKING_TIME = 9.8; 

class TimeoutException : public std::exception {};

struct Board {
    uint64_t b = 0; 
    uint64_t w = 0; 
};

struct Move {
    int8_t start_r = 0, start_c = 0;
    int8_t end_r = 0, end_c = 0;
    bool is_jump = false;

    bool operator==(const Move& o) const {
        return start_r == o.start_r && start_c == o.start_c &&
               end_r == o.end_r && end_c == o.end_c && is_jump == o.is_jump;
    }

    std::string to_string() const {
        auto coords_to_notation =[](int r, int c) {
            char col_char = (char)(c + 'A');
            char row_char = '8' - r;
            return std::string{col_char, row_char};
        };
        if (!is_jump) return coords_to_notation(start_r, start_c);
        return coords_to_notation(start_r, start_c) + "-" + coords_to_notation(end_r, end_c);
    }
};

struct MoveList {
    Move moves[128];
    int count = 0;
    inline void push_back(const Move& m) { moves[count++] = m; }
    inline Move* begin() { return moves; }
    inline Move* end() { return moves + count; }
    inline const Move* begin() const { return moves; }
    inline const Move* end() const { return moves + count; }
    inline bool empty() const { return count == 0; }
    inline size_t size() const { return count; }
    inline void clear() { count = 0; }
    inline Move& operator[](size_t i) { return moves[i]; }
    inline const Move& operator[](size_t i) const { return moves[i]; }
};

struct Stats {
    long long nodes_evaluated = 0;
    long long cutoffs = 0;
    long long tt_hits = 0; 
};

// --- ZOBRIST HASHING ---
uint64_t zobrist_b[64];
uint64_t zobrist_w[64];
uint64_t zobrist_turn;

void init_zobrist() {
    std::mt19937_64 rng(12345); 
    for (int i = 0; i < 64; ++i) {
        zobrist_b[i] = rng();
        zobrist_w[i] = rng();
    }
    zobrist_turn = rng();
}

uint64_t compute_zobrist(const Board& board, char player) {
    uint64_t hash = 0;
    uint64_t b = board.b;
    while (b) {
        int idx = __builtin_ctzll(b);
        hash ^= zobrist_b[idx];
        b &= b - 1;
    }
    uint64_t w = board.w;
    while (w) {
        int idx = __builtin_ctzll(w);
        hash ^= zobrist_w[idx];
        w &= w - 1;
    }
    if (player == 'B') hash ^= zobrist_turn;
    return hash;
}

// --- GLOBAL SHARED TRANSPOSITION TABLE ---
enum class TTFlag : uint8_t { EXACT = 0, LOWERBOUND = 1, UPPERBOUND = 2 };

struct TTEntry {
    uint64_t key;
    int score;
    int depth;
    TTFlag flag;
    Move best_move; 
};

const int TT_SIZE = 268435456; // Maps perfectly into ~6.4 GB RAM limit
const int LOCK_SIZE = 65536; 

struct alignas(64) SpinLock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT;
    inline void lock() { 
        while (locked.test_and_set(std::memory_order_acquire)) {
            // HARDWARE HACK: Stop memory bus flooding when a thread is waiting
            #if defined(__x86_64__) || defined(_M_X64)
            _mm_pause(); 
            #endif
        } 
    }
    inline void unlock() { locked.clear(std::memory_order_release); }
};

struct TranspositionTable {
    TTEntry* table;
    std::unique_ptr<SpinLock[]> locks; 

    TranspositionTable() {
        table = (TTEntry*)std::calloc(TT_SIZE, sizeof(TTEntry));
        locks.reset(new SpinLock[LOCK_SIZE]);
    }
    
    ~TranspositionTable() { std::free(table); }
    
    void store(uint64_t key, int depth, int score, TTFlag flag, Move best_move) {
        int index = key & (TT_SIZE - 1); 
        int lock_idx = key & (LOCK_SIZE - 1); 
        
        locks[lock_idx].lock();
        if (table[index].depth <= depth) { 
            table[index] = {key, score, depth, flag, best_move};
        }
        locks[lock_idx].unlock();
    }
    
    bool probe(uint64_t key, int depth, int alpha, int beta, int& out_score, Move& out_move) {
        int index = key & (TT_SIZE - 1);
        int lock_idx = key & (LOCK_SIZE - 1);
        
        locks[lock_idx].lock();
        TTEntry entry = table[index];
        locks[lock_idx].unlock();
        
        if (entry.key == key) {
            out_move = entry.best_move; 
            if (entry.depth >= depth) {
                if (entry.flag == TTFlag::EXACT) { out_score = entry.score; return true; }
                if (entry.flag == TTFlag::LOWERBOUND && entry.score >= beta) { out_score = entry.score; return true; }
                if (entry.flag == TTFlag::UPPERBOUND && entry.score <= alpha) { out_score = entry.score; return true; }
            }
        }
        return false;
    }
};

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

double get_elapsed(TimePoint start) {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

// --- GAME LOGIC ---
inline Board clone_board(const Board& board) { return board; }

void apply_move(Board& board, const Move& move, char current_player, uint64_t& hash) {
    uint64_t& my_pieces = (current_player == 'B') ? board.b : board.w;
    uint64_t& opp_pieces = (current_player == 'B') ? board.w : board.b;
    
    int start_idx = move.start_r * 8 + move.start_c;
    const uint64_t* my_zobrist = (current_player == 'B') ? zobrist_b : zobrist_w;
    const uint64_t* opp_zobrist = (current_player == 'B') ? zobrist_w : zobrist_b;

    if (!move.is_jump) { 
        my_pieces &= ~(1ULL << start_idx); 
        hash ^= my_zobrist[start_idx];
        hash ^= zobrist_turn;
        return;
    }

    int dr = (move.end_r == move.start_r) ? 0 : ((move.end_r > move.start_r) ? 1 : -1);
    int dc = (move.end_c == move.start_c) ? 0 : ((move.end_c > move.start_c) ? 1 : -1);
    
    int end_idx = move.end_r * 8 + move.end_c;
    int delta_idx = dr * 8 + dc; 
    
    int curr_idx = start_idx;
    my_pieces &= ~(1ULL << curr_idx); 
    hash ^= my_zobrist[curr_idx];
    
    while (curr_idx != end_idx) {
        curr_idx += delta_idx;
        if (curr_idx != end_idx) {
            opp_pieces &= ~(1ULL << curr_idx); 
            hash ^= opp_zobrist[curr_idx];
            curr_idx += delta_idx;
        }
    }
    my_pieces |= (1ULL << end_idx); 
    hash ^= my_zobrist[end_idx];
    hash ^= zobrist_turn; 
}

MoveList get_legal_moves(const Board& board, char player) {
    MoveList moves;
    int empty_count = 64 - __builtin_popcountll(board.b | board.w);
    
    if (empty_count == 0) {
        if (player == 'B') { moves.push_back({3, 3, 3, 3, false}); moves.push_back({4, 4, 4, 4, false}); }
        return moves;
    } else if (empty_count == 1) {
        if (player == 'W') {
            if (board.w & (1ULL << (3 * 8 + 4))) moves.push_back({3, 4, 3, 4, false}); 
            if (board.w & (1ULL << (4 * 8 + 3))) moves.push_back({4, 3, 4, 3, false}); 
        }
        return moves;
    }

    uint64_t my_pieces = (player == 'B') ? board.b : board.w;
    uint64_t opp_pieces = (player == 'B') ? board.w : board.b;
    uint64_t empty = ~(board.b | board.w);
    
    const int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    uint64_t pieces = my_pieces;
    
    while (pieces) {
        int idx = __builtin_ctzll(pieces); 
        pieces &= pieces - 1; 
        int r = idx / 8, c = idx % 8;
        
        for (const auto& dir : directions) {
            int dr = dir[0], dc = dir[1];
            int curr_r = r, curr_c = c;
            
            while (true) {
                int mid_r = curr_r + dr, mid_c = curr_c + dc;
                int dest_r = curr_r + 2 * dr, dest_c = curr_c + 2 * dc;
                
                if (dest_r >= 0 && dest_r < 8 && dest_c >= 0 && dest_c < 8) {
                    int mid_idx = mid_r * 8 + mid_c;
                    int dest_idx = dest_r * 8 + dest_c;
                    
                    if ((opp_pieces & (1ULL << mid_idx)) && (empty & (1ULL << dest_idx))) {
                        moves.push_back({(int8_t)r, (int8_t)c, (int8_t)dest_r, (int8_t)dest_c, true});
                        curr_r = dest_r; curr_c = dest_c;
                    } else break;
                } else break;
            }
        }
    }
    return moves;
}

int count_legal_moves(const Board& board, char player) {
    int empty_count = 64 - __builtin_popcountll(board.b | board.w);
    if (empty_count == 0) return (player == 'B') ? 2 : 0;
    if (empty_count == 1) {
        int count = 0;
        if (player == 'W') {
            if (board.w & (1ULL << (3 * 8 + 4))) count++;
            if (board.w & (1ULL << (4 * 8 + 3))) count++;
        }
        return count;
    }

    uint64_t mine = (player == 'B') ? board.b : board.w;
    uint64_t opp = (player == 'B') ? board.w : board.b;
    uint64_t empty = ~(board.b | board.w);
    
    int count = 0;
    const uint64_t notA = 0xFEFEFEFEFEFEFEFEULL; 
    const uint64_t notH = 0x7F7F7F7F7F7F7F7FULL; 

    uint64_t movers = mine;
    while (movers) {
        uint64_t step1 = (movers << 1) & opp & notA;      
        uint64_t step2 = (step1 << 1) & empty & notA;     
        if (!step2) break;
        count += __builtin_popcountll(step2);             
        movers = step2;                                   
    }
    
    movers = mine;
    while (movers) {
        uint64_t step1 = (movers >> 1) & opp & notH;
        uint64_t step2 = (step1 >> 1) & empty & notH;
        if (!step2) break;
        count += __builtin_popcountll(step2);
        movers = step2;
    }

    movers = mine;
    while (movers) {
        uint64_t step1 = (movers << 8) & opp;
        uint64_t step2 = (step1 << 8) & empty;
        if (!step2) break;
        count += __builtin_popcountll(step2);
        movers = step2;
    }

    movers = mine;
    while (movers) {
        uint64_t step1 = (movers >> 8) & opp;
        uint64_t step2 = (step1 >> 8) & empty;
        if (!step2) break;
        count += __builtin_popcountll(step2);
        movers = step2;
    }

    return count;
}

int evaluate_board(const Board& board, char player) {
    char opp = (player == 'B') ? 'W' : 'B';
    int my_moves = count_legal_moves(board, player);
    int opp_moves = count_legal_moves(board, opp);
    
    if (opp_moves == 0) return 10000; 
    if (my_moves == 0) return -10000;
    return my_moves - opp_moves;
}

// --- MINIMAX ---
std::pair<int, Move> minimax(Board board, int depth, int alpha, int beta, bool is_maximizing, char current_player, char original_player, TimePoint start_time, Stats& stats, TranspositionTable& tt, std::atomic<bool>& time_out, uint64_t hash_key, int history[64][64]) {
    
    if ((stats.nodes_evaluated & 2047) == 0) {
        if (get_elapsed(start_time) > THINKING_TIME) {
            time_out.store(true, std::memory_order_relaxed);
            throw TimeoutException();
        }
    }
    if (time_out.load(std::memory_order_relaxed)) throw TimeoutException();
    
    stats.nodes_evaluated++;

    int tt_score;
    Move tt_move;
    if (tt.probe(hash_key, depth, alpha, beta, tt_score, tt_move)) {
        stats.tt_hits++;
        return {tt_score, tt_move}; 
    }

    if (depth == 0) {
        int eval = evaluate_board(board, original_player);
        tt.store(hash_key, depth, eval, TTFlag::EXACT, Move()); 
        return {eval, Move()};
    }

    MoveList moves = get_legal_moves(board, current_player);
    
    if (moves.empty()) {
        int eval = (current_player == original_player) ? -10000 - depth : 10000 + depth;
        tt.store(hash_key, depth, eval, TTFlag::EXACT, Move());
        return {eval, Move()};
    }

    int scores[128];
    for (size_t i = 0; i < moves.size(); ++i) {
        if (moves[i] == tt_move) {
            scores[i] = 10000000; 
        } else {
            int jump_dist = moves[i].is_jump ? std::abs(moves[i].end_r - moves[i].start_r) + std::abs(moves[i].end_c - moves[i].start_c) : 0;
            int start_idx = moves[i].start_r * 8 + moves[i].start_c;
            int end_idx = moves[i].end_r * 8 + moves[i].end_c;
            scores[i] = (jump_dist * 1000) + history[start_idx][end_idx]; 
        }
    }
    
    for (size_t i = 1; i < moves.size(); ++i) {
        Move key_m = moves[i];
        int key_s = scores[i];
        int j = i - 1;
        while (j >= 0 && scores[j] < key_s) {
            moves[j + 1] = moves[j];
            scores[j + 1] = scores[j];
            j--;
        }
        moves[j + 1] = key_m;
        scores[j + 1] = key_s;
    }

    Move best_move = moves[0];
    char next_player = (current_player == 'B') ? 'W' : 'B';
    int orig_alpha = alpha;
    int orig_beta = beta;

    if (is_maximizing) {
        int max_eval = -1e9;
        int move_idx = 0;
        for (const auto& move : moves) {
            Board new_board = board;
            uint64_t new_hash = hash_key;
            apply_move(new_board, move, current_player, new_hash);
            
            int eval_score;
            if (depth >= 3 && move_idx >= 3) {
                auto[score, _] = minimax(new_board, depth - 2, alpha, beta, false, next_player, original_player, start_time, stats, tt, time_out, new_hash, history);
                eval_score = score;
                if (eval_score > alpha) { 
                    auto[full_score, _] = minimax(new_board, depth - 1, alpha, beta, false, next_player, original_player, start_time, stats, tt, time_out, new_hash, history);
                    eval_score = full_score;
                }
            } else {
                auto[score, _] = minimax(new_board, depth - 1, alpha, beta, false, next_player, original_player, start_time, stats, tt, time_out, new_hash, history);
                eval_score = score;
            }
            
            if (eval_score > max_eval) { max_eval = eval_score; best_move = move; }
            alpha = std::max(alpha, eval_score);
            if (beta <= alpha) { stats.cutoffs++; history[move.start_r * 8 + move.start_c][move.end_r * 8 + move.end_c] += (depth * depth); break; }
            move_idx++;
        }
        
        TTFlag flag = TTFlag::EXACT;
        if (max_eval <= orig_alpha) flag = TTFlag::UPPERBOUND;
        else if (max_eval >= orig_beta) flag = TTFlag::LOWERBOUND;
        tt.store(hash_key, depth, max_eval, flag, best_move); 
        return {max_eval, best_move};
        
    } else {
        int min_eval = 1e9;
        int move_idx = 0;
        for (const auto& move : moves) {
            Board new_board = board;
            uint64_t new_hash = hash_key;
            apply_move(new_board, move, current_player, new_hash);
            
            int eval_score;
            if (depth >= 3 && move_idx >= 3) {
                auto[score, _] = minimax(new_board, depth - 2, alpha, beta, true, next_player, original_player, start_time, stats, tt, time_out, new_hash, history);
                eval_score = score;
                if (eval_score < beta) {
                    auto[full_score, _] = minimax(new_board, depth - 1, alpha, beta, true, next_player, original_player, start_time, stats, tt, time_out, new_hash, history);
                    eval_score = full_score;
                }
            } else {
                auto[score, _] = minimax(new_board, depth - 1, alpha, beta, true, next_player, original_player, start_time, stats, tt, time_out, new_hash, history);
                eval_score = score;
            }
            
            if (eval_score < min_eval) { min_eval = eval_score; best_move = move; }
            beta = std::min(beta, eval_score);
            if (beta <= alpha) { stats.cutoffs++; history[move.start_r * 8 + move.start_c][move.end_r * 8 + move.end_c] += (depth * depth); break; }
            move_idx++;
        }
        
        TTFlag flag = TTFlag::EXACT;
        if (min_eval <= orig_alpha) flag = TTFlag::UPPERBOUND;
        else if (min_eval >= orig_beta) flag = TTFlag::LOWERBOUND;
        tt.store(hash_key, depth, min_eval, flag, best_move); 
        return {min_eval, best_move};
    }
}

std::string format_with_commas(long long value) {
    std::string s = std::to_string(value);
    int insert_pos = (int)s.length() - 3;
    while (insert_pos > 0) {
        s.insert(insert_pos, ",");
        insert_pos -= 3;
    }
    return s;
}

Move get_best_move(const Board& board, char player, TranspositionTable& global_tt, bool debug_thinking = true) {
    auto start_time = std::chrono::high_resolution_clock::now();
    MoveList legal_moves = get_legal_moves(board, player);
    if (legal_moves.empty()) return Move();
    
    // STRICT CORE ENFORCEMENT
    unsigned int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 4; 
    num_cores = std::min(num_cores, 4u); 
    
    std::atomic<bool> time_out{false};
    uint64_t root_hash = compute_zobrist(board, player);
    
    Move best_move = legal_moves[0];
    int last_completed_depth = 0;
    std::vector<std::pair<Move, int>> last_completed_evals;

    // Use raw threads so we can control CPU properties
    std::vector<std::thread> workers;
    std::vector<Stats> thread_stats(num_cores);

    auto worker_thread = [&](int thread_id) {
        // --- SMART LINUX THREAD PINNING ---
        // Reads which cores we are legally allowed to use, and securely binds
        // this exact thread to one of them permanently.
        #if defined(__linux__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == 0) {
            std::vector<int> available_cores;
            for (int i = 0; i < CPU_SETSIZE; ++i) {
                if (CPU_ISSET(i, &cpuset)) available_cores.push_back(i);
            }
            if (!available_cores.empty()) {
                cpu_set_t single_core;
                CPU_ZERO(&single_core);
                int target_core = available_cores[thread_id % available_cores.size()];
                CPU_SET(target_core, &single_core);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &single_core);
            }
        }
        #endif

        Stats local_stats;
        MoveList moves = legal_moves;
        std::mt19937 rng(12345 + thread_id); 
        char next_player = (player == 'B') ? 'W' : 'B';
        
        int history[64][64] = {0}; 
        
        try {
            int depth = 1;
            while (!time_out.load(std::memory_order_relaxed)) {
                if (thread_id > 0) std::shuffle(moves.begin(), moves.end(), rng);
                
                int alpha = -1e9;
                int beta = 1e9;
                int alpha_orig = alpha;
                int beta_orig = beta;
                
                bool use_window = (thread_id == 0 && depth >= 3 && !last_completed_evals.empty());
                if (use_window) {
                    alpha = last_completed_evals[0].second - 50;
                    beta = last_completed_evals[0].second + 50;
                    alpha_orig = alpha;
                    beta_orig = beta;
                }

                while (true && !time_out.load(std::memory_order_relaxed)) {
                    int max_eval = -1e9;
                    Move current_best_move = moves[0];
                    std::vector<std::pair<Move, int>> current_scores;
                    int current_alpha = alpha;
                    
                    for (const auto& move : moves) {
                        Board new_board = board;
                        uint64_t new_hash = root_hash;
                        apply_move(new_board, move, player, new_hash);
                        
                        auto [score, _] = minimax(new_board, depth - 1, current_alpha, beta, false, next_player, player, start_time, local_stats, global_tt, time_out, new_hash, history);
                        
                        if (thread_id == 0) current_scores.push_back({move, score});
                        
                        if (score > max_eval) { max_eval = score; current_best_move = move; }
                        current_alpha = std::max(current_alpha, score);
                    }
                    
                    if (use_window && (max_eval <= alpha_orig || max_eval >= beta_orig) && !time_out.load(std::memory_order_relaxed)) {
                        alpha = -1e9; beta = 1e9; use_window = false; continue;
                    }
                    
                    if (thread_id == 0) {
                        best_move = current_best_move;
                        std::sort(current_scores.begin(), current_scores.end(),[](const auto& a, const auto& b) { return a.second > b.second; });
                        last_completed_evals = current_scores;
                        last_completed_depth = depth;
                        moves.clear();
                        for (const auto& eval : current_scores) moves.push_back(eval.first);
                    }
                    break; 
                }
                if (depth >= 64) break; 
                depth++;
            }
        } catch (const TimeoutException&) {
            time_out.store(true, std::memory_order_relaxed);
        }
        thread_stats[thread_id] = local_stats;
    };

    // Spin up the explicit raw threads
    for (unsigned int i = 1; i < num_cores; ++i) {
        workers.emplace_back(worker_thread, i);
    }
    
    // The main thread does the work for Thread 0
    worker_thread(0);
    
    // Wait for all worker threads to safely finish
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
    
    Stats global_stats;
    for (const auto& s : thread_stats) {
        global_stats.nodes_evaluated += s.nodes_evaluated;
        global_stats.cutoffs += s.cutoffs;
        global_stats.tt_hits += s.tt_hits;
    }
    
    double time_taken = get_elapsed(start_time);
        
    if (debug_thinking && !last_completed_evals.empty()) {
        std::cout << "\n[THOUGHT PROCESS (Depth " << last_completed_depth << ")]\n";
        std::cout << "   Active CPU Threads : " << num_cores << "\n";
        std::cout << "   Nodes Evaluated    : " << format_with_commas(global_stats.nodes_evaluated) 
                  << " | Alpha-Beta Cutoffs: " << format_with_commas(global_stats.cutoffs) << "\n";
        std::cout << "   TT Cache Hits      : " << format_with_commas(global_stats.tt_hits) << "\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "   Time Taken         : " << time_taken << " seconds (Time Limit: " << THINKING_TIME << " seconds)\n";
        
        long long nps = (time_taken > 0) ? static_cast<long long>(global_stats.nodes_evaluated / time_taken) : 0;
        std::cout << "   Search Speed       : " << format_with_commas(nps) << " nodes/sec\n\n";
    }
    return best_move;
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \n\r\t");
    if (std::string::npos == first) return "";
    size_t last = str.find_last_not_of(" \n\r\t");
    return str.substr(first, (last - first + 1));
}

Board parse_board_file(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin.is_open()) { std::cerr << "Could not open file: " << filename << std::endl; exit(1); }
    std::string content, line;
    while (std::getline(fin, line)) content += line;
    content.erase(std::remove(content.begin(), content.end(), '\r'), content.end());
    
    Board board;
    for (int i = 0; i < 64 && i < content.length(); ++i) {
        if (content[i] == 'B') board.b |= (1ULL << i);
        else if (content[i] == 'W') board.w |= (1ULL << i);
    }
    return board;
}

Move parse_move(const std::string& str) {
    Move m;
    m.start_c = str[0] - 'A';
    m.start_r = 8 - (str[1] - '0');
    
    size_t dash = str.find('-');
    if (dash != std::string::npos && dash + 2 < str.length()) {
        m.is_jump = true;
        m.end_c = str[dash + 1] - 'A';
        m.end_r = 8 - (str[dash + 2] - '0');
    } else {
        m.is_jump = false;
        m.end_c = m.start_c;
        m.end_r = m.start_r;
    }
    return m;
}

int main(int argc, char* argv[]) {
    if (argc != 3) { std::cerr << "Invalid input\n"; return 1; }
    init_zobrist();

    TranspositionTable global_tt;

    std::string board_file = argv[1];
    char colour = argv[2][0];
    Board board = parse_board_file(board_file);
    
    uint64_t fake_hash = 0; 
    MoveList moves = get_legal_moves(board, colour);
    if (moves.empty()) return 0; 

    Move best_move = get_best_move(board, colour, global_tt, false); 
    std::cout << best_move.to_string() << std::endl; 
    apply_move(board, best_move, colour, fake_hash);

    std::string opponent_move_str;
    while (std::getline(std::cin, opponent_move_str)) {
        opponent_move_str = trim(opponent_move_str);
        if (opponent_move_str.empty()) continue; 

        char opp_colour = (colour == 'B') ? 'W' : 'B';
        Move opp_move = parse_move(opponent_move_str);
        
        apply_move(board, opp_move, opp_colour, fake_hash);

        moves = get_legal_moves(board, colour);
        if (moves.empty()) break; 

        best_move = get_best_move(board, colour, global_tt, false);
        std::cout << best_move.to_string() << std::endl;
        apply_move(board, best_move, colour, fake_hash);
    }
    return 0;
}