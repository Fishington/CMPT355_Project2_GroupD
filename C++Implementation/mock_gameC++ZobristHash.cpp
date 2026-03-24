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
#include <mutex>
#include <atomic>
#include <memory>

// $ g++ -std=c++17 mock_gameC++ZobristHash.cpp -o konaneCPP -O3 -march=native


// --- SETTINGS ---
const double THINKING_TIME = 10.0;  // 10 seconds per turn

class TimeoutException : public std::exception {};

struct Board {
    // Bitboard! 
    // Instead of having a 2D array like char board[8][8], we can use Bitboards
    // As a standard 64-bit integer uint64_t has exactly 64 bits
    // b holds all black pieces
    // w holds all white pieces
    // Performance is hella good this way, it only takes 1 CPU instruction using bitwise math than compiler functions 
    uint64_t b = 0; 
    uint64_t w = 0; 
};

struct Move {
    int start_r = 0, start_c = 0;
    int end_r = 0, end_c = 0;
    bool is_jump = false;

    bool operator==(const Move& o) const {
        return start_r == o.start_r && start_c == o.start_c &&
               end_r == o.end_r && end_c == o.end_c && is_jump == o.is_jump;
    }

    std::string to_string() const {
        auto coords_to_notation =[](int r, int c) {
            char col_char = (char)(c + 'A');
            char row_char = std::to_string(8 - r)[0];
            return std::string{col_char, row_char};
        };
        if (!is_jump) return coords_to_notation(start_r, start_c);
        return coords_to_notation(start_r, start_c) + "-" + coords_to_notation(end_r, end_c);
    }
};

struct Stats {
    long long nodes_evaluated = 0;
    long long cutoffs = 0;
    long long tt_hits = 0; 
};

// --- ZOBRIST HASHING SETUP ---
// We use Zobrist Hashing to check to see if a board state has already been seen. 
// So we turn the board into a unique ID (or Hash)

// XOR is incredibly fast and pretty much guarantees a near-unique hash for all board states. Which will be used as a 
// lookup key for the transposition table

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

// --- GLOBAL SHARED TRANSPOSITION TABLE (LAZY SMP) ---

// When the AI looks ahead, it will encounter the exact same board state through different move orders
// The TT table stores the evaluation of previously seen boards so no time waste on reclac
enum class TTFlag : uint8_t { EXACT, LOWERBOUND, UPPERBOUND };

struct TTEntry {
    uint64_t key;
    int score;
    int depth;
    TTFlag flag;
    Move best_move; 
};

// 1,048,576 entries ensures a massive memory pool for the hivemind
const int TT_SIZE = 1048576; 
const int LOCKS_SIZE = 4096; // 4096 mutexes prevents Threads from waiting in line for memory

struct TranspositionTable {

    // We need to make sure that there are NO LOCKS on this, so we create 4096 seperate locks. 
    // A thread only locks the specific tiny section of the table it needs, which allow for concurrency! 
    std::vector<TTEntry> table;
    std::unique_ptr<std::mutex[]> locks; // Striped Locking prevents memory bottlenecks

    TranspositionTable() : 
        table(TT_SIZE, {0, 0, -1, TTFlag::EXACT, Move()}), 
        locks(new std::mutex[LOCKS_SIZE]) {}
    
    void store(uint64_t key, int depth, int score, TTFlag flag, Move best_move) {
        int index = key & (TT_SIZE - 1); 
        int lock_idx = index & (LOCKS_SIZE - 1);
        
        std::lock_guard<std::mutex> lock(locks[lock_idx]);
        if (table[index].depth <= depth) { 
            table[index] = {key, score, depth, flag, best_move};
        }
    }
    
    bool probe(uint64_t key, int depth, int alpha, int beta, int& out_score, Move& out_move) {
        int index = key & (TT_SIZE - 1);
        int lock_idx = index & (LOCKS_SIZE - 1);
        
        std::lock_guard<std::mutex> lock(locks[lock_idx]);
        TTEntry entry = table[index];
        
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

// struct TranspositionTable {
//     std::vector<TTEntry> table;
//     std::unique_ptr<std::mutex[]> locks;
    
//     // --- NEW: Add an array of atomic counters ---
//     std::unique_ptr<std::atomic<uint64_t>[]> lock_usage_counts;

//     TranspositionTable() : 
//         table(TT_SIZE, {0, 0, -1, TTFlag::EXACT, Move()}), 
//         locks(new std::mutex[LOCKS_SIZE]),
//         // --- NEW: Initialize the counters ---
//         lock_usage_counts(new std::atomic<uint64_t>[LOCKS_SIZE]) {
        
//         for(int i = 0; i < LOCKS_SIZE; ++i) {
//             lock_usage_counts[i].store(0);
//         }
//     }
    
//     void store(uint64_t key, int depth, int score, TTFlag flag, Move best_move) {
//         int index = key & (TT_SIZE - 1); 
//         int lock_idx = index & (LOCKS_SIZE - 1);
        
//         // --- NEW: Track that this specific lock was hit ---
//         lock_usage_counts[lock_idx].fetch_add(1, std::memory_order_relaxed);
        
//         std::lock_guard<std::mutex> lock(locks[lock_idx]);
//         if (table[index].depth <= depth) { 
//             table[index] = {key, score, depth, flag, best_move};
//         }
//     }
    
//     bool probe(uint64_t key, int depth, int alpha, int beta, int& out_score, Move& out_move) {
//         int index = key & (TT_SIZE - 1);
//         int lock_idx = index & (LOCKS_SIZE - 1);
        
//         // --- NEW: Track that this specific lock was hit ---
//         lock_usage_counts[lock_idx].fetch_add(1, std::memory_order_relaxed);
        
//         std::lock_guard<std::mutex> lock(locks[lock_idx]);
//         TTEntry entry = table[index];
        
//         if (entry.key == key) {
//             out_move = entry.best_move; 
            
//             if (entry.depth >= depth) {
//                 if (entry.flag == TTFlag::EXACT) { out_score = entry.score; return true; }
//                 if (entry.flag == TTFlag::LOWERBOUND && entry.score >= beta) { out_score = entry.score; return true; }
//                 if (entry.flag == TTFlag::UPPERBOUND && entry.score <= alpha) { out_score = entry.score; return true; }
//             }
//         }
//         return false;
//     }
    
//     // --- NEW: A helper function to print the lock statistics ---
//     void print_lock_debug_stats() {
//         int unique_locks_used = 0;
//         uint64_t total_lock_hits = 0;
//         uint64_t max_hits_on_single_lock = 0;
        
//         for(int i = 0; i < LOCKS_SIZE; i++) {
//             uint64_t hits = lock_usage_counts[i].load(std::memory_order_relaxed);
//             if(hits > 0) {
//                 unique_locks_used++;
//                 total_lock_hits += hits;
//                 if(hits > max_hits_on_single_lock) {
//                     max_hits_on_single_lock = hits;
//                 }
//             }
//         }
        
//         std::cout << "   --- LOCK DEBUG STATS ---\n";
//         std::cout << "   Unique Locks Used : " << unique_locks_used << " out of " << LOCKS_SIZE << "\n";
//         std::cout << "   Total Lock Hits   : " << format_with_commas(total_lock_hits) << "\n";
//         std::cout << "   Max Hits on 1 Lock: " << format_with_commas(max_hits_on_single_lock) << "\n";
//         if (unique_locks_used > 0) {
//             std::cout << "   Avg Hits per Lock : " << format_with_commas(total_lock_hits / unique_locks_used) << "\n";
//         }
//         std::cout << "   ------------------------\n";
//     }
// };

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

double get_elapsed(TimePoint start) {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

// --- GAME LOGIC ---
Board create_initial_board() {
    Board board;
    board.b = 0xAA55AA55AA55AA55ULL;
    board.w = 0x55AA55AA55AA55AAULL;
    return board;
}

void print_board(const Board& board) {
    std::cout << "   A B C D E F G H\n  -----------------\n";
    for (int r = 0; r < 8; ++r) {
        std::cout << " " << (8 - r) << " |";
        for (int c = 0; c < 8; ++c) {
            int idx = r * 8 + c;
            if (board.b & (1ULL << idx)) std::cout << "B";
            else if (board.w & (1ULL << idx)) std::cout << "W";
            else std::cout << ".";
            std::cout << (c < 7 ? " " : "");
        }
        std::cout << "| " << (8 - r) << "\n";
    }
    std::cout << "  -----------------\n   A B C D E F G H\n\n";
}

Board clone_board(const Board& board) { return board; }

void apply_move(Board& board, const Move& move, char current_player) {
    uint64_t& my_pieces = (current_player == 'B') ? board.b : board.w;
    uint64_t& opp_pieces = (current_player == 'B') ? board.w : board.b;
    
    int start_idx = move.start_r * 8 + move.start_c;
    
    if (!move.is_jump) { 
        my_pieces &= ~(1ULL << start_idx); 
        return;
    }

    int dr = (move.end_r == move.start_r) ? 0 : ((move.end_r > move.start_r) ? 1 : -1);
    int dc = (move.end_c == move.start_c) ? 0 : ((move.end_c > move.start_c) ? 1 : -1);
    
    int end_idx = move.end_r * 8 + move.end_c;
    int delta_idx = dr * 8 + dc; 
    
    int curr_idx = start_idx;
    my_pieces &= ~(1ULL << curr_idx); 
    
    while (curr_idx != end_idx) {
        curr_idx += delta_idx;
        if (curr_idx != end_idx) {
            opp_pieces &= ~(1ULL << curr_idx); 
            curr_idx += delta_idx;
        }
    }
    my_pieces |= (1ULL << end_idx); 
}

std::vector<Move> get_legal_moves(const Board& board, char player) {
    std::vector<Move> moves;
    moves.reserve(32); 
    
    int empty_count = 64 - __builtin_popcountll(board.b | board.w);
    
    if (empty_count == 0) {
        if (player == 'B') {
            moves.push_back({3, 3, 3, 3, false}); 
            moves.push_back({4, 4, 4, 4, false}); 
        }
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
        
        int r = idx / 8;
        int c = idx % 8;
        
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
                        moves.push_back({r, c, dest_r, dest_c, true});
                        curr_r = dest_r;
                        curr_c = dest_c;
                    } else break;
                } else break;
            }
        }
    }
    return moves;
}

// --- MINIMAX  ---
int evaluate_board(const Board& board, char player) {
    char opp = (player == 'B') ? 'W' : 'B';
    int my_moves = get_legal_moves(board, player).size();
    int opp_moves = get_legal_moves(board, opp).size();
    
    if (opp_moves == 0) return 10000; 
    if (my_moves == 0) return -10000;
    return my_moves - opp_moves;
}

std::pair<int, Move> minimax(Board board, int depth, int alpha, int beta, bool is_maximizing, char current_player, char original_player, TimePoint start_time, Stats& stats, TranspositionTable& tt, std::atomic<bool>& time_out) {
    
    // OPTIMIZATION: Only 1 thread hits the system clock occasionally. If true, it flips atomic bool for everyone.
    if ((stats.nodes_evaluated & 2047) == 0) {
        if (get_elapsed(start_time) > THINKING_TIME) {
            time_out.store(true, std::memory_order_relaxed);
            throw TimeoutException();
        }
    }
    
    // Atomic bool check takes almost 0 CPU cycles compared to a chrono clock evaluation
    if (time_out.load(std::memory_order_relaxed)) {
        throw TimeoutException();
    }
    
    stats.nodes_evaluated++;

    uint64_t hash_key = compute_zobrist(board, current_player);
    int tt_score;
    Move tt_move;
    
    if (tt.probe(hash_key, depth, alpha, beta, tt_score, tt_move)) {
        stats.tt_hits++;
        return {tt_score, tt_move}; 
    }

    auto moves = get_legal_moves(board, current_player);
    
    if (depth == 0 || moves.empty()) {
        int eval = evaluate_board(board, original_player);
        tt.store(hash_key, depth, eval, TTFlag::EXACT, Move()); 
        return {eval, Move()};
    }

    auto score_move = [&](const Move& m) {
        if (m == tt_move) return 1000000;
        if (!m.is_jump) return 0;
        return std::abs(m.end_r - m.start_r) + std::abs(m.end_c - m.start_c);
    };

    std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        return score_move(a) > score_move(b);
    });

    Move best_move = moves[0];
    char next_player = (current_player == 'B') ? 'W' : 'B';
    
    int orig_alpha = alpha;
    int orig_beta = beta;

    if (is_maximizing) {
        int max_eval = -1e9;
        for (const auto& move : moves) {
            Board new_board = clone_board(board);
            apply_move(new_board, move, current_player);
            
            auto[eval_score, _] = minimax(new_board, depth - 1, alpha, beta, false, next_player, original_player, start_time, stats, tt, time_out);
            
            if (eval_score > max_eval) {
                max_eval = eval_score;
                best_move = move;
            }
            alpha = std::max(alpha, eval_score);
            if (beta <= alpha) {
                stats.cutoffs++; 
                break; 
            }
        }
        
        TTFlag flag = TTFlag::EXACT;
        if (max_eval <= orig_alpha) flag = TTFlag::UPPERBOUND;
        else if (max_eval >= orig_beta) flag = TTFlag::LOWERBOUND;
        
        tt.store(hash_key, depth, max_eval, flag, best_move); 
        return {max_eval, best_move};
        
    } else {
        int min_eval = 1e9;
        for (const auto& move : moves) {
            Board new_board = clone_board(board);
            apply_move(new_board, move, current_player);
            auto[eval_score, _] = minimax(new_board, depth - 1, alpha, beta, true, next_player, original_player, start_time, stats, tt, time_out);
            
            if (eval_score < min_eval) {
                min_eval = eval_score;
                best_move = move;
            }
            beta = std::min(beta, eval_score);
            if (beta <= alpha) {
                stats.cutoffs++; 
                break;
            }
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

Move get_best_move(const Board& board, char player, bool debug_thinking = true) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto legal_moves = get_legal_moves(board, player);
    if (legal_moves.empty()) return Move();
    
    // Grabs ALL available CPU Cores natively!
    unsigned int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 4; // Failsafe
    
    // The Global Hivemind Transposition Table
    TranspositionTable global_tt;
    
    // Lightning fast cross-thread abort signal
    std::atomic<bool> time_out{false};
    
    Move best_move = legal_moves[0];
    int last_completed_depth = 0;
    std::vector<std::pair<Move, int>> last_completed_evals;

    // --- LAZY SMP WORKER FUNCTION ---
    auto worker_thread = [&](int thread_id) {
        Stats local_stats;
        auto moves = legal_moves;
        std::mt19937 rng(12345 + thread_id); // Unique seed for thread divergence
        char next_player = (player == 'B') ? 'W' : 'B';
        
        try {
            int depth = 1;
            while (!time_out.load(std::memory_order_relaxed)) {
                
                // MAIN THREAD (0) sorts perfectly. HELPER THREADS (1+) shuffle moves to find random hidden combinations!
                if (thread_id > 0) {
                    std::shuffle(moves.begin(), moves.end(), rng);
                }
                
                std::vector<std::pair<Move, int>> current_scores;
                int alpha = -1e9;
                int beta = 1e9;
                int max_eval = -1e9;
                Move current_best_move = moves[0];
                
                for (const auto& move : moves) {
                    Board new_board = clone_board(board);
                    apply_move(new_board, move, player);
                    
                    auto [score, _] = minimax(new_board, depth - 1, alpha, beta, false, next_player, player, start_time, local_stats, global_tt, time_out);
                    
                    if (thread_id == 0) {
                        current_scores.push_back({move, score});
                    }
                    
                    if (score > max_eval) {
                        max_eval = score;
                        current_best_move = move;
                    }
                    alpha = std::max(alpha, score);
                }
                
                // Only Thread 0 dictates the actual move we pick for the game
                if (thread_id == 0) {
                    best_move = current_best_move;
                    
                    std::sort(current_scores.begin(), current_scores.end(),[](const auto& a, const auto& b) { 
                        return a.second > b.second; 
                    });
                    
                    last_completed_evals = current_scores;
                    last_completed_depth = depth;
                    
                    moves.clear();
                    for (const auto& eval : current_scores) moves.push_back(eval.first);
                }
                
                if (max_eval >= 10000) break;
                depth++;
            }
        } catch (const TimeoutException&) {
            // Tell all other threads that time is up!
            time_out.store(true, std::memory_order_relaxed);
        }
        
        return local_stats;
    };

    // Spin up all Helper Threads
    std::vector<std::future<Stats>> futures;
    for (unsigned int i = 1; i < num_cores; ++i) {
        futures.push_back(std::async(std::launch::async, worker_thread, i));
    }
    
    // Spin up the Main Thread (Thread 0) on this thread
    Stats global_stats = worker_thread(0);
    
    // Gather all Stats
    for (auto& fut : futures) {
        Stats s = fut.get();
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
        std::cout << "   TT Cache Hits      : " << format_with_commas(global_stats.tt_hits) << " (Hivemind Shared Data)\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "   Time Taken         : " << time_taken << " seconds (Time Limit: " << THINKING_TIME << " seconds)\n";
        
        long long nps = (time_taken > 0) ? static_cast<long long>(global_stats.nodes_evaluated / time_taken) : 0;
        std::cout << "   Search Speed       : " << format_with_commas(nps) << " nodes/sec\n\n";

        
        
        for (const auto& eval : last_completed_evals) {
            const Move& m = eval.first;
            int s = eval.second;
            std::string score_str;
            if (s >= 9000) score_str = "WINNING TRAP FOUND!";
            else if (s <= -9000) score_str = "AVOIDING LOSS!";
            else { char buf[64]; snprintf(buf, sizeof(buf), "%+d move advantage", s); score_str = buf; }
            
            std::string marker = (m == best_move) ? ">> " : "   ";
            std::cout << "    " << marker << "Move: " << std::left << std::setw(8) << m.to_string() 
                      << " | Score: " << score_str << "\n";
        }
        std::cout << "\n";
    }
            
    return best_move;
}

// --- MOCK GAME LOOP ---
void play_mock_game() {
    std::cout << "=== KONANE MOCK GAME ===\n";
    
    Board board = create_initial_board();
    char current_player = 'B';
    int turn_number = 1;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    while (true) {
        std::cout << "--- Turn " << turn_number << " ---\n";
        print_board(board);
        
        auto moves = get_legal_moves(board, current_player);
        if (moves.empty()) {
            std::string winner = (current_player == 'B') ? "White" : "Black";
            std::cout << "*** GAME OVER! " << current_player << " has no legal moves. ***\n";
            std::cout << "*** " << winner << " WINS! ***\n";
            break;
        }
        
        Move move;
        if (current_player == 'B') {
            std::cout << "Black (AI) is thinking...\n";
            move = get_best_move(board, current_player, true);
        } else {
            std::cout << "White (Random) is picking...\n";
            std::uniform_int_distribution<> dist(0, moves.size() - 1);
            move = moves[dist(gen)];
            std::this_thread::sleep_for(std::chrono::seconds(1)); 
        }
            
        std::cout << "-> Player " << current_player << " plays: " << move.to_string() << "\n\n";
        apply_move(board, move, current_player);
        current_player = (current_player == 'B') ? 'W' : 'B';
        turn_number++;
    }
}

int main() {
    init_zobrist();
    play_mock_game();
    return 0;
}