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

// --- SETTINGS ---
const double THINKING_TIME = 9.8;  // 10 seconds per turn

class TimeoutException : public std::exception {};

struct Board {
    uint64_t b = 0; 
    uint64_t w = 0; 
};

// COMPACTED: Shrunk to 5 bytes to keep TT entries small
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

// ZERO-ALLOCATION MOVES: Eliminates standard vector allocations in search tree
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

// --- ZOBRIST HASHING SETUP ---
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
enum class TTFlag : uint8_t { EXACT, LOWERBOUND, UPPERBOUND };

struct TTEntry {
    uint64_t key;
    int score;
    int depth;
    TTFlag flag;
    Move best_move; 
};

const int TT_SIZE = 1048576; 

// SPINLOCK: Lightning fast alternative to std::mutex for multi-threading
struct SpinLock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT;
    inline void lock() { while (locked.test_and_set(std::memory_order_acquire)) {} }
    inline void unlock() { locked.clear(std::memory_order_release); }
};

struct TranspositionTable {
    std::vector<TTEntry> table;
    std::unique_ptr<SpinLock[]> locks; 

    TranspositionTable() : 
        table(TT_SIZE, {0, 0, -1, TTFlag::EXACT, Move()}), 
        locks(new SpinLock[TT_SIZE]) {}
    
    void store(uint64_t key, int depth, int score, TTFlag flag, Move best_move) {
        int index = key & (TT_SIZE - 1); 
        
        locks[index].lock();
        if (table[index].depth <= depth) { 
            table[index] = {key, score, depth, flag, best_move};
        }
        locks[index].unlock();
    }
    
    bool probe(uint64_t key, int depth, int alpha, int beta, int& out_score, Move& out_move) {
        int index = key & (TT_SIZE - 1);
        
        locks[index].lock();
        TTEntry entry = table[index];
        locks[index].unlock();
        
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

// INCREMENTAL ZOBRIST: Avoids recomputing hash from scratch on every turn
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
                        moves.push_back({(int8_t)r, (int8_t)c, (int8_t)dest_r, (int8_t)dest_c, true});
                        curr_r = dest_r;
                        curr_c = dest_c;
                    } else break;
                } else break;
            }
        }
    }
    return moves;
}

// BITBOARD OPTIMIZATION: Extremely fast move counter without generating structures
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

    uint64_t my_pieces = (player == 'B') ? board.b : board.w;
    uint64_t opp_pieces = (player == 'B') ? board.w : board.b;
    uint64_t empty = ~(board.b | board.w);
    int count = 0;
    
    while (my_pieces) {
        int idx = __builtin_ctzll(my_pieces);
        my_pieces &= my_pieces - 1;
        int r = idx / 8, c = idx % 8;
        
        int curr_r = r;
        while (curr_r >= 2) {
            if ((opp_pieces & (1ULL << ((curr_r - 1) * 8 + c))) && (empty & (1ULL << ((curr_r - 2) * 8 + c)))) { count++; curr_r -= 2; } else break;
        }
        curr_r = r;
        while (curr_r <= 5) {
            if ((opp_pieces & (1ULL << ((curr_r + 1) * 8 + c))) && (empty & (1ULL << ((curr_r + 2) * 8 + c)))) { count++; curr_r += 2; } else break;
        }
        int curr_c = c;
        while (curr_c >= 2) {
            if ((opp_pieces & (1ULL << (r * 8 + curr_c - 1))) && (empty & (1ULL << (r * 8 + curr_c - 2)))) { count++; curr_c -= 2; } else break;
        }
        curr_c = c;
        while (curr_c <= 5) {
            if ((opp_pieces & (1ULL << (r * 8 + curr_c + 1))) && (empty & (1ULL << (r * 8 + curr_c + 2)))) { count++; curr_c += 2; } else break;
        }
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
std::pair<int, Move> minimax(Board board, int depth, int alpha, int beta, bool is_maximizing, char current_player, char original_player, TimePoint start_time, Stats& stats, TranspositionTable& tt, std::atomic<bool>& time_out, uint64_t hash_key) {
    
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
        // DEPTH ADJUSTMENT: Prioritizes taking the fastest paths to victory, or longest paths to defeat
        int eval = (current_player == original_player) ? -10000 - depth : 10000 + depth;
        tt.store(hash_key, depth, eval, TTFlag::EXACT, Move());
        return {eval, Move()};
    }

    // INLINE MOVE ORDERING: Bubbles cached TT move straight to the front
    int tt_move_idx = -1;
    for (size_t i = 0; i < moves.size(); ++i) {
        if (moves[i] == tt_move) { tt_move_idx = i; break; }
    }
    if (tt_move_idx != -1 && tt_move_idx != 0) {
        std::swap(moves[0], moves[tt_move_idx]);
    }
    
    // Sort remainder safely (jumps by lengths to maximize alpha-beta pruning)
    if (moves.size() > 2) {
        int scores[128];
        for (size_t i = 1; i < moves.size(); ++i) {
            scores[i] = moves[i].is_jump ? std::abs(moves[i].end_r - moves[i].start_r) + std::abs(moves[i].end_c - moves[i].start_c) : 0;
        }
        for (size_t i = 2; i < moves.size(); ++i) {
            Move key_m = moves[i];
            int key_s = scores[i];
            int j = i - 1;
            while (j >= 1 && scores[j] < key_s) {
                moves[j + 1] = moves[j];
                scores[j + 1] = scores[j];
                j--;
            }
            moves[j + 1] = key_m;
            scores[j + 1] = key_s;
        }
    }

    Move best_move = moves[0];
    char next_player = (current_player == 'B') ? 'W' : 'B';
    int orig_alpha = alpha;
    int orig_beta = beta;

    if (is_maximizing) {
        int max_eval = -1e9;
        for (const auto& move : moves) {
            Board new_board = board;
            uint64_t new_hash = hash_key;
            apply_move(new_board, move, current_player, new_hash);
            
            auto[eval_score, _] = minimax(new_board, depth - 1, alpha, beta, false, next_player, original_player, start_time, stats, tt, time_out, new_hash);
            
            if (eval_score > max_eval) {
                max_eval = eval_score;
                best_move = move;
            }
            alpha = std::max(alpha, eval_score);
            if (beta <= alpha) { stats.cutoffs++; break; }
        }
        
        TTFlag flag = TTFlag::EXACT;
        if (max_eval <= orig_alpha) flag = TTFlag::UPPERBOUND;
        else if (max_eval >= orig_beta) flag = TTFlag::LOWERBOUND;
        tt.store(hash_key, depth, max_eval, flag, best_move); 
        return {max_eval, best_move};
        
    } else {
        int min_eval = 1e9;
        for (const auto& move : moves) {
            Board new_board = board;
            uint64_t new_hash = hash_key;
            apply_move(new_board, move, current_player, new_hash);
            
            auto[eval_score, _] = minimax(new_board, depth - 1, alpha, beta, true, next_player, original_player, start_time, stats, tt, time_out, new_hash);
            
            if (eval_score < min_eval) {
                min_eval = eval_score;
                best_move = move;
            }
            beta = std::min(beta, eval_score);
            if (beta <= alpha) { stats.cutoffs++; break; }
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
    MoveList legal_moves = get_legal_moves(board, player);
    if (legal_moves.empty()) return Move();
    
    unsigned int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 4; 
    
    TranspositionTable global_tt;
    std::atomic<bool> time_out{false};
    uint64_t root_hash = compute_zobrist(board, player);
    
    Move best_move = legal_moves[0];
    int last_completed_depth = 0;
    std::vector<std::pair<Move, int>> last_completed_evals;

    auto worker_thread = [&](int thread_id) {
        Stats local_stats;
        MoveList moves = legal_moves;
        std::mt19937 rng(12345 + thread_id); 
        char next_player = (player == 'B') ? 'W' : 'B';
        
        try {
            int depth = 1;
            while (!time_out.load(std::memory_order_relaxed)) {
                if (thread_id > 0) std::shuffle(moves.begin(), moves.end(), rng);
                
                std::vector<std::pair<Move, int>> current_scores;
                int alpha = -1e9;
                int beta = 1e9;
                int max_eval = -1e9;
                Move current_best_move = moves[0];
                
                for (const auto& move : moves) {
                    Board new_board = board;
                    uint64_t new_hash = root_hash;
                    apply_move(new_board, move, player, new_hash);
                    
                    auto [score, _] = minimax(new_board, depth - 1, alpha, beta, false, next_player, player, start_time, local_stats, global_tt, time_out, new_hash);
                    
                    if (thread_id == 0) current_scores.push_back({move, score});
                    
                    if (score > max_eval) {
                        max_eval = score;
                        current_best_move = move;
                    }
                    alpha = std::max(alpha, score);
                }
                
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
                
                if (max_eval >= 9000) break;
                depth++;
            }
        } catch (const TimeoutException&) {
            time_out.store(true, std::memory_order_relaxed);
        }
        return local_stats;
    };

    std::vector<std::future<Stats>> futures;
    for (unsigned int i = 1; i < num_cores; ++i) {
        futures.push_back(std::async(std::launch::async, worker_thread, i));
    }
    
    Stats global_stats = worker_thread(0);
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

// --- FILE & INPUT PARSING HELPER FUNCTIONS ---
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \n\r\t");
    if (std::string::npos == first) return "";
    size_t last = str.find_last_not_of(" \n\r\t");
    return str.substr(first, (last - first + 1));
}

Board parse_board_file(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(1);
    }
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

// --- MAIN I/O LOOP ---
int main(int argc, char* argv[]) {
    if (argc != 3) { std::cerr << "Invalid input\n"; return 1; }
    init_zobrist();

    std::string board_file = argv[1];
    char colour = argv[2][0];
    Board board = parse_board_file(board_file);
    
    uint64_t fake_hash = 0; // Temp hash as it computes fresh during `get_best_move` anyway
    MoveList moves = get_legal_moves(board, colour);
    if (moves.empty()) return 0; 

    Move best_move = get_best_move(board, colour, false); 
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

        best_move = get_best_move(board, colour, false);
        std::cout << best_move.to_string() << std::endl;
        apply_move(board, best_move, colour, fake_hash);
    }
    return 0;
}