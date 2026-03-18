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
#include <cstdint> // Added for uint64_t

// --- SETTINGS ---
const double THINKING_TIME = 10.0;  // 6 seconds per turn

class TimeoutException : public std::exception {};

// OPTIMIZATION: The Board is now just two 64-bit integers instead of an 8x8 array.
// Each bit (0 to 63) represents a square. Bit 0 is A8, Bit 63 is H1.
struct Board {
    uint64_t b = 0; // Black pieces bitboard
    uint64_t w = 0; // White pieces bitboard
};

struct Move {
    int start_r = 0, start_c = 0;
    int end_r = 0, end_c = 0;
    bool is_jump = false;

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
};

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

double get_elapsed(TimePoint start) {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

// --- GAME LOGIC ---
Board create_initial_board() {
    Board board;
    // Magic Hex numbers representing the checkered start state of Konane exactly.
    board.b = 0xAA55AA55AA55AA55ULL;
    board.w = 0x55AA55AA55AA55AAULL;
    return board;
}

void print_board(const Board& board) {
    std::cout << "   A B C D E F G H\n";
    std::cout << "  -----------------\n";
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
    std::cout << "  -----------------\n";
    std::cout << "   A B C D E F G H\n\n";
}

std::pair<int, int> notation_to_coords(const std::string& notation) {
    char col_char = toupper(notation[0]);
    char row_char = notation[1];
    int col = col_char - 'A';
    int row = 8 - (row_char - '0');
    return {row, col};
}

std::string coords_to_notation(int r, int c) {
    char col_char = (char)(c + 'A');
    char row_char = std::to_string(8 - r)[0];
    return std::string{col_char, row_char};
}

Board clone_board(const Board& board) {
    // A bitboard copy takes practically 0 clock cycles (just moving 16 bytes)
    return board;
}

void apply_move(Board& board, const Move& move, char current_player) {
    uint64_t& my_pieces = (current_player == 'B') ? board.b : board.w;
    uint64_t& opp_pieces = (current_player == 'B') ? board.w : board.b;
    
    int start_idx = move.start_r * 8 + move.start_c;
    
    // First turn piece removals
    if (!move.is_jump) { 
        my_pieces &= ~(1ULL << start_idx); // Turn off the bit (remove piece)
        return;
    }

    // Applying Jump Masks
    int dr = (move.end_r == move.start_r) ? 0 : ((move.end_r > move.start_r) ? 1 : -1);
    int dc = (move.end_c == move.start_c) ? 0 : ((move.end_c > move.start_c) ? 1 : -1);
    
    int end_idx = move.end_r * 8 + move.end_c;
    int delta_idx = dr * 8 + dc; // Flattened 1D directional jump
    
    int curr_idx = start_idx;
    my_pieces &= ~(1ULL << curr_idx); // Remove from start position
    
    while (curr_idx != end_idx) {
        curr_idx += delta_idx;
        if (curr_idx != end_idx) {
            opp_pieces &= ~(1ULL << curr_idx); // Remove jumped piece
            curr_idx += delta_idx;
        }
    }
    
    my_pieces |= (1ULL << end_idx); // Add to final position
}

std::vector<Move> get_legal_moves(const Board& board, char player) {
    std::vector<Move> moves;
    moves.reserve(32); 
    
    // POPCOUNT: Instantly counts how many pieces exist. 64 minus pieces = empty spaces.
    int empty_count = 64 - __builtin_popcountll(board.b | board.w);
    
    if (empty_count == 0) {
        if (player == 'B') {
            moves.push_back({3, 3, 3, 3, false}); // D5
            moves.push_back({4, 4, 4, 4, false}); // E4
        }
        return moves;
    } else if (empty_count == 1) {
        if (player == 'W') {
            if (board.w & (1ULL << (3 * 8 + 4))) moves.push_back({3, 4, 3, 4, false}); // E5
            if (board.w & (1ULL << (4 * 8 + 3))) moves.push_back({4, 3, 4, 3, false}); // D4
        }
        return moves;
    }

    uint64_t my_pieces = (player == 'B') ? board.b : board.w;
    uint64_t opp_pieces = (player == 'B') ? board.w : board.b;
    uint64_t empty = ~(board.b | board.w);
    
    const int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    uint64_t pieces = my_pieces;
    
    // CTZLL: "Count Trailing Zeros". Instantly skips to the next piece on the board!
    while (pieces) {
        int idx = __builtin_ctzll(pieces); 
        pieces &= pieces - 1; // Clear the lowest bit we just processed
        
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
                    
                    // Direct Bitmask lookup for Jump Validation
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

std::pair<int, Move> minimax(Board board, int depth, int alpha, int beta, bool is_maximizing, char current_player, char original_player, TimePoint start_time, Stats& stats) {
    if ((stats.nodes_evaluated & 1023) == 0) {
        if (get_elapsed(start_time) > THINKING_TIME) {
            throw TimeoutException();
        }
    }
    
    stats.nodes_evaluated++;

    auto moves = get_legal_moves(board, current_player);
    
    if (depth == 0 || moves.empty()) {
        return {evaluate_board(board, original_player), Move()};
    }

    Move best_move = moves[0];
    char next_player = (current_player == 'B') ? 'W' : 'B';

    if (is_maximizing) {
        int max_eval = -1e9;
        for (const auto& move : moves) {
            Board new_board = clone_board(board);
            apply_move(new_board, move, current_player);
            
            auto[eval_score, _] = minimax(new_board, depth - 1, alpha, beta, false, next_player, original_player, start_time, stats);
            
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
        return {max_eval, best_move};
    } else {
        int min_eval = 1e9;
        for (const auto& move : moves) {
            Board new_board = clone_board(board);
            apply_move(new_board, move, current_player);
            auto [eval_score, _] = minimax(new_board, depth - 1, alpha, beta, true, next_player, original_player, start_time, stats);
            
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

struct AsyncResult {
    Move move;
    int score;
    Stats stats;
};

Move get_best_move(const Board& board, char player, bool debug_thinking = true) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto legal_moves = get_legal_moves(board, player);
    if (legal_moves.empty()) return Move();
    
    Move best_move = legal_moves[0];
    int depth = 1;
    
    std::vector<std::pair<Move, int>> last_completed_evals;
    int last_completed_depth = 0;
    Stats global_stats; 
    
    try {
        while (true) {
            std::vector<std::future<AsyncResult>> futures;
            char next_player = (player == 'B') ? 'W' : 'B';
            
            // --- MULTITHREADED TOP LEVEL ---
            for (const auto& move : legal_moves) {
                futures.push_back(std::async(std::launch::async,[board, move, depth, next_player, player, start_time]() {
                    Stats local_stats;
                    Board new_board = clone_board(board);
                    apply_move(new_board, move, player);
                    
                    int alpha = -1e9;
                    int beta = 1e9;
                    
                    auto[score, _] = minimax(new_board, depth - 1, alpha, beta, false, next_player, player, start_time, local_stats);
                    
                    return AsyncResult{move, score, local_stats};
                }));
            }
            
            std::vector<std::pair<Move, int>> move_scores;
            int max_eval = -1e9;
            Move current_best_move = legal_moves[0];
            Stats depth_stats;
            
            for (auto& fut : futures) {
                AsyncResult result = fut.get(); 
                
                depth_stats.nodes_evaluated += result.stats.nodes_evaluated;
                depth_stats.cutoffs += result.stats.cutoffs;
                move_scores.push_back({result.move, result.score});
                
                if (result.score > max_eval) {
                    max_eval = result.score;
                    current_best_move = result.move;
                }
            }
            
            global_stats.nodes_evaluated += depth_stats.nodes_evaluated;
            global_stats.cutoffs += depth_stats.cutoffs;
            
            best_move = current_best_move;
            last_completed_evals = move_scores;
            last_completed_depth = depth;
            
            if (max_eval >= 10000) break; 
            depth++;
        }
    } catch (const TimeoutException&) {
    }
    
    double time_taken = get_elapsed(start_time);
        
    if (debug_thinking && !last_completed_evals.empty()) {
        std::cout << "\n[THOUGHT PROCESS (Depth " << last_completed_depth << ")]\n";
        std::cout << "   [Heuristic: Net Mobility (+ is good for Black)]\n";
        std::cout << "   Nodes Evaluated: " << format_with_commas(global_stats.nodes_evaluated) 
                  << " | Alpha-Beta Cutoffs: " << format_with_commas(global_stats.cutoffs) << "\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "   Time Taken: " << time_taken << " seconds (Time Limit: " << THINKING_TIME << " seconds)\n";
        
        long long nps = (time_taken > 0) ? static_cast<long long>(global_stats.nodes_evaluated / time_taken) : 0;
        std::cout << "  Search Speed      : " << format_with_commas(nps) << " nodes/sec\n";
        
        std::sort(last_completed_evals.begin(), last_completed_evals.end(),[](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        
        for (const auto& eval : last_completed_evals) {
            const Move& m = eval.first;
            int s = eval.second;
            std::string score_str;
            
            if (s >= 9000) score_str = "WINNING TRAP FOUND!";
            else if (s <= -9000) score_str = "AVOIDING LOSS!";
            else {
                char buf[64];
                snprintf(buf, sizeof(buf), "%+d move advantage", s);
                score_str = buf;
            }
            
            std::string marker = (m.to_string() == best_move.to_string()) ? ">> " : "   ";
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
    std::cout << "Black (B): Minimax AI\n";
    std::cout << "White (W): Random Player\n\n";
    
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
    play_mock_game();
    return 0;
}