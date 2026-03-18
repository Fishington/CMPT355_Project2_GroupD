#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <thread>
#include <random>
#include <array>

// --- SETTINGS ---
const double THINKING_TIME = 10.0;  // 6 seconds per turn

class TimeoutException : public std::exception {};

using Board = std::array<std::array<char, 8>, 8>;

struct Move {
    int start_r = 0, start_c = 0;
    int end_r = 0, end_c = 0;
    bool is_jump = false;

    // OPTIMIZATION: Kept string generation inside the struct. Only triggered when displaying.
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
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            board[r][c] = ((r + c) % 2 == 0) ? 'B' : 'W';
        }
    }
    return board;
}

void print_board(const Board& board) {
    // Prints Debug Board with row/column labels and '.' for empty spaces, Makes it easier to visualize compared to lab spec
    std::cout << "   A B C D E F G H\n";
    std::cout << "  -----------------\n";
    for (int r = 0; r < 8; ++r) {
        std::cout << " " << (8 - r) << " |";
        for (int c = 0; c < 8; ++c) {
            char piece = board[r][c] == 'O' ? '.' : board[r][c];
            std::cout << piece << (c < 7 ? " " : "");
        }
        std::cout << "| " << (8 - r) << "\n";
    }
    std::cout << "  -----------------\n";
    std::cout << "   A B C D E F G H\n\n";
}

std::pair<int, int> notation_to_coords(const std::string& notation) {
    // Converts from standard chess-like notation (e.g. "D5") to (row, col) indices
    char col_char = toupper(notation[0]);
    char row_char = notation[1];
    int col = col_char - 'A';
    int row = 8 - (row_char - '0');
    return {row, col};
}

std::string coords_to_notation(int r, int c) {
    // Converts from (row, col) indices back to standard notation (e.g. (3, 3) -> "D5")
    char col_char = (char)(c + 'A');
    char row_char = std::to_string(8 - r)[0];
    return std::string{col_char, row_char};
}

Board clone_board(const Board& board) {
    // Creates a deep copy of the board so we can simulate moves without affecting the original
    // OPTIMIZATION: C++ std::array copies inherently as a fast memory block
    return board;
}

void apply_move(Board& board, const Move& move, char current_player) {
    // Applies a move to the board. Handles both the opening placement and jump moves.
    // OPTIMIZATION: We utilize the Move struct to skip string split operations.
    
    // We use 'O' to represent empty spaces internally for easier processing, but we will print them as '.' for better visualization.
    if (!move.is_jump) { 
        board[move.start_r][move.start_c] = 'O';
        return;
    }

    int start_r = move.start_r, start_c = move.start_c;
    int end_r = move.end_r, end_c = move.end_c;
    
    int dr = (end_r == start_r) ? 0 : ((end_r > start_r) ? 1 : -1);
    int dc = (end_c == start_c) ? 0 : ((end_c > start_c) ? 1 : -1);
    
    int curr_r = start_r, curr_c = start_c;
    while (curr_r != end_r || curr_c != end_c) {
        board[curr_r][curr_c] = 'O';
        curr_r += dr;
        curr_c += dc;
        if (curr_r != end_r || curr_c != end_c) {
            board[curr_r][curr_c] = 'O';
            curr_r += dr;
            curr_c += dc;
        }
    }
    board[end_r][end_c] = current_player;
}

std::vector<Move> get_legal_moves(const Board& board, char player) {
    // Returns a list of legal moves for the given player in standard notation (e.g. "D5" for opening, "D5-F5" for jump)
    std::vector<Move> moves;
    moves.reserve(32); // Preallocate for speed
    
    int empty_count = 0;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            if (board[r][c] == 'O') empty_count++;
        }
    }
    
    if (empty_count == 0) {
        if (player == 'B') {
            moves.push_back({3, 3, 3, 3, false}); // D5
            moves.push_back({4, 4, 4, 4, false}); // E4
        }
        return moves;
    } else if (empty_count == 1) {
        if (player == 'W') {
            if (board[3][4] == 'W') moves.push_back({3, 4, 3, 4, false}); // E5
            if (board[4][3] == 'W') moves.push_back({4, 3, 4, 3, false}); // D4
        }
        return moves;
    }

    char opp_color = (player == 'B') ? 'W' : 'B';
    // We check every piece of the current player and see if it can jump in any of the 4 directions.
    const int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    // We will generate all possible jump moves. For each piece, we can potentially have multiple jumps in a row, so we keep jumping in the same direction until we can't anymore.
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            if (board[r][c] == player) {
                // 
                for (const auto& dir : directions) {
                    int dr = dir[0], dc = dir[1];
                    int curr_r = r, curr_c = c;
                    while (true) {
                        int mid_r = curr_r + dr, mid_c = curr_c + dc;
                        int dest_r = curr_r + 2 * dr, dest_c = curr_c + 2 * dc;
                        // We check if the destination is on the board, if there's an opponent piece to jump over, and if the landing spot is empty. If all conditions are met, it's a legal move.
                        if (dest_r >= 0 && dest_r < 8 && dest_c >= 0 && dest_c < 8) {
                            if (board[mid_r][mid_c] == opp_color && board[dest_r][dest_c] == 'O') {
                                moves.push_back({r, c, dest_r, dest_c, true});
                                curr_r = dest_r;
                                curr_c = dest_c;
                            } else break;
                        } else break;
                    }
                }
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
    // Check for timeout at the start of each minimax call. 
    // OPTIMIZATION: Only hit the clock every 1024 operations to avoid bogging down the OS.
    if ((stats.nodes_evaluated & 1023) == 0) {
        // If we've exceeded our time limit, we raise a TimeoutException which will be caught in get_best_move.
        if (get_elapsed(start_time) > THINKING_TIME) {
            throw TimeoutException();
        }
    }
    
    stats.nodes_evaluated++;

    auto moves = get_legal_moves(board, current_player);
    
    // If we are at depth 0 (leaf node) or there are no legal moves, we evaluate the board and return the score.
    if (depth == 0 || moves.empty()) {
        return {evaluate_board(board, original_player), Move()};
    }

    Move best_move = moves[0];
    char next_player = (current_player == 'B') ? 'W' : 'B';

    if (is_maximizing) {
        // We want to maximize the score for the original player, so we look for the highest score among the child nodes.
        int max_eval = -1e9;
        for (const auto& move : moves) {
            // We create a new board for each child node so we can simulate the move without affecting the current board state.
            Board new_board = clone_board(board);
            
            apply_move(new_board, move, current_player);
            
            // We recursively call minimax for the child node, flipping the is_maximizing flag and switching the current player.
            auto[eval_score, _] = minimax(new_board, depth - 1, alpha, beta, false, next_player, original_player, start_time, stats);
            
            // If the score from this child node is better than our current best score, we update max_eval and best_move.
            if (eval_score > max_eval) {
                max_eval = eval_score;
                best_move = move;
            }
            
            // Alpha-beta pruning: we update alpha and check if we can prune the remaining branches.
            alpha = std::max(alpha, eval_score);
            if (beta <= alpha) {
                stats.cutoffs++; // A-B Pruning
                break;
            }
        }
        return {max_eval, best_move};
    } else {
        // Same logic as above but we want to minimize the score for the opponent, so we look for the lowest score among the child nodes.
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
                stats.cutoffs++; // A-B Pruning
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

Move get_best_move(const Board& board, char player, bool debug_thinking = true) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto legal_moves = get_legal_moves(board, player);
    if (legal_moves.empty()) return Move();
    
    Move best_move = legal_moves[0];
    int depth = 1;
    
    std::vector<std::pair<Move, int>> last_completed_evals;
    int last_completed_depth = 0;
    
    Stats stats;
    
    try {
        while (true) {
            // --- TOP LEVEL LOOP (For Visualization) ---
            // We manually do the top layer of Minimax here so we can record 
            // the score of every single possible move at this depth.
            
            std::vector<std::pair<Move, int>> move_scores;
            int alpha = -1e9;
            int beta = 1e9;
            int max_eval = -1e9;
            Move current_best_move = legal_moves[0];
            char next_player = (player == 'B') ? 'W' : 'B';
            
            for (const auto& move : legal_moves) {
                if (get_elapsed(start_time) > THINKING_TIME) {
                    throw TimeoutException();
                }
                    
                Board new_board = clone_board(board);
                apply_move(new_board, move, player);
                
                // Pass into standard minimax for the rest of the depth
                auto [score, _] = minimax(new_board, depth - 1, alpha, beta, false, next_player, player, start_time, stats);
                
                move_scores.push_back({move, score});
                
                if (score > max_eval) {
                    max_eval = score;
                    current_best_move = move;
                }
                
                alpha = std::max(alpha, score);
            }
                
            // If we successfully evaluated all moves without timing out, save them
            best_move = current_best_move;
            last_completed_evals = move_scores;
            last_completed_depth = depth;
            
            if (max_eval >= 10000) break; // Found a forced win, stop thinking
            depth++;
        }
    } catch (const TimeoutException&) {
        // Time ran out, we will just print the last fully completed depth
    }
    
    double time_taken = get_elapsed(start_time);
        
    if (debug_thinking && !last_completed_evals.empty()) {
        std::cout << "\n[THOUGHT PROCESS (Depth " << last_completed_depth << ")]\n";
        std::cout << "   [Heuristic: Net Mobility (+ is good for Black)]\n";
        std::cout << "   Nodes Evaluated: " << format_with_commas(stats.nodes_evaluated) 
                  << " | Alpha-Beta Cutoffs: " << format_with_commas(stats.cutoffs) << "\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "   Time Taken: " << time_taken << " seconds (Time Limit: " << THINKING_TIME << " seconds)\n";
        
        long long nps = (time_taken > 0) ? static_cast<long long>(stats.nodes_evaluated / time_taken) : 0;
        std::cout << "  Search Speed      : " << format_with_commas(nps) << " nodes/sec\n";
        
        // Sort the moves from Best to Worst
        std::sort(last_completed_evals.begin(), last_completed_evals.end(),[](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        
        for (const auto& eval : last_completed_evals) {
            const Move& m = eval.first;
            int s = eval.second;
            std::string score_str;
            
            // Format the 
            if (s >= 9000) score_str = "WINNING TRAP FOUND!";
            else if (s <= -9000) score_str = "AVOIDING LOSS!";
            else {
                char buf[64];
                snprintf(buf, sizeof(buf), "%+d move advantage", s);
                score_str = buf;
            }
            
            // Highlight the chosen move
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
            // move = get_best_move(board, current_player, false); // For testing AI vs AI, uncomment this and comment out the random move below
            
            std::uniform_int_distribution<> dist(0, moves.size() - 1);
            move = moves[dist(gen)]; // Random move for White
            
            std::this_thread::sleep_for(std::chrono::seconds(1)); // Pause so you can read the AI's thoughts before the board updates
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