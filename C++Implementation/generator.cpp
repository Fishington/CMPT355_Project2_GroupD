#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <atomic>
#include <future>
#include <chrono>
#include <iomanip>

// --- BITBOARD DEFINITIONS ---
struct Board { uint64_t b = 0; uint64_t w = 0; };

struct Move {
    int8_t start_r = 0, start_c = 0;
    int8_t end_r = 0, end_c = 0;
    bool is_jump = false;
};

struct MoveList {
    Move moves[128];
    int count = 0;
    inline void push_back(const Move& m) { moves[count++] = m; }
    inline bool empty() const { return count == 0; }
    inline size_t size() const { return count; }
    inline Move& operator[](size_t i) { return moves[i]; }
};

// --- GENOME (The AI's Brain) ---
struct Genome {
    int pst_quadrant[16]; // The 16 unique structural squares of an 8x8 board
    int mobility_weight;  
    int position_weight;  
    
    int wins = 0;
    int losses = 0;
    int draws = 0;

    void get_full_pst(int full_pst[64]) const {
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                // Mirror coordinates to the top-left quadrant (0-3, 0-3)
                int sym_r = (r < 4) ? r : 7 - r;
                int sym_c = (c < 4) ? c : 7 - c;
                full_pst[r * 8 + c] = pst_quadrant[sym_r * 4 + sym_c];
            }
        }
    }
};

// --- GAME LOGIC (Stripped down for speed) ---
void apply_move(Board& board, const Move& move, char current_player) {
    uint64_t& my_pieces = (current_player == 'B') ? board.b : board.w;
    uint64_t& opp_pieces = (current_player == 'B') ? board.w : board.b;
    int start_idx = move.start_r * 8 + move.start_c;

    if (!move.is_jump) { my_pieces &= ~(1ULL << start_idx); return; }

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
                    if ((opp_pieces & (1ULL << (mid_r * 8 + mid_c))) && (empty & (1ULL << (dest_r * 8 + dest_c)))) {
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
        if (player == 'W') {
            int count = 0;
            if (board.w & (1ULL << (3 * 8 + 4))) count++;
            if (board.w & (1ULL << (4 * 8 + 3))) count++;
            return count;
        }
        return 0;
    }

    uint64_t mine = (player == 'B') ? board.b : board.w;
    uint64_t opp = (player == 'B') ? board.w : board.b;
    uint64_t empty = ~(board.b | board.w);
    int count = 0;
    
    const uint64_t notA = 0xFEFEFEFEFEFEFEFEULL; 
    const uint64_t notH = 0x7F7F7F7F7F7F7F7FULL; 

    uint64_t movers = mine;
    while (movers) { uint64_t s1 = (movers << 1) & opp & notA; uint64_t s2 = (s1 << 1) & empty & notA; if (!s2) break; count += __builtin_popcountll(s2); movers = s2; }
    movers = mine;
    while (movers) { uint64_t s1 = (movers >> 1) & opp & notH; uint64_t s2 = (s1 >> 1) & empty & notH; if (!s2) break; count += __builtin_popcountll(s2); movers = s2; }
    movers = mine;
    while (movers) { uint64_t s1 = (movers << 8) & opp; uint64_t s2 = (s1 << 8) & empty; if (!s2) break; count += __builtin_popcountll(s2); movers = s2; }
    movers = mine;
    while (movers) { uint64_t s1 = (movers >> 8) & opp; uint64_t s2 = (s1 >> 8) & empty; if (!s2) break; count += __builtin_popcountll(s2); movers = s2; }
    return count;
}

// --- GENETIC EVALUATOR ---
int evaluate_board_genome(const Board& board, char player, const Genome& genome, const int full_pst[64]) {
    char opp = (player == 'B') ? 'W' : 'B';
    int my_moves = count_legal_moves(board, player);
    int opp_moves = count_legal_moves(board, opp);
    
    if (opp_moves == 0) return 10000; 
    if (my_moves == 0) return -10000;
    
    int mobility_score = my_moves - opp_moves;
    
    int positional_score = 0;
    uint64_t my_pieces = (player == 'B') ? board.b : board.w;
    uint64_t opp_pieces = (player == 'B') ? board.w : board.b;
    
    while (my_pieces) {
        int idx = __builtin_ctzll(my_pieces);
        positional_score += full_pst[idx];
        my_pieces &= my_pieces - 1;
    }
    while (opp_pieces) {
        int idx = __builtin_ctzll(opp_pieces);
        positional_score -= full_pst[idx];
        opp_pieces &= opp_pieces - 1;
    }

    return (mobility_score * genome.mobility_weight) + (positional_score * genome.position_weight);
}

// Strict Fixed-Depth Minimax without time limits to ensure fair tests
std::pair<int, Move> minimax_genome(Board board, int depth, int alpha, int beta, bool is_maximizing, char current_player, char original_player, const Genome& genome, const int full_pst[64]) {
    MoveList moves = get_legal_moves(board, current_player);
    if (depth == 0 || moves.empty()) {
        int eval = evaluate_board_genome(board, original_player, genome, full_pst);
        if (moves.empty()) eval = (current_player == original_player) ? -10000 - depth : 10000 + depth;
        return {eval, Move()};
    }

    Move best_move = moves[0];
    char next_player = (current_player == 'B') ? 'W' : 'B';

    if (is_maximizing) {
        int max_eval = -1e9;
        for (int i = 0; i < moves.size(); ++i) {
            Board new_board = board;
            apply_move(new_board, moves[i], current_player);
            auto[eval_score, _] = minimax_genome(new_board, depth - 1, alpha, beta, false, next_player, original_player, genome, full_pst);
            if (eval_score > max_eval) { max_eval = eval_score; best_move = moves[i]; }
            alpha = std::max(alpha, eval_score);
            if (beta <= alpha) break;
        }
        return {max_eval, best_move};
    } else {
        int min_eval = 1e9;
        for (int i = 0; i < moves.size(); ++i) {
            Board new_board = board;
            apply_move(new_board, moves[i], current_player);
            auto[eval_score, _] = minimax_genome(new_board, depth - 1, alpha, beta, true, next_player, original_player, genome, full_pst);
            if (eval_score < min_eval) { min_eval = eval_score; best_move = moves[i]; }
            beta = std::min(beta, eval_score);
            if (beta <= alpha) break;
        }
        return {min_eval, best_move};
    }
}

// Play a full game between two genomes
int play_match(const Genome& g1, const Genome& g2, int match_id) {
    Board board;
    // Standard starting Konane board
    board.b = 0xAA55AA55AA55AA55ULL;
    board.w = 0x55AA55AA55AA55AAULL;

    int pst1[64]; g1.get_full_pst(pst1);
    int pst2[64]; g2.get_full_pst(pst2);

    std::mt19937 rng(12345 + match_id); // Ensure deterministic random openings
    
    char current_player = 'B';
    int move_count = 0;
    
    // Play until game over (or 100 moves to prevent infinite loops)
    while (move_count < 100) {
        MoveList moves = get_legal_moves(board, current_player);
        if (moves.empty()) {
            return (current_player == 'B') ? -1 : 1; // 1 means g1 (Black) won, -1 means g2 (White) won
        }

        Move chosen_move;
        
        // Force the first 2 moves to be completely random to explore different game trees
        if (move_count < 2) {
            std::uniform_int_distribution<int> dist(0, moves.size() - 1);
            chosen_move = moves[dist(rng)];
        } else {
            // Depth 4 is fast enough to run millions of games, but deep enough to show intelligence
            if (current_player == 'B') {
                chosen_move = minimax_genome(board, 7, -1e9, 1e9, true, 'B', 'B', g1, pst1).second;
            } else {
                chosen_move = minimax_genome(board, 7, -1e9, 1e9, true, 'W', 'W', g2, pst2).second;
            }
        }

        apply_move(board, chosen_move, current_player);
        current_player = (current_player == 'B') ? 'W' : 'B';
        move_count++;
    }
    return 0; // Draw by move limit
}

// --- EVOLUTIONARY ALGORITHM ---
int main() {
    const int POPULATION_SIZE = 1000;
    const int GENERATIONS = 2500;
    const int NUM_CORES = std::thread::hardware_concurrency(); // Will detect all 80 cores
    
    std::cout << "Starting 80-Core Konane Genetic Trainer...\n";
    std::cout << "Detected Cores: " << NUM_CORES << "\n\n";

    std::vector<Genome> population(POPULATION_SIZE);
    std::mt19937 rng(std::random_device{}());
    
    // Initialize random population
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        std::uniform_int_distribution<int> pst_dist(-50, 100);
        std::uniform_int_distribution<int> weight_dist(10, 100);
        for (int j = 0; j < 16; ++j) population[i].pst_quadrant[j] = pst_dist(rng);
        population[i].mobility_weight = weight_dist(rng);
        population[i].position_weight = weight_dist(rng);
    }

    for (int gen = 1; gen <= GENERATIONS; ++gen) {
        auto start_gen = std::chrono::high_resolution_clock::now();
        
        // Reset stats
        for (auto& g : population) { g.wins = 0; g.losses = 0; g.draws = 0; }

        // Generate Matchups (Round Robin: everyone plays 5 random opponents)
        std::vector<std::pair<int, int>> matches;
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            for (int j = 0; j < 5; ++j) {
                int opp = rng() % POPULATION_SIZE;
                if (i != opp) matches.push_back({i, opp});
            }
        }

        // --- 80-CORE MULTITHREADED MATCH EXECUTION ---
        std::atomic<int> match_idx{0};
        auto worker =[&]() {
            while (true) {
                int idx = match_idx.fetch_add(1);
                if (idx >= matches.size()) break;
                
                int p1 = matches[idx].first;
                int p2 = matches[idx].second;
                
                // P1 plays as Black, P2 as White
                int result = play_match(population[p1], population[p2], idx);
                
                if (result == 1) {
                    population[p1].wins++; population[p2].losses++;
                } else if (result == -1) {
                    population[p2].wins++; population[p1].losses++;
                } else {
                    population[p1].draws++; population[p2].draws++;
                }
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < NUM_CORES; ++i) threads.emplace_back(worker);
        for (auto& t : threads) t.join();

        // Sort by wins
        std::sort(population.begin(), population.end(),[](const Genome& a, const Genome& b) {
            return a.wins > b.wins;
        });

        // Breed the top 20% to replace the bottom 80%
        int elite_count = POPULATION_SIZE / 5;
        std::uniform_int_distribution<int> elite_dist(0, elite_count - 1);
        std::uniform_int_distribution<int> mutate_dist(-10, 10); // Small mutation step
        std::uniform_int_distribution<int> mutate_chance(0, 100);
        
        for (int i = elite_count; i < POPULATION_SIZE; ++i) {
            Genome parentA = population[elite_dist(rng)];
            Genome parentB = population[elite_dist(rng)];
            
            Genome child;
            // Crossover & Mutate PST
            for (int j = 0; j < 16; ++j) {
                child.pst_quadrant[j] = (rng() % 2 == 0) ? parentA.pst_quadrant[j] : parentB.pst_quadrant[j];
                if (mutate_chance(rng) < 15) child.pst_quadrant[j] += mutate_dist(rng); // 15% mutation chance
            }
            // Crossover weights
            child.mobility_weight = (rng() % 2 == 0) ? parentA.mobility_weight : parentB.mobility_weight;
            child.position_weight = (rng() % 2 == 0) ? parentA.position_weight : parentB.position_weight;
            
            if (mutate_chance(rng) < 15) child.mobility_weight = std::max(1, child.mobility_weight + mutate_dist(rng));
            if (mutate_chance(rng) < 15) child.position_weight = std::max(1, child.position_weight + mutate_dist(rng));
            
            population[i] = child;
        }

        auto end_gen = std::chrono::high_resolution_clock::now();
        double gen_time = std::chrono::duration<double>(end_gen - start_gen).count();

        // Print Generation Results
        std::cout << "Generation " << gen << " Completed in " << std::fixed << std::setprecision(1) << gen_time << "s\n";
        std::cout << "   Best AI Stats : " << population[0].wins << " Wins, " << population[0].losses << " Losses\n";
        std::cout << "   Best Weights  : Mobility(" << population[0].mobility_weight << "), Position(" << population[0].position_weight << ")\n";
        
        if (gen % 10 == 0 || gen == GENERATIONS) {
            int best_pst[64];
            population[0].get_full_pst(best_pst);
            std::cout << "\n--- CURRENT BEST PIECE SQUARE TABLE (Copy this into main.cpp) ---\nconstexpr int PIECE_SQUARE_TABLE[64] = {\n    ";
            for (int i = 0; i < 64; ++i) {
                std::cout << std::setw(3) << best_pst[i] << ((i != 63) ? ", " : "");
                if ((i + 1) % 8 == 0 && i != 63) std::cout << "\n    ";
            }
            std::cout << "\n};\n-----------------------------------------------------------------\n\n";
        }
    }

    return 0;
}