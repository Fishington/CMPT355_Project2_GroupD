#!/usr/bin/env python3

import time
import random
import copy
import sys

# --- SETTINGS ---
THINKING_TIME = 10  # 10 seconds per turn

class TimeoutException(Exception):
    pass

# --- GAME LOGIC ---
def create_initial_board():
    # [8][8] board with 'B' for Black pieces, 'W' for White pieces, and 'O' for empty spaces.
    return [['B' if (r+c)%2==0 else 'W' for c in range(8)] for r in range(8)]

def print_board(board):
    # Prints Debug Board with row/column labels and '.' for empty spaces, Makes it easier to visualize compared to lab spec
    print("   A B C D E F G H")
    print("  -----------------")
    for r in range(8):
        row_str = " ".join(board[r]).replace('O', 'O')
        print(f"{8-r} |{row_str}| {8-r}")
    print("  -----------------")
    print("   A B C D E F G H\n")

def notation_to_coords(notation):
    # Converts from standard chess-like notation (e.g. "D5") to (row, col) indices
    col_char = notation[0].upper()
    row_char = notation[1]
    col = ord(col_char) - ord('A')
    row = 8 - int(row_char)
    return row, col

def coords_to_notation(r, c):
    # Converts from (row, col) indices back to standard notation (e.g. (3, 3) -> "D5")
    col_char = chr(c + ord('A'))
    row_char = str(8 - r)
    return f"{col_char}{row_char}"

def clone_board(board):
    # Creates a deep copy of the board so we can simulate moves without affecting the original
    return [row[:] for row in board]

def apply_move(board, move_str, current_player):
    # Applies a move to the board. Handles both the opening placement and jump moves.
    
    # We use 'O' to represent empty spaces internally for easier processing, but we will print them as '.' for better visualization.
    if '-' not in move_str: 
        r, c = notation_to_coords(move_str)
        board[r][c] = 'O'
        return

    parts = move_str.split('-')
    start_r, start_c = notation_to_coords(parts[0])
    end_r, end_c = notation_to_coords(parts[1])
    
    dr = 0 if end_r == start_r else (1 if end_r > start_r else -1)
    dc = 0 if end_c == start_c else (1 if end_c > start_c else -1)
    
    curr_r, curr_c = start_r, start_c
    while (curr_r, curr_c) != (end_r, end_c):
        board[curr_r][curr_c] = 'O'
        curr_r += dr
        curr_c += dc
        if (curr_r, curr_c) != (end_r, end_c):
            board[curr_r][curr_c] = 'O'
            curr_r += dr
            curr_c += dc
            
    board[end_r][end_c] = current_player

def get_legal_moves(board, player):
    # Returns a list of legal moves for the given player in standard notation (e.g. "D5" for opening, "D5-F5" for jump)
    moves =[]
    empty_count = sum(row.count('O') for row in board)
    
    if empty_count == 0:
        if player == 'B': return ["D5", "E4"]
    elif empty_count == 1:
        if player == 'W':
            valid_w = []
            if board[3][4] == 'W': valid_w.append("E5")
            if board[4][3] == 'W': valid_w.append("D4")
            return valid_w

    opp_color = 'W' if player == 'B' else 'B'
    # We check every piece of the current player and see if it can jump in any of the 4 directions.
    directions =[(-1, 0), (1, 0), (0, -1), (0, 1)] 
    
    # We will generate all possible jump moves. For each piece, we can potentially have multiple jumps in a row, so we keep jumping in the same direction until we can't anymore.
    for r in range(8):
        for c in range(8):
            if board[r][c] == player:
                start_notation = coords_to_notation(r, c)
                
                # 
                for dr, dc in directions:
                    curr_r, curr_c = r, c
                    while True:
                        mid_r, mid_c = curr_r + dr, curr_c + dc
                        dest_r, dest_c = curr_r + 2*dr, curr_c + 2*dc
                        # We check if the destination is on the board, if there's an opponent piece to jump over, and if the landing spot is empty. If all conditions are met, it's a legal move.
                        if 0 <= dest_r < 8 and 0 <= dest_c < 8:
                            if board[mid_r][mid_c] == opp_color and board[dest_r][dest_c] == 'O':
                                
                                end_notation = coords_to_notation(dest_r, dest_c)
                                moves.append(f"{start_notation}-{end_notation}")
                                curr_r, curr_c = dest_r, dest_c
                            else: break
                        else: break
    return moves

# --- MINIMAX  ---
def evaluate_board(board, player):
    opp = 'W' if player == 'B' else 'B'
    my_moves = len(get_legal_moves(board, player))
    opp_moves = len(get_legal_moves(board, opp))
    
    if opp_moves == 0: return 10000 
    if my_moves == 0: return -10000
    return my_moves - opp_moves

def minimax(board, depth, alpha, beta, is_maximizing, current_player, original_player, start_time, stats):
    # Minimax with alpha-beta pruning. Returns (score, best_move) where best_move is only meaningful at the top level of the search.
    
    
    # Check for timeout at the start of each minimax call. 
    # If we've exceeded our time limit, we raise a TimeoutException which will be caught in get_best_move.
    if time.time() - start_time > THINKING_TIME:
        raise TimeoutException()
    
    stats['nodes_evaluated'] += 1

    moves = get_legal_moves(board, current_player)
    
    # If we are at depth 0 (leaf node) or there are no legal moves, we evaluate the board and return the score.
    if depth == 0 or not moves:
        return evaluate_board(board, original_player), None

    best_move = moves[0]
    next_player = 'W' if current_player == 'B' else 'B'

    if is_maximizing:
        # We want to maximize the score for the original player, so we look for the highest score among the child nodes.
        max_eval = -float('inf')
        for move in moves:
            # We create a new board for each child node so we can simulate the move without affecting the current board state.
            new_board = clone_board(board)
            
            apply_move(new_board, move, current_player)
            
            # We recursively call minimax for the child node, flipping the is_maximizing flag and switching the current player.
            eval_score, _ = minimax(new_board, depth - 1, alpha, beta, False, next_player, original_player, start_time, stats)
            
            # If the score from this child node is better than our current best score, we update max_eval and best_move.
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            
            # Alpha-beta pruning: we update alpha and check if we can prune the remaining branches.
            alpha = max(alpha, eval_score)
            if beta <= alpha: 
                stats['cutoffs'] += 1 # A-B Pruning
                break
        return max_eval, best_move
    else:
        # Same logic as above but we want to minimize the score for the opponent, so we look for the lowest score among the child nodes.
        min_eval = float('inf')
        for move in moves:
            new_board = clone_board(board)
            apply_move(new_board, move, current_player)
            eval_score, _ = minimax(new_board, depth - 1, alpha, beta, True, next_player, original_player, start_time, stats)
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha: 
                stats['cutoffs'] += 1 # A-B Pruning
                break
        return min_eval, best_move

def get_best_move(board, player, debug_thinking=True):
    start_time = time.time()
    legal_moves = get_legal_moves(board, player)
    if not legal_moves: return None
    
    best_move = legal_moves[0] 
    depth = 1
    
    last_completed_evals =[]
    last_completed_depth = 0
    
    stats = {
        'nodes_evaluated': 0,
        'cutoffs': 0
    }
    
    try:
        while True:
            # --- TOP LEVEL LOOP (For Visualization) ---
            # We manually do the top layer of Minimax here so we can record 
            # the score of every single possible move at this depth.
            
            move_scores =[]
            alpha = -float('inf')
            beta = float('inf')
            max_eval = -float('inf')
            current_best_move = legal_moves[0]
            next_player = 'W' if player == 'B' else 'B'
            
            for move in legal_moves:
                if time.time() - start_time > THINKING_TIME:
                    raise TimeoutException()
                    
                new_board = clone_board(board)
                apply_move(new_board, move, player)
                
                # Pass into standard minimax for the rest of the depth
                score, _ = minimax(new_board, depth - 1, alpha, beta, False, next_player, player, start_time, stats)
                
                move_scores.append((move, score))
                
                if score > max_eval:
                    max_eval = score
                    current_best_move = move
                
                alpha = max(alpha, score)
                
            # If we successfully evaluated all moves without timing out, save them
            best_move = current_best_move
            last_completed_evals = move_scores
            last_completed_depth = depth
            
            if max_eval == 10000: break # Found a forced win, stop thinking
            depth += 1
            
    except TimeoutException:
        pass # Time ran out, we will just print the last fully completed depth
    
    
    time_taken = time.time() - start_time
        
    if debug_thinking and last_completed_evals:
        print(f"\n[THOUGHT PROCESS (Depth {last_completed_depth})]")
        print("   [Heuristic: Net Mobility (+ is good for Black)]")
        print(f"   Nodes Evaluated: {stats['nodes_evaluated']} | Alpha-Beta Cutoffs: {stats['cutoffs']}")
        print(f"   Time Taken: {time.time() - start_time:.2f} seconds (Time Limit: {THINKING_TIME} seconds)")
        
        nps = int(stats['nodes_evaluated'] / time_taken) if time_taken > 0 else 0
        print(f"  Search Speed      : {nps:,} nodes/sec")
        
        # Sort the moves from Best to Worst
        sorted_evals = sorted(last_completed_evals, key=lambda x: x[1], reverse=True)
        
        for m, s in sorted_evals:
            # Format the 
            if s >= 9000: score_str = "WINNING TRAP FOUND!"
            elif s <= -9000: score_str = "AVOIDING LOSS!"
            else: score_str = f"{s:+d} move advantage"
            
            # Highlight the chosen move
            marker = ">> " if m == best_move else "   "
            print(f"    {marker}Move: {m:8} | Score: {score_str}")
        print("")
            
    return best_move

# --- MOCK GAME LOOP ---
def play_mock_game():
    print("=== KONANE MOCK GAME ===")
    print("Black (B): Minimax AI")
    print("White (W): Random Player\n")
    
    USE_CUSTOM_BOARD = True  # Set to True to use a custom board state for testing, False for standard game start
    
    if USE_CUSTOM_BOARD:
        # Define your custom board state here. 
        # 'B' = Black, 'W' = White, 'O' = Empty
        # This example is the board exactly after Black's first turn (D5 removed)
        custom_board_str =[
            "BWBWBWBW", # Rank 8 (Index 0)
            "WBWBWBWB", # Rank 7
            "BWBWBWBW", # Rank 6
            "WBWBOOWB", # Rank 5
            "BWBOBWBW", # Rank 4
            "WBWOWBWB", # Rank 3
            "BWBWBWBW", # Rank 2
            "WBWBWBWB"  # Rank 1 (Index 7)
        ]
        # Convert the list of strings into a 2D list of characters
        board =[list(row) for row in custom_board_str]
        
        # NOTE: If you set up a mid-game board, make sure you set whose turn it is!
        # Since Black already moved in this custom board, White goes next.
        current_player = 'B' 
        turn_number = 4
        
    else:
        # Standard Game Start
        board = create_initial_board()
        current_player = 'B'
        turn_number = 1
    # -----------------------------
    
    while True:
        print(f"--- Turn {turn_number} ---")
        print_board(board)
        
        moves = get_legal_moves(board, current_player)
        if not moves:
            winner = 'White' if current_player == 'B' else 'Black'
            print(f"*** GAME OVER! {current_player} has no legal moves. ***")
            print(f"*** {winner} WINS! ***")
            break
            
        if current_player == 'B':
            print("Black (AI) is thinking...")
            move = get_best_move(board, current_player, debug_thinking=True)
        else:
            print("White (Random) is picking...")
            # move = get_best_move(board, current_player, debug_thinking=False) # For testing AI vs AI, uncomment this and comment out the random move below
            move = random.choice(get_legal_moves(board, current_player)) # Random move for White
            
            time.sleep(1.0) # Pause so you can read the AI's thoughts before the board updates
            
        print(f"-> Player {current_player} plays: {move}\n")
        
        apply_move(board, move, current_player)
        current_player = 'W' if current_player == 'B' else 'B'
        turn_number += 1

'''s
def main():
    if len(sys.argv) != 3:
        print("Invalid input")
        return

    board_str = sys.argv[1]
    colour = sys.argv[2]

    # Convert the board string into a 2D list
    with open(board_str, 'r') as f:
        board_str = f.read().replace('\n', '')
    board = [list(board_str[i:i+8]) for i in range(0, 64, 8)]

    while True: # Need to keep playing until the end
        # Find the best move
        best_move = get_best_move(board, colour, debug_thinking=False)

        # Print to stdout
        print(best_move)

        # Flush stdout
        sys.stdout.flush()
    '''
def main():
    if len(sys.argv) != 3:
        print("Invalid input", file=sys.stderr)
        return

    board_file = sys.argv[1]
    colour = sys.argv[2]

    # Convert the board string into a 2D list
    with open(board_file, 'r') as f:
        board_str = f.read().replace('\n', '')
    board = [list(board_str[i:i+8]) for i in range(0, 64, 8)]

    if colour == 'B':
        best_move = get_best_move(board, colour, debug_thinking=False)
        if best_move is None:
            return
        print(best_move)
        sys.stdout.flush()
        apply_move(board, best_move, colour)

    while True:
        opponent_move = sys.stdin.readline()
        if not opponent_move:
            continue

        opponent_move = opponent_move.strip()

        apply_move(board, opponent_move, 'W' if colour == 'B' else 'B')

        best_move = get_best_move(board, colour, debug_thinking=False)
        if best_move is None:
            break

        print(best_move)
        sys.stdout.flush()

        apply_move(board, best_move, colour)



if __name__ == "__main__":
    # play_mock_game()
    main()

