from random import random
import sys
import time

THINKING_TIME = 9.5 
START_TIME = 0

def evaluate(board, my_color, opp_color):
    """
    The Heuristic Function.
    Konane is a game of mobility. The best state is one where you have 
    many available moves, and your opponent has very few.
    """
    my_moves = len(get_legal_moves(board, my_color))
    opp_moves = len(get_legal_moves(board, opp_color))
    
    # If opponent has no moves, we win! Give a massive score.
    if opp_moves == 0:
        return 10000
    # If we have no moves, we lose.
    if my_moves == 0:
        return -10000
        
    return my_moves - opp_moves

def minimax(board, depth, alpha, beta, maximizing_player, my_color, opp_color):
    """
    Minimax algorithm with Alpha-Beta pruning.
    Throws a TimeoutError if we are about to run out of time.
    """
    global START_TIME
    if time.time() - START_TIME > THINKING_TIME:
        raise TimeoutError("Out of time!")

    if depth == 0:
        return evaluate(board, my_color, opp_color)

    current_player = my_color if maximizing_player else opp_color
    moves = get_legal_moves(board, current_player)

    if not moves:
        # If the current player has no moves, they lose.
        return -10000 if maximizing_player else 10000

    if maximizing_player:
        max_eval = float('-inf')
        for move in moves:
            # Create a copy of the board to test the move
            new_board = [row[:] for row in board] 
            apply_move(new_board, move, current_player)
            
            ev = minimax(new_board, depth - 1, alpha, beta, False, my_color, opp_color)
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break # Alpha-Beta Pruning
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            new_board = [row[:] for row in board]
            apply_move(new_board, move, current_player)
            
            ev = minimax(new_board, depth - 1, alpha, beta, True, my_color, opp_color)
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break # Alpha-Beta Pruning
        return min_eval

def get_best_move(board, my_color):
    """
    Uses Iterative Deepening to find the best move within the time limit.
    """
    global START_TIME
    START_TIME = time.time()
    
    opp_color = 'W' if my_color == 'B' else 'B'
    moves = get_legal_moves(board, my_color)
    
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0] # Don't waste time thinking if there's only 1 forced move
        
    best_move = moves[0]
    depth = 1
    
    try:
        # Iterative Deepening: Search deeper and deeper until time runs out
        while True:
            # Check if we have enough time to start a new depth layer
            if time.time() - START_TIME > THINKING_TIME - 0.5:
                break 
                
            current_best = None
            max_eval = float('-inf')
            alpha = float('-inf')
            beta = float('inf')
            
            for move in moves:
                new_board = [row[:] for row in board]
                apply_move(new_board, move, my_color)
                
                # Evaluate this move using Minimax
                eval_score = minimax(new_board, depth - 1, alpha, beta, False, my_color, opp_color)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    current_best = move
                    
                alpha = max(alpha, eval_score)
                
            if current_best:
                best_move = current_best
                
            # print(f"Depth {depth} completed. Best move so far: {best_move}", file=sys.stderr)
            depth += 1
            
    except TimeoutError:
        # Time ran out during the middle of a depth search. 
        # We catch the error and safely return the best move found from the PREVIOUS fully completed depth.
        pass 
        
    return best_move

def parse_board_file(filename):
    """
    Reads the board configuration from a text file.
    Expects 8 lines of 8 characters:
      'B' = Black
      'W' = White
      'O' = Empty (Capital letter O, not zero)
    
    Mapping:
    - Line 0 of the file corresponds to Rank 8 (Top of the board)
    - Line 7 of the file corresponds to Rank 1 (Bottom of the board)
    - Column 0 corresponds to File A
    - Column 7 corresponds to File H
    
    Returns: A 2D list representing the board [row][col].
    """
    board = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            row_str = line.strip()
            if not row_str: continue # Skip empty lines
            
            row = list(row_str)
            if len(row) != 8:
                raise ValueError(f"Invalid row length at line {line_idx + 1}: Expected 8 chars, got {len(row)}")
            
            for char in row:
                if char not in ['B', 'W', 'O']:
                    raise ValueError(f"Invalid character '{char}' found. Use 'B', 'W', or 'O'.")

            board.append(row)

        if len(board) != 8:
            raise ValueError(f"Invalid board size: Expected 8 rows, got {len(board)}.")

        return board

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing board: {e}")
        sys.exit(1)
        
def notation_to_coords(notation):
    """Converts 'D5' to (3, 3)"""
    col_char = notation[0].upper()
    row_char = notation[1]
    
    col = ord(col_char) - ord('A')
    row = 8 - int(row_char)
    return row, col
    
def coords_to_notation(r, c):
    """Converts (3, 3) to 'D5'"""
    col_char = chr(c + ord('A'))
    row_char = str(8 - r)
    return f"{col_char}{row_char}"

def apply_move(board, move_str, current_player):
    """
    Updates the internal 2D board array based on a move string.
    Handles both 'D5' (removal) and 'F5-D5' (jump).
    """
    move_str = move_str.strip()
    
    # Opening Phase: Removal (e.g., "D5")
    if '-' not in move_str:
        r, c = notation_to_coords(move_str)
        board[r][c] = 'O'
        return

    # Main Phase: Jump (e.g., "F5-D5")
    parts = move_str.split('-')
    start_r, start_c = notation_to_coords(parts[0])
    end_r, end_c = notation_to_coords(parts[1])
    
    # Calculate step direction to remove jumped pieces
    dr = 0 if end_r == start_r else (1 if end_r > start_r else -1)
    dc = 0 if end_c == start_c else (1 if end_c > start_c else -1)
    
    # Execute Jump
    curr_r, curr_c = start_r, start_c
    while (curr_r, curr_c) != (end_r, end_c):
        board[curr_r][curr_c] = 'O' # Remove the jumping piece and jumped opponent
        curr_r += dr
        curr_c += dc
        if (curr_r, curr_c) != (end_r, end_c):
            board[curr_r][curr_c] = 'O' # Clear opponent
            curr_r += dr
            curr_c += dc
            
    # Place piece at final destination
    board[end_r][end_c] = current_player
    
    

def print_board(board):
    """
    Helper function to print the internal board state in a human-readable format.
     - Ranks are displayed from 8 (top) to 1 (bottom)
    """
    print("\n-- Internal Board State --")
    for r in range(len(board)):
        # Calculate Rank for display (Row 0 is Rank 8)
        rank = 8 - r
        row_str = " ".join(board[r])
        print(f"{rank} | {row_str}")
    print("    " + "-" * 15)
    print("    A B C D E F G H")
    
    
def get_legal_moves(board, player):
    """
    Generates all legal moves for the given player based on the current board.
    Handles both the opening removal phase and main jumping phase.
    """
    moves =[]
    
    # Check how many pieces have been removed to determine the phase
    empty_count = sum(row.count('O') for row in board)
    
    if empty_count == 0:
        # First turn (Black): Remove a center piece
        # Coordinates for D5, E5, D4, E4
        center_squares =[(3, 3), (3, 4), (4, 3), (4, 4)]
        for r, c in center_squares:
            if board[r][c] == player:
                moves.append(coords_to_notation(r, c))
                
    elif empty_count == 1:
        # Second turn (White): Remove an adjacent piece
        empty_r, empty_c = -1, -1
        # Find the empty square
        for r in range(8):
            for c in range(8):
                if board[r][c] == 'O':
                    empty_r, empty_c = r, c
                    break
                    
        # Find player's pieces adjacent to the empty square
        for dr, dc in[(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = empty_r + dr, empty_c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if board[nr][nc] == player:
                    # The spec hints White might also be restricted to center pieces 
                    # for the opening, so we verify it's one of the 4 centers just in case.
                    if (nr, nc) in[(3, 3), (3, 4), (4, 3), (4, 4)]:
                        moves.append(coords_to_notation(nr, nc))
                        
    else:
        # Main phase: Leaps and captures
        opp_color = 'W' if player == 'B' else 'B'
        directions =[(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        
        for r in range(8):
            for c in range(8):
                if board[r][c] == player:
                    start_notation = coords_to_notation(r, c)
                    
                    # Try jumping in all 4 directions
                    for dr, dc in directions:
                        curr_r, curr_c = r, c
                        
                        # Continue jumping as long as possible in the SAME direction
                        while True:
                            mid_r, mid_c = curr_r + dr, curr_c + dc
                            dest_r, dest_c = curr_r + 2*dr, curr_c + 2*dc
                            
                            # Ensure we don't jump off the board
                            if 0 <= dest_r < 8 and 0 <= dest_c < 8:
                                # Must jump over opponent into an empty space
                                if board[mid_r][mid_c] == opp_color and board[dest_r][dest_c] == 'O':
                                    end_notation = coords_to_notation(dest_r, dest_c)
                                    # Add this landing spot as a valid move
                                    moves.append(f"{start_notation}-{end_notation}")
                                    
                                    # Update current position to check for multi-jumps
                                    curr_r, curr_c = dest_r, dest_c
                                else:
                                    break # Path is blocked
                            else:
                                break # Edge of board reached

    return moves

    
    
def main():
    # The spec says command line args are: [Program Name] [Board File] [Player Color]
    # Example: python konane_parser.py board.txt B
    
    if len(sys.argv) < 3:
        print("Usage: python Project2.py <board_file> <B|W>", file=sys.stderr)
        sys.exit(1)

    filename = sys.argv[1]
    my_color = sys.argv[2].upper()
    opp_color = 'W' if my_color == 'B' else 'B'

    # Load initial board
    board = parse_board_file(filename)

    # 1. First move (if we are Black)
    if my_color == 'B':
        my_move = get_best_move(board, my_color)
        if not my_move:
            sys.exit(0)
            
        # MUST FLUSH STDOUT!
        print(my_move, flush=True) 
        apply_move(board, my_move, my_color)

    # 2. Main Game Loop (Ping-Pong with driver)
    while True:
        # Wait for opponent's move from standard input
        opp_move = sys.stdin.readline().strip()
        
        # If readline is empty, the game is over
        if not opp_move:
            break
            
        # Update board with opponent's move
        apply_move(board, opp_move, opp_color)

        # Calculate our best move using Minimax
        my_move = get_best_move(board, my_color)
        
        # If we have no moves, we lose. Exit cleanly.
        if not my_move:
            break
            
        # Send our move
        print(my_move, flush=True)
        
        # Update board with our move
        apply_move(board, my_move, my_color)

if __name__ == "__main__":
    main()