import sys
import time
import copy

# --- SETTINGS ---
# Time limit in seconds. Set to 9.0 to give a 1-second safety buffer 
# against the 10-second penalty mentioned in the PDF.
THINKING_TIME = 9.0 

class TimeoutException(Exception):
    pass

# --- STEP 1: PARSING ---
def parse_board_file(filename):
    board =[]
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line_idx, line in enumerate(lines):
            row_str = line.strip()
            if not row_str: continue 
            board.append(list(row_str))
        return board
    except Exception as e:
        print(f"Error parsing board: {e}", file=sys.stderr)
        sys.exit(1)

def notation_to_coords(notation):
    col_char = notation[0].upper()
    row_char = notation[1]
    col = ord(col_char) - ord('A')
    row = 8 - int(row_char)
    return row, col

def coords_to_notation(r, c):
    col_char = chr(c + ord('A'))
    row_char = str(8 - r)
    return f"{col_char}{row_char}"

def clone_board(board):
    return [row[:] for row in board]

# --- STEP 2: GAME STATE MANAGEMENT ---
def apply_move(board, move_str, current_player):
    move_str = move_str.strip()
    
    if '-' not in move_str: # Opening removal
        r, c = notation_to_coords(move_str)
        board[r][c] = 'O'
        return

    # Jump move
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
    moves =[]
    empty_count = sum(row.count('O') for row in board)
    
    # 1. Opening Phase Restrictions (Per PDF)
    if empty_count == 0:
        if player == 'B':
            # PDF explicitly lists these as the only initial moves
            return ["D5", "E4"]
    elif empty_count == 1:
        if player == 'W':
            # PDF explicitly restricts centerpieces. 
            # D5 was removed? White removes E5 or D4.
            # E4 was removed? White removes D4 or E5.
            valid_w = []
            if board[3][4] == 'W': valid_w.append("E5")
            if board[4][3] == 'W': valid_w.append("D4")
            return valid_w

    # 2. Main Phase: Jumping
    opp_color = 'W' if player == 'B' else 'B'
    directions =[(-1, 0), (1, 0), (0, -1), (0, 1)] 
    
    for r in range(8):
        for c in range(8):
            if board[r][c] == player:
                start_notation = coords_to_notation(r, c)
                
                for dr, dc in directions:
                    curr_r, curr_c = r, c
                    
                    while True:
                        mid_r, mid_c = curr_r + dr, curr_c + dc
                        dest_r, dest_c = curr_r + 2*dr, curr_c + 2*dc
                        
                        if 0 <= dest_r < 8 and 0 <= dest_c < 8:
                            if board[mid_r][mid_c] == opp_color and board[dest_r][dest_c] == 'O':
                                end_notation = coords_to_notation(dest_r, dest_c)
                                moves.append(f"{start_notation}-{end_notation}")
                                curr_r, curr_c = dest_r, dest_c
                            else:
                                break
                        else:
                            break
    return moves

# --- STEP 3: MINIMAX AI ---
def evaluate_board(board, player):
    """
    Heuristic: Mobility.
    The player with more available moves in the future has a massive advantage.
    Score = (My Moves) - (Opponent Moves).
    """
    opp = 'W' if player == 'B' else 'B'
    my_moves = len(get_legal_moves(board, player))
    opp_moves = len(get_legal_moves(board, opp))
    
    # If the opponent has no moves, we WIN immediately.
    if opp_moves == 0:
        return 10000 
    # If we have no moves, we LOSE immediately.
    if my_moves == 0:
        return -10000
        
    return my_moves - opp_moves

def minimax(board, depth, alpha, beta, is_maximizing, current_player, original_player, start_time):
    # Check if we are running out of time
    if time.time() - start_time > THINKING_TIME:
        raise TimeoutException()

    moves = get_legal_moves(board, current_player)
    
    # Base case: reached depth limit or game over
    if depth == 0 or not moves:
        return evaluate_board(board, original_player), None

    best_move = moves[0]
    next_player = 'W' if current_player == 'B' else 'B'

    if is_maximizing:
        max_eval = -float('inf')
        for move in moves:
            new_board = clone_board(board)
            apply_move(new_board, move, current_player)
            
            eval_score, _ = minimax(new_board, depth - 1, alpha, beta, False, next_player, original_player, start_time)
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break # Alpha-Beta Pruning
        return max_eval, best_move
        
    else:
        min_eval = float('inf')
        for move in moves:
            new_board = clone_board(board)
            apply_move(new_board, move, current_player)
            
            eval_score, _ = minimax(new_board, depth - 1, alpha, beta, True, next_player, original_player, start_time)
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                
            beta = min(beta, eval_score)
            if beta <= alpha:
                break # Alpha-Beta Pruning
        return min_eval, best_move

def get_best_move(board, player):
    """
    Uses Iterative Deepening. It searches Depth 1, then Depth 2, then Depth 3...
    until the time limit runs out. This ensures we ALWAYS have a good move ready
    and never get a 25% penalty for taking too long.
    """
    start_time = time.time()
    best_move = None
    depth = 1
    
    # Fallback to a valid move instantly just in case
    legal_moves = get_legal_moves(board, player)
    if not legal_moves:
        return None
    best_move = legal_moves[0] 
    
    try:
        while True:
            # We pass original_player so the evaluation function knows who we are rooting for
            score, move = minimax(board, depth, -float('inf'), float('inf'), True, player, player, start_time)
            if move:
                best_move = move
            
            # If we see a forced win, no need to think deeper
            if score == 10000:
                break
                
            depth += 1
            
    except TimeoutException:
        # Time ran out! We catch this and return the best_move 
        # from the highest completely finished depth.
        pass 
        
    return best_move

# --- STEP 4: GAME EXECUTION ---
def main():
    if len(sys.argv) < 3:
        sys.exit(1)

    filename = sys.argv[1]
    my_color = sys.argv[2].upper()
    opp_color = 'W' if my_color == 'B' else 'B'

    board = parse_board_file(filename)

    # 1. First move (if we are Black)
    if my_color == 'B':
        my_move = get_best_move(board, my_color)
        print(my_move, flush=True) 
        apply_move(board, my_move, my_color)

    # 2. Main Game Loop
    while True:
        # Wait for opponent's move
        opp_move = sys.stdin.readline().strip()
        
        if not opp_move:
            break # Game over
            
        # Register opponent's move
        apply_move(board, opp_move, opp_color)

        # Calculate our move
        my_move = get_best_move(board, my_color)
        
        if not my_move:
            break # We have no moves, we lost
            
        # Send our move
        print(my_move, flush=True)
        
        # Register our move locally
        apply_move(board, my_move, my_color)

if __name__ == "__main__":
    main()