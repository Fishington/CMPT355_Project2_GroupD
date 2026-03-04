import sys

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

        # Iterate through lines
        for line_idx, line in enumerate(lines):
            # Clean
            row_str = line.strip()
            row = list(row_str)
            
            # Make sure row is exactly 8 chars
            if len(row) != 8:
                raise ValueError(f"Invalid row length at line {line_idx + 1}: Expected 8 chars, got {len(row)}")
            
            # Make sure all chars are valid
            for char in row:
                if char not in ['B', 'W', 'O']:
                    raise ValueError(f"Invalid character '{char}' found. Use 'B', 'W', or 'O'.")

            board.append(row)

        # Make sure there are exactly 8 rows
        if len(board) != 8:
            raise ValueError(f"Invalid board size: Expected 8 rows, got {len(board)}.")

        return board

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing board: {e}")
        sys.exit(1)

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
    
    
def main():
    # The spec says command line args are: [Program Name] [Board File] [Player Color]
    # Example: python konane_parser.py board.txt B
    
    if len(sys.argv) < 3:
        print("Usage: python konane_parser.py <board_file> <B|W>")
        sys.exit(1)

    #File name
    filename = sys.argv[1]
    #Player color
    player_color = sys.argv[2].upper()

    if player_color not in ['B', 'W']:
        print("Error: Player color must be 'B' or 'W'.")
        sys.exit(1)

    # Parse the File
    current_board = parse_board_file(filename)
    
    # Print the board and player color
    print(f"Successfully loaded board for player: {player_color}")
    print_board(current_board)
    
    print(current_board[3][4])
    

if __name__ == "__main__":
    main()