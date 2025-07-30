import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pygame
import os
import copy
import time
import threading
import chess

# === Utility Functions ===
def clamp(value, minimum, maximum):
    """
    Clamp value between minimum and maximum inclusive.
    """
    return max(minimum, min(value, maximum))


def in_bounds(col, row):
    """
    Check if board coordinates (col, row) are within 0..7.
    """
    return 0 <= col < 8 and 0 <= row < 8


def squares_between(start_file, end_file):
    """
    Return all file indices strictly between start_file and end_file.
    """
    low, high = sorted((start_file, end_file))
    return list(range(low + 1, high))


# === Piece Base Class ===
class Piece:
    symbol = '?'

    def __init__(self, color):
        """
        color: 'w' for white, 'b' for black
        """
        self.color = color
        self.has_moved = False

    def get_moves(self, position, board):
        """
        Return a list of pseudo-legal moves [(col, row), ...].
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def _is_opponent(self, other):
        return other is not None and other.color != self.color

    def _is_empty(self, square):
        return square is None


# === Specific Piece Classes ===
class King(Piece):
    symbol = 'K'

    def get_moves(self, position, board):
        col, row = position
        moves = []

        # Standard king moves (one square any direction)
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dc == 0 and dr == 0:
                    continue
                target_col = col + dc
                target_row = row + dr
                if in_bounds(target_col, target_row):
                    occupant = board[target_row][target_col]
                    if self._is_empty(occupant) or self._is_opponent(occupant):
                        moves.append((target_col, target_row))
        
        # Castling - only if king hasn't moved
        if not self.has_moved:
            # Kingside castling (O-O)
            if col + 3 < 8:  # Make sure we're in bounds
                rook = board[row][col + 3]
                if isinstance(rook, Rook) and not rook.has_moved and rook.color == self.color:
                    # Check if squares between king and rook are empty
                    if all(board[row][c] is None for c in range(col + 1, col + 3)):
                        moves.append((col + 2, row))  # Castle kingside
                    
            # Queenside castling (O-O-O)
            if col - 4 >= 0:  # Make sure we're in bounds
                rook = board[row][col - 4]
                if isinstance(rook, Rook) and not rook.has_moved and rook.color == self.color:
                    # Check if squares between king and rook are empty
                    if all(board[row][c] is None for c in range(col - 3, col)):
                        moves.append((col - 2, row))  # Castle queenside

        return moves


class Queen(Piece):
    symbol = 'Q'

    def get_moves(self, position, board):
        # Combine rook and bishop moves
        from itertools import chain
        return list(chain(
            Rook(self.color).get_moves(position, board),
            Bishop(self.color).get_moves(position, board)
        ))


class Rook(Piece):
    symbol = 'R'

    def get_moves(self, position, board):
        col, row = position
        moves = []

        # Horizontal and vertical directions
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dc, dr in directions:
            curr_col, curr_row = col + dc, row + dr
            while in_bounds(curr_col, curr_row):
                occupant = board[curr_row][curr_col]
                if self._is_empty(occupant):
                    moves.append((curr_col, curr_row))
                elif self._is_opponent(occupant):
                    moves.append((curr_col, curr_row))
                    break
                else:
                    break
                curr_col += dc
                curr_row += dr

        return moves


class Bishop(Piece):
    symbol = 'B'

    def get_moves(self, position, board):
        col, row = position
        moves = []

        # Diagonal directions
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dc, dr in directions:
            curr_col, curr_row = col + dc, row + dr
            while in_bounds(curr_col, curr_row):
                occupant = board[curr_row][curr_col]
                if self._is_empty(occupant):
                    moves.append((curr_col, curr_row))
                elif self._is_opponent(occupant):
                    moves.append((curr_col, curr_row))
                    break
                else:
                    break
                curr_col += dc
                curr_row += dr

        return moves


class Knight(Piece):
    symbol = 'N'

    def get_moves(self, position, board):
        col, row = position
        moves = []

        # L-shaped moves
        deltas = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                  (1, 2), (1, -2), (-1, 2), (-1, -2)]
        for dc, dr in deltas:
            target_col = col + dc
            target_row = row + dr
            if in_bounds(target_col, target_row):
                occupant = board[target_row][target_col]
                if self._is_empty(occupant) or self._is_opponent(occupant):
                    moves.append((target_col, target_row))

        return moves


class Pawn(Piece):
    symbol = 'P'

    def get_moves(self, position, board):
        col, row = position
        moves = []

        # Determine forward direction
        direction = -1 if self.color == 'w' else 1

        # Single-step forward
        forward_row = row + direction
        if in_bounds(col, forward_row) and self._is_empty(board[forward_row][col]):
            moves.append((col, forward_row))

            # Two-step from starting rank
            start_rank = 6 if self.color == 'w' else 1
            two_step_row = row + 2 * direction
            if row == start_rank and self._is_empty(board[two_step_row][col]):
                moves.append((col, two_step_row))

        # Diagonal captures
        for dc in (-1, 1):
            capture_col = col + dc
            capture_row = row + direction
            if in_bounds(capture_col, capture_row):
                occupant = board[capture_row][capture_col]
                if self._is_opponent(occupant):
                    moves.append((capture_col, capture_row))

        return moves


# === Main Game Class ===
class Game:
    def __init__(self, square_size=60):
        pygame.init()
        self.square_size = square_size
        self.board_size = 8 * square_size
        
        # Create a slightly larger window for menu buttons
        self.window_width = self.board_size
        self.window_height = self.board_size + 50  # Extra space for buttons
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Chess")
        
        # Initialize fonts
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 18)

        # Colors for rendering
        self.WHITE = (245, 245, 220)
        self.BLACK = (50, 50, 50)
        self.GRAY = (100, 100, 100)
        self.HIGHLIGHT = (124, 252, 0)  # Light green
        self.BUTTON = (80, 80, 200)
        self.BUTTON_HOVER = (100, 100, 220)
        self.TEXT = (255, 255, 255)

        # Game state
        self.state = "menu"  # "menu", "game", "game_over", "ai_learning", "self_learn", "pgn_learn"
        self.mode = None  # "2p" or "ai"
        self.ai = None
        self.game_result = ""  # For displaying the result
        self.learning_progress = ""
        self.training_log = []
        self.current_board = None
        self.training_game = False
        self.stop_training = False  # Flag to stop training
        self.stop_button = pygame.Rect(self.window_width//2 - 100, self.window_height - 70, 200, 50)  # Stop button for training

        # Build initial board setup
        self.chessboard = self._create_board()

        # Selection state
        self.picked_piece = None
        self.picked_pos = None

        # Track turns ('w' = White, 'b' = Black)
        self.turn_color = 'w'

        # Track last move for en passant
        self.last_move = None  # Format: (piece, from_pos, to_pos)
        
        # State for pawn promotion
        self.promotion_pawn = None  # Position of pawn awaiting promotion
        self.promotion_choices = []  # List of clickable areas for promotion choices

        # Load piece images
        self.images = self._load_images()
    
    def _create_board(self):
        """
        Set up starting position:
        - 8th rank: R N B Q K B N R (black)
        - 7th rank: 8 black pawns
        - ranks 3–6: empty
        - 2nd rank: 8 white pawns
        - 1st rank: R N B Q K B N R (white)
        """
        black_back = [Rook('b'), Knight('b'), Bishop('b'), Queen('b'), King('b'), Bishop('b'), Knight('b'), Rook('b')]
        black_pawns = [Pawn('b') for _ in range(8)]
        empty = [[None] * 8 for _ in range(4)]
        white_pawns = [Pawn('w') for _ in range(8)]
        white_back = [Rook('w'), Knight('w'), Bishop('w'), Queen('w'), King('w'), Bishop('w'), Knight('w'), Rook('w')]

        return [black_back, black_pawns] + empty + [white_pawns, white_back]

    def _load_images(self):
        """
        Load PNGs named 'wK.png', 'bQ.png', etc., from 'src/images/'.
        """
        images = {}
        asset_dir = 'src/images'
        for color in ('w', 'b'):
            for cls in (King, Queen, Rook, Bishop, Knight, Pawn):
                key = color + cls.symbol
                path = os.path.join(asset_dir, f"{key}.png")
                if os.path.exists(path):
                    img = pygame.image.load(path).convert_alpha()
                    images[key] = pygame.transform.smoothscale(img, (self.square_size, self.square_size))
        return images
    
    def _is_check(self, color, board=None):
        """
        Determine if the king of the specified color is in check.
        
        Args:
            color: 'w' or 'b' - the color of the king to check
            board: Optional board state to check. If None, uses the current board.
            
        Returns:
            True if the king is in check, False otherwise
        """
        # Use the current board if none is provided
        if board is None:
            board = self.chessboard
            
        # Find the king's position
        king_pos = None
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if isinstance(piece, King) and piece.color == self.turn_color:
                    king_pos = (col, row)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return False  # Should not happen in a valid game
        
        # Check if any opponent piece can capture the king
        opponent_color = 'b' if self.turn_color == 'w' else 'w'
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece and piece.color == opponent_color:
                    moves = piece.get_moves((col, row), board)
                    if king_pos in moves:
                        return True
        return False

    def _is_checkmate(self, color):
        """
        Determine if the king of the specified color is in checkmate.
        
        Args:
            color: 'w' or 'b' - the color of the king to check
            
        Returns:
            True if the king is in checkmate, False otherwise
        """
        # If not in check, can't be checkmate
        if not self._is_check(color):
            return False
            
        # Try all possible moves for all pieces of this color
        for row in range(8):
            for col in range(8):
                piece = self.chessboard[row][col]
                if piece and piece.color == color:
                    for move_col, move_row in piece.get_moves((col, row), self.chessboard):
                        # Create a hypothetical board with this move
                        test_board = copy.deepcopy(self.chessboard)
                        test_board[move_row][move_col] = test_board[row][col]
                        test_board[row][col] = None
                        
                        # If this move gets us out of check, it's not checkmate
                        if not self._is_check(color, test_board):
                            return False
        # No move gets us out of check, so it's checkmate
        return True

    def reset_game(self):
        """Reset the game to starting position"""
        self.chessboard = self._create_board()
        self.picked_piece = None
        self.picked_pos = None
        self.turn_color = 'w'
        self.last_move = None
        self.promotion_pawn = None
        self.promotion_choices = []
        self.state = "game"
        
        # Initialize python-chess board
        import chess
        self.current_board = chess.Board()

    def get_legal_moves(self, position):
        """
        Return moves for the piece at 'position', respecting turn order and check.
        """
        col, row = position
        piece = self.chessboard[row][col]
        if not piece or piece.color != self.turn_color:
            return []
            
        # Get pseudo-legal moves
        moves = piece.get_moves(position, self.chessboard)
        
        # Add en passant moves for pawns
        if isinstance(piece, Pawn) and self.last_move:
            last_piece, (last_from_col, last_from_row), (last_to_col, last_to_row) = self.last_move
            
            # Check if last move was a pawn moving two squares
            if isinstance(last_piece, Pawn) and abs(last_to_row - last_from_row) == 2:
                # Check if our pawn is adjacent to the opponent's pawn
                if abs(last_to_col - col) == 1 and row == last_to_row:
                    # Determine en passant capture square
                    capture_row = row + (-1 if piece.color == 'w' else 1)
                    moves.append((last_to_col, capture_row))
        
        # Filter out moves that would leave the king in check
        legal_moves = []
        for move_col, move_row in moves:
            # Special handling for castling - king can't castle through check
            if isinstance(piece, King) and abs(move_col - col) == 2:
                # Direction of castling
                step = 1 if move_col > col else -1
                
                # Check each square the king passes through for check
                in_check = False
                for test_col in range(col, move_col + step, step):
                    if test_col == col:  # Skip king's starting position
                        continue
                        
                    # Create test board with king at this position
                    test_board = copy.deepcopy(self.chessboard)
                    test_board[row][test_col] = test_board[row][col]
                    test_board[row][col] = None
                    
                    # If king would be in check, castling is illegal
                    if self._is_check(piece.color, test_board):
                        in_check = True
                        break
                        
                if in_check:
                    continue  # Skip this move
            
            # Handle en passant in the test board
            test_board = copy.deepcopy(self.chessboard)
            # If this is an en passant capture
            is_en_passant = (
                isinstance(piece, Pawn) and 
                move_col != col and 
                test_board[move_row][move_col] is None
            )
            
            # Move the piece in the test board
            test_board[move_row][move_col] = test_board[row][col]
            test_board[row][col] = None
            
            # For en passant, remove the captured pawn
            if is_en_passant and self.last_move:
                _, _, (last_to_col, last_to_row) = self.last_move
                test_board[last_to_row][last_to_col] = None
            
            # If the move doesn't leave the king in check, it's legal
            if not self._is_check(piece.color, test_board):
                legal_moves.append((move_col, move_row))
                
        return legal_moves

    def _mouse_to_square(self, x, y):
        return (
            clamp(x // self.square_size, 0, 7),
            clamp(y // self.square_size, 0, 7)
        )

    def _print_board(self):
        for rank in self.chessboard:
            print([f"{p.symbol}{p.color}" if p else None for p in rank])
        print('-' * 32)

    def _draw_menu(self):
        """Draw the main menu with game mode options"""
        # Clear screen
        self.window.fill(self.BLACK)
        
        # Draw title
        title = self.font.render("Chess Game", True, self.WHITE)
        title_rect = title.get_rect(center=(self.window_width//2, 50))
        self.window.blit(title, title_rect)
        
        # Draw "2 Players" button
        two_player_btn = pygame.Rect(self.window_width//2 - 100, 120, 200, 50)
        two_player_hover = two_player_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.window, self.BUTTON_HOVER if two_player_hover else self.BUTTON, two_player_btn)
        two_player_text = self.font.render("2 Players", True, self.TEXT)
        two_player_rect = two_player_text.get_rect(center=two_player_btn.center)
        self.window.blit(two_player_text, two_player_btn)
        
        # Draw "Player vs AI" button
        ai_btn = pygame.Rect(self.window_width//2 - 100, 200, 200, 50)
        ai_hover = ai_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.window, self.BUTTON_HOVER if ai_hover else self.BUTTON, ai_btn)
        ai_text = self.font.render("Player vs AI", True, self.TEXT)
        ai_rect = ai_text.get_rect(center=ai_btn.center)
        self.window.blit(ai_text, ai_btn)
        
        # Draw "AI Learning" button
        ai_learning_btn = pygame.Rect(self.window_width//2 - 100, 280, 200, 50)
        ai_learning_hover = ai_learning_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.window, self.BUTTON_HOVER if ai_learning_hover else self.BUTTON, ai_learning_btn)
        ai_learning_text = self.font.render("AI Learning", True, self.TEXT)
        ai_learning_rect = ai_learning_text.get_rect(center=ai_learning_btn.center)
        self.window.blit(ai_learning_text, ai_learning_btn)
        
        return two_player_btn, ai_btn, ai_learning_btn
    
    def _handle_menu_click(self, pos):
        """Handle clicks on the main menu"""
        two_player_btn = pygame.Rect(self.window_width//2 - 100, 120, 200, 50)
        ai_btn = pygame.Rect(self.window_width//2 - 100, 200, 200, 50)
        ai_learning_btn = pygame.Rect(self.window_width//2 - 100, 280, 200, 50)
        
        if two_player_btn.collidepoint(pos):
            self.mode = "2p"
            self.reset_game()
        elif ai_btn.collidepoint(pos):
            self.mode = "ai"
            # Initialize AI if not already
            if self.ai is None:
                try:
                    from chess_ai import ChessAI
                    self.ai = ChessAI(color='b')
                except ImportError:
                    print("Could not import ChessAI. Using random moves.")
                    # Define a simple fallback AI
                    class SimpleAI:
                        def __init__(self, color='b'):
                            self.color = color
                        def make_move(self, game):
                            import random
                            moves = []
                            for r in range(8):
                                for c in range(8):
                                    piece = game.chessboard[r][c]
                                    if piece and piece.color == self.color:
                                        for move in game.get_legal_moves((c, r)):
                                            moves.append(((c, r), move))
                            return random.choice(moves) if moves else None
                    self.ai = SimpleAI(color='b')
            self.reset_game()
        elif ai_learning_btn.collidepoint(pos):
            self.state = "ai_learning"

    def _draw_ai_learning_menu(self):
        """Draw the AI learning options menu"""
        # Clear screen
        self.window.fill(self.BLACK)
        
        # Draw title
        title = self.font.render("AI Learning Options", True, self.WHITE)
        title_rect = title.get_rect(center=(self.window_width//2, 50))
        self.window.blit(title, title_rect)
        
        # Draw "Self-Learn" button
        self_learn_btn = pygame.Rect(self.window_width//2 - 100, 120, 200, 50)
        self_learn_hover = self_learn_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.window, self.BUTTON_HOVER if self_learn_hover else self.BUTTON, self_learn_btn)
        self_learn_text = self.font.render("Self-Learn", True, self.TEXT)
        self_learn_rect = self_learn_text.get_rect(center=self_learn_btn.center)
        self.window.blit(self_learn_text, self_learn_btn)
        
        # Draw "Preprocess PGNs" button
        preprocess_btn = pygame.Rect(self.window_width//2 - 100, 190, 200, 50)
        preprocess_hover = preprocess_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.window, self.BUTTON_HOVER if preprocess_hover else self.BUTTON, preprocess_btn)
        preprocess_text = self.font.render("Preprocess PGNs", True, self.TEXT)
        preprocess_rect = preprocess_text.get_rect(center=preprocess_btn.center)
        self.window.blit(preprocess_text, preprocess_btn)
        
        # Draw "PGN Learn" button
        pgn_learn_btn = pygame.Rect(self.window_width//2 - 100, 260, 200, 50)
        pgn_learn_hover = pgn_learn_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.window, self.BUTTON_HOVER if pgn_learn_hover else self.BUTTON, pgn_learn_btn)
        pgn_learn_text = self.font.render("PGN Learn", True, self.TEXT)
        pgn_learn_rect = pgn_learn_text.get_rect(center=pgn_learn_btn.center)
        self.window.blit(pgn_learn_text, pgn_learn_btn)
        
        # Draw "Back" button
        back_btn = pygame.Rect(self.window_width//2 - 100, self.window_height - 70, 200, 50)
        back_hover = back_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.window, self.BUTTON_HOVER if back_hover else self.BUTTON, back_btn)
        back_text = self.font.render("Back to Menu", True, self.TEXT)
        back_rect = back_text.get_rect(center=back_btn.center)
        self.window.blit(back_text, back_btn)
        
        # Draw "Stop Training" button only during training
        if self.training_game:
            stop_button = pygame.Rect(300, 400, 200, 50)
            pygame.draw.rect(self.window, self.BUTTON, stop_button)
            stop_text = self.font.render("Stop Training", True, self.WHITE)
            self.window.blit(stop_text, (300, 400))
            self.stop_button = stop_button  # Store for event handling

        return self_learn_btn, pgn_learn_btn, preprocess_btn, back_btn

    def _handle_ai_learning_click(self, pos):
        """Handle clicks on the AI learning menu"""
        self_learn_btn = pygame.Rect(self.window_width//2 - 100, 120, 200, 50)
        pgn_learn_btn = pygame.Rect(self.window_width//2 - 100, 260, 200, 50)
        preprocess_btn = pygame.Rect(self.window_width//2 - 100, 190, 200, 50)
        back_btn = pygame.Rect(self.window_width//2 - 100, self.window_height - 70, 200, 50)
        
        if self_learn_btn.collidepoint(pos):
            self.state = "self_learn"
            # Initialize AI if not already done
            if self.ai is None:
                try:
                    from chess_ai import ChessAI
                    self.ai = ChessAI(color='b')
                except ImportError:
                    print("Could not import ChessAI. AI learning unavailable.")
                    self.state = "ai_learning"
                    return
            
            # Start self-play training
            def update_progress(msg):
                self.learning_progress = msg
                
            self.ai.train_self_play(num_games=100, callback=update_progress)
            
        elif pgn_learn_btn.collidepoint(pos):
            self.state = "pgn_learn"
            # Initialize AI if not already done
            if self.ai is None:
                try:
                    from chess_ai import ChessAI
                    self.ai = ChessAI(color='b')
                except ImportError:
                    print("Could not import ChessAI. AI learning unavailable.")
                    self.state = "ai_learning"
                    return
            
            # Start PGN training
            def update_progress(msg):
                self.learning_progress = msg
                
            self.ai.train_from_pgn(callback=update_progress)
            
        elif preprocess_btn.collidepoint(pos):
            # Start PGN preprocessing
            def update_progress(msg):
                self.learning_progress = msg
                
            self.state = "ai_learning"  # Stay in the same menu
            self.training_log.append("Starting PGN preprocessing...")
            print("[UI] Starting PGN preprocessing...")
            
            # Run in a separate thread
            def run_filter():
                try:
                    from src.pgn_filter import filter_and_annotate_pgn
                    filter_and_annotate_pgn(callback=update_progress)
                except Exception as e:
                    self.training_log.append(f"Error in PGN filter: {e}")
                    print(f"[UI] Error in PGN filter: {e}")
            
            threading.Thread(target=run_filter).start()
            
        elif back_btn.collidepoint(pos):
            self.state = "menu"

    def _draw_self_learn_screen(self):
        """Draw the self-learning progress screen"""
        # Clear screen
        self.window.fill(self.BLACK)
        
        # Draw title
        title = self.font.render("AI Self-Learning", True, self.WHITE)
        title_rect = title.get_rect(center=(self.window_width//2, 50))
        self.window.blit(title, title_rect)
        
        # Draw progress
        progress_text = self.font.render(self.learning_progress, True, self.WHITE)
        progress_rect = progress_text.get_rect(center=(self.window_width//2, 100))
        self.window.blit(progress_text, progress_rect)
        
        # Draw log of training
        log_y = 150
        if hasattr(self.ai, 'training_log'):
            log_title = self.font.render("Training Log:", True, self.WHITE)
            self.window.blit(log_title, (50, log_y - 30))

            for i, log_entry in enumerate(self.ai.training_log[-10:]):  # Show last 10 log entries
                log_text = self.small_font.render(log_entry, True, self.WHITE)
                self.window.blit(log_text, (50, log_y + i * 25))
        
        # Draw visualization of self-play games
        if hasattr(self.ai, 'self_play_results') and self.ai.self_play_results:
            # Find the latest game
            latest_game = max(result['game'] for result in self.ai.self_play_results)
            latest_moves = [r for r in self.ai.self_play_results if r['game'] == latest_game]
            
            if latest_moves:
                # Display the latest board position
                latest_board = latest_moves[-1]['board']
                board_text = self.font.render(f"Latest position from game {latest_game}", True, self.WHITE)
                self.window.blit(board_text, (50, 400))
                
                # Show last few moves
                move_y = 430
                for i, move in enumerate(latest_moves[-5:]):
                    move_text = self.small_font.render(f"Move: {move['move']}", True, self.WHITE)
                    self.window.blit(move_text, (50, move_y + i * 25))
        
        # Draw "Back" button
        back_btn = pygame.Rect(self.window_width//2 - 100, self.window_height - 70, 200, 50)
        back_hover = back_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.window, self.BUTTON_HOVER if back_hover else self.BUTTON, back_btn)
        back_text = self.font.render("Back to Menu", True, self.TEXT)
        back_rect = back_text.get_rect(center=back_btn.center)
        self.window.blit(back_text, back_btn)
        
        # Draw "Stop Training" button
        stop_btn = self.stop_button
        stop_hover = stop_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.window, self.BUTTON_HOVER if stop_hover else self.BUTTON, stop_btn)
        stop_text = self.font.render("Stop Training", True, self.TEXT)
        stop_rect = stop_text.get_rect(center=stop_btn.center)
        self.window.blit(stop_text, stop_btn)
        
        # Check if training is complete
        if hasattr(self.ai, 'training_in_progress') and not self.ai.training_in_progress:
            if hasattr(self.ai, 'training_log') and self.ai.training_log and "complete" in self.ai.training_log[-1].lower():
                complete_text = self.font.render("Training Complete!", True, (0, 255, 0))
                self.window.blit(complete_text, (self.window_width//2 - 100, 350))
        
        return back_btn

    def self_train(self):
        import chess
        self.current_board = chess.Board()
        self.training_game = True
        self.stop_training = False

        while not self.current_board.is_game_over() and not self.stop_training:
            for color in [chess.WHITE, chess.BLACK]:
                if self.current_board.is_game_over() or self.stop_training:
                    break

                move = self.ai_suggest_move(self.current_board, color)
                self.current_board.push(move)

                self._draw_board()
                pygame.display.flip()
                pygame.time.wait(500)

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        return
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if self.stop_button.collidepoint(event.pos):
                            self.stop_training = True
                            print("[UI] Self-training stopped.")
                            return

        self.training_game = False  # Hide stop button after training ends

    def _draw_pgn_learn_screen(self):
        """Draw the PGN learning progress screen"""
        # Clear screen
        self.window.fill(self.BLACK)

        # Draw title
        title = self.font.render("AI PGN Learning", True, self.WHITE)
        title_rect = title.get_rect(center=(self.window_width//2, 50))
        self.window.blit(title, title_rect)

        # Draw progress
        progress_text = self.font.render(self.learning_progress, True, self.WHITE)
        progress_rect = progress_text.get_rect(center=(self.window_width//2, 100))
        self.window.blit(progress_text, progress_rect)

        # Check for db directory
        db_dir = "filtered_db"
        if not os.path.exists(db_dir):
            dir_text = self.font.render(f"Directory '{db_dir}' not found. Create it and add PGN files.", True, (255, 100, 100))
            self.window.blit(dir_text, (50, 150))
        else:
            pgn_files = [f for f in os.listdir(db_dir) if f.endswith('.pgn')]
            if not pgn_files:
                files_text = self.font.render("No PGN files found in the db directory.", True, (255, 100, 100))
                self.window.blit(files_text, (50, 150))
            else:
                files_text = self.font.render(f"Found {len(pgn_files)} PGN files in the db directory.", True, self.WHITE)
                self.window.blit(files_text, (50, 150))

                # List some of the files
                for i, pgn_file in enumerate(pgn_files[:5]):  # Show up to 5 files
                    file_text = self.small_font.render(pgn_file, True, self.WHITE)
                    self.window.blit(file_text, (70, 180 + i * 25))
                if len(pgn_files) > 5:
                    more_text = self.small_font.render(f"... and {len(pgn_files) - 5} more", True, self.WHITE)
                    self.window.blit(more_text, (70, 180 + 5 * 25))

        # Draw log of training
        log_y = 350
        if hasattr(self.ai, 'training_log'):
            log_title = self.font.render("Training Log:", True, self.WHITE)
            self.window.blit(log_title, (50, log_y - 30))

            for i, log_entry in enumerate(self.ai.training_log[-10:]):  # Show last 10 log entries
                log_text = self.small_font.render(log_entry, True, self.WHITE)
                self.window.blit(log_text, (50, log_y + i * 25))

        # Draw "Back" button
        back_btn = pygame.Rect(self.window_width//2 - 100, self.window_height - 70, 200, 50)
        back_hover = back_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.window, self.BUTTON_HOVER if back_hover else self.BUTTON, back_btn)
        back_text = self.font.render("Back to Menu", True, self.TEXT)
        back_rect = back_text.get_rect(center=back_btn.center)
        self.window.blit(back_text, back_btn)

        # Check if training is complete
        if hasattr(self.ai, 'training_in_progress') and not self.ai.training_in_progress:
            if hasattr(self.ai, 'training_log') and self.ai.training_log and "complete" in self.ai.training_log[-1].lower():
                complete_text = self.font.render("Training Complete!", True, (0, 255, 0))
                self.window.blit(complete_text, (self.window_width//2 - 100, 300))

        return back_btn

    def _handle_learning_screen_click(self, pos):
        """Handle clicks on the learning screens"""
        back_btn = pygame.Rect(self.window_width//2 - 100, self.window_height - 70, 200, 50)

        if back_btn.collidepoint(pos):
            self.state = "menu"

    def _draw_game_over(self):
        """Draw the game over screen"""
        # Draw a semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Black with alpha
        self.window.blit(overlay, (0, 0))

        # Draw game over message
        message = self.font.render(self.game_result, True, self.WHITE)
        message_rect = message.get_rect(center=( self.window_width//2, self.window_height//2 - 30))
        self.window.blit(message, message_rect)

        # Draw "Back to Menu" button
        menu_btn = pygame.Rect(self.window_width//2 - 100, self.window_height//2 + 20, 200, 50)
        menu_hover = menu_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.window, self.BUTTON_HOVER if menu_hover else self.BUTTON, menu_btn)
        menu_text = self.font.render("Back to Menu", True, self.TEXT)
        menu_rect = menu_text.get_rect(center=menu_btn.center)
        self.window.blit(menu_text, menu_btn)

        return menu_btn

    def _handle_game_over_click(self, pos):
        """Handle clicks on the game over screen"""
        menu_btn = pygame.Rect(self.window_width//2 - 100, self.window_height//2 + 20, 200, 50)

        if menu_btn.collidepoint(pos):
            self.state = "menu"

    def _draw_promotion_selection(self):
        """Draw the promotion piece selection dialog"""
        if not self.promotion_pawn:
            return

        col, row = self.promotion_pawn
        # Determine if we're promoting a white or black pawn
        color = self.chessboard[row][col].color

        # Create a background for the promotion dialog
        dialog_width = self.square_size * 4
        dialog_height = self.square_size
        dialog_x = col * self.square_size
        dialog_y = row * self.square_size

        # Adjust if dialog would go off-screen
        if dialog_x + dialog_width > self.board_size:
            dialog_x = self.board_size - dialog_width

        dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
        pygame.draw.rect(self.window, (220, 220, 220), dialog_rect)
        pygame.draw.rect(self.window, (0, 0, 0), dialog_rect, 2)  # Black border

        # Draw the four promotion options: Queen, Rook, Bishop, Knight
        piece_classes = [Queen, Rook, Bishop, Knight]
        self.promotion_choices = []

        for i, piece_class in enumerate(piece_classes):
            piece_rect = pygame.Rect(
                dialog_x + i * self.square_size,
                dialog_y,
                self.square_size,
                self.square_size
            )
            # Store the clickable area and the piece class
            self.promotion_choices.append((piece_rect, piece_class))

            # Draw piece background
            pygame.draw.rect(self.window, (255, 255, 255) if i % 2 == 0 else (200, 200, 200), piece_rect)

            # Draw piece image
            key = color + piece_class.symbol
            img = self.images.get(key)
            if img:
                self.window.blit(img, piece_rect)

    def _complete_promotion(self, piece_class):
        """Replace the pawn with the chosen piece"""
        if not self.promotion_pawn:
            return

        col, row = self.promotion_pawn
        pawn = self.chessboard[row][col]
        if pawn:
            # Create the new piece
            color = pawn.color
            self.chessboard[row][col] = piece_class(color)
            print(f"Promoted to {piece_class.symbol}{color}")

            # Check for checkmate after promotion
            if self._is_check(self.turn_color):
                print(f"Check!")
                if self._is_checkmate(self.turn_color):
                    winner = 'White' if self.turn_color == 'b' else 'Black'
                    self.game_result = f"Checkmate! {winner} wins!"
                    self.state = "game_over"
                    print(self.game_result)

        # Reset promotion state
        self.promotion_pawn = None
        self.promotion_choices = []

        # Switch turns (since this was delayed until promotion choice)
        self.turn_color = 'b' if self.turn_color == 'w' else 'w'

    def _check_for_promotion(self, piece, row):
        """Check if a pawn has reached the promotion rank"""
        if isinstance(piece, Pawn):
            # Pawns promote on the 1st rank (white) or 8th rank (black)
            if (piece.color == 'w' and row == 0) or (piece.color == 'b' and row == 7):
                return True
        return False

    def _make_move(self, from_pos, to_pos):
        """Execute a move from from_pos to to_pos"""
        old_col, old_row = from_pos
        col, row = to_pos
        piece = self.chessboard[old_row][old_col]

        if not piece:
            return False

        # Check for castling (king moves 2 squares horizontally)
        is_castling = isinstance(piece, King) and abs(col - old_col) == 2

        # Check for en passant
        is_en_passant = (
            isinstance(piece, Pawn) and 
            old_col != col and  # Diagonal move
            self.chessboard[row][col] is None  # Target square is empty
        )

        # Handle en passant capture
        if is_en_passant and self.last_move:
            _, _, (last_to_col, last_to_row) = self.last_move
            # Remove the captured pawn
            self.chessboard[last_to_row][last_to_col] = None
            print(f"En passant capture at {(last_to_col, last_to_row)}")

        # Record this move for en passant on next turn
        self.last_move = (piece, (old_col, old_row), (col, row))

        # Execute the main move
        self.chessboard[row][col] = piece
        self.chessboard[old_row][old_col] = None

        # Handle castling - move the rook too
        if is_castling:
            if col > old_col:  # Kingside
                # Move rook from h-file to f-file
                rook_old_col = old_col + 3
                rook_new_col = old_col + 1
                self.chessboard[row][rook_new_col] = self.chessboard[row][rook_old_col]
                self.chessboard[row][rook_old_col] = None
                print(f"Kingside castle")
            else:  # Queenside
                # Move rook from a-file to d-file
                rook_old_col = old_col - 4
                rook_new_col = old_col - 1
                self.chessboard[row][rook_new_col] = self.chessboard[row][rook_old_col]
                self.chessboard[row][rook_old_col] = None
                print(f"Queenside castle")

            # Make sure to mark the rook as moved too
            self.chessboard[row][rook_new_col].has_moved = True

        # Mark the piece as moved
        piece.has_moved = True

        # Update python-chess Board
        try:
            # Convert custom move to UCI format for python-chess
            from_square = chess.square(old_col, old_row)
            to_square = chess.square(col, row)
            move = chess.Move(from_square, to_square)
            self.current_board.push(move)
        except Exception as e:
            print(f"Error updating python-chess Board: {e}")

        # Check for pawn promotion
        if self._check_for_promotion(piece, row):
            self.promotion_pawn = (col, row)
            # Don't switch turns yet - wait for promotion choice
            return True

        # Switch turns
        self.turn_color = 'b' if self.turn_color == 'w' else 'w'

        # Check for checkmate
        if self._is_check(self.turn_color):
            print(f"Check!")
            if self._is_checkmate(self.turn_color):
                winner = 'White' if self.turn_color == 'b' else 'Black'
                self.game_result = f"Checkmate! {winner} wins!"
                self.state = "game_over"
                print(self.game_result)

        self._print_board()
        return True

    def _handle_click(self, x, y):
        """Handle mouse clicks during the game"""
        # Ignore clicks outside the board
        if y >= self.board_size:
            return

        col, row = self._mouse_to_square(x, y)
        piece = self.chessboard[row][col]

        # First click: select piece
        if self.picked_piece is None:
            if piece and piece.color == self.turn_color:
                self.picked_piece = piece
                self.picked_pos = (col, row)
                moves = self.get_legal_moves(self.picked_pos)
                print(f"Selected {piece.symbol}{piece.color} at {self.picked_pos}, moves: {moves}")
            elif piece:
                print(f"It's {self.turn_color.upper()}'s turn.")

        # Second click: attempt move
        else:
            target = (col, row)
            moves = self.get_legal_moves(self.picked_pos)

            if target in moves:
                self._make_move(self.picked_pos, target)
            else:
                print(f"Illegal move to {target}.")

            # Reset selection
            self.picked_piece = None
            self.picked_pos = None

    def _draw_board(self):
        """Draw the chess board and pieces"""
        # Clear screen
        self.window.fill(self.BLACK)

        for r in range(8):
            for c in range(8):
                color = self.WHITE if (r + c) % 2 == 0 else self.BLACK
                rect = pygame.Rect(
                    c * self.square_size,
                    r * self.square_size,
                    self.square_size,
                    self.square_size
                )
                pygame.draw.rect(self.window, color, rect)

                # Highlight selected piece's possible moves
                if self.picked_pos and self.picked_piece:
                    moves = self.get_legal_moves(self.picked_pos)
                    if (c, r) in moves:
                        # Draw a circle to highlight possible moves
                        center = (c * self.square_size + self.square_size // 2,
                                  r * self.square_size + self.square_size // 2)
                        radius = self.square_size // 6
                        pygame.draw.circle(self.window, self.HIGHLIGHT, center, radius)

                # Draw pieces
                piece = self.chessboard[r][c]
                if piece:
                    key = piece.color + piece.symbol
                    img = self.images.get(key)
                    if img:
                        self.window.blit(img, rect)

        # Highlight selected piece
        if self.picked_pos:
            col, row = self.picked_pos
            rect = pygame.Rect(
                col * self.square_size,
                row * self.square_size,
                self.square_size,
                self.square_size
            )
            pygame.draw.rect(self.window, self.HIGHLIGHT, rect, 3)  # 3 pixel border

        # Draw game mode indicator
        mode_text = "Two Players" if self.mode == "2p" else "Player vs AI"
        mode_surface = self.font.render(mode_text, True, self.WHITE)
        self.window.blit(mode_surface, (10, self.board_size + 10))

        # Draw turn indicator
        turn_text = "White's Turn" if self.turn_color == 'w' else "Black's Turn"
        turn_surface = self.font.render(turn_text, True, self.WHITE)
        turn_rect = turn_surface.get_rect()
        turn_rect.right = self.window_width - 10
        turn_rect.top = self.board_size + 10
        self.window.blit(turn_surface, turn_rect)

        # Draw promotion selection if needed
        if self.promotion_pawn:
            self._draw_promotion_selection()

        # Skip move notation during training
        if not self.training_game:
            # Only draw move list if current_board is available
            if self.current_board is not None:
                move_list = self.current_board.move_stack
                for i, move in enumerate(move_list):
                    move_text = f"{i+1}. {move}"

            else:
                # Fallback to custom board if python-chess Board is missing
                move_list = list(self.chessboard)  # Replace with actual move tracking if needed
                for i, move in enumerate(move_list):
                    move_text = f"{i+1}. {move}"
                    #text = self.font.render(move_text, True, self.TEXT)
                    #self.window.blit(text, (self.window_width - 200, 50 + i * 30))

    def run(self):
        """
        Main loop: handle events, draw board, update display.
        """
        clock = pygame.time.Clock()
        running = True

        while running:
            # Handle AI's turn
            if self.state == "game" and self.mode == "ai" and self.turn_color == 'b' and not self.promotion_pawn:
                if self.ai is not None:
                    ai_move = self.ai.make_move(self)
                    if ai_move:
                        from_pos, to_pos = ai_move
                        self._make_move(from_pos, to_pos)
                        # Add a small delay to make AI moves visible
                        time.sleep(0.5)

            # Print training progress to terminal
            if self.state in ["self_learn", "pgn_learn"] and hasattr(self.ai, 'learning_progress'):
                print(f"[GAME] {self.ai.learning_progress}")

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.promotion_pawn:
                        # Handle promotion piece selection
                        for rect, piece_class in self.promotion_choices:
                            if rect.collidepoint(event.pos):
                                self._complete_promotion(piece_class)
                                break
                    elif self.state == "menu":
                        self._handle_menu_click(event.pos)
                    elif self.state == "ai_learning":
                        self._handle_ai_learning_click(event.pos)
                    elif self.state in ["self_learn", "pgn_learn"]:
                        self._handle_learning_screen_click(event.pos)
                    elif self.state == "game_over":
                        self._handle_game_over_click(event.pos)
                    elif self.state == "game" and (self.turn_color == 'w' or self.mode == "2p"):
                        # Only allow moves for the human player(s)
                        self._handle_click(*event.pos)

            # Draw the current screen
            if self.state == "menu":
                self._draw_menu()
            elif self.state == "ai_learning":
                self._draw_ai_learning_menu()
            elif self.state == "self_learn":
                self._draw_self_learn_screen()
            elif self.state == "pgn_learn":
                self._draw_pgn_learn_screen()
            elif self.state == "game":
                self._draw_board()
            elif self.state == "game_over":
                self._draw_board()
                self._draw_game_over()

            pygame.display.flip()
            clock.tick(60)  # Limit to 60 FPS


if __name__ == "__main__":
    Game().run()