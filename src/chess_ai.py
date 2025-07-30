import random
import os
import time
import numpy as np
import chess
import chess.pgn
import pygame
import threading
import io
import json
import ast

from collections import deque

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers, optimizers
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("TensorFlow not available. Using random moves.")

class ChessAI:
    def __init__(self, color='b'):
        self.color = color
        self.model = None
        self.learning_mode = "random"
        self.training_in_progress = False
        self.training_log = []
        self.learning_progress = ""

        # Check for TensorFlow availability
        self.TENSORFLOW_AVAILABLE = True
        try:
            import tensorflow as tf
            from tensorflow.keras import Sequential, layers, optimizers
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
            from tensorflow.keras.models import load_model
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            self.TENSORFLOW_AVAILABLE = False
            print("TensorFlow not available. Using random moves.")

        # Try to load existing model
        model_path = os.path.join("models", "chess_model.h5")
        if self.TENSORFLOW_AVAILABLE and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                self.learning_mode = "neural"
                self.learning_progress = "Loaded existing model."
            except Exception as e:
                print(f"Error loading model: {e}")
                self.learning_progress = "Failed to load model. Using random moves."

    def board_to_input(self, board):
        """
        Convert a chess board to neural network input format.
        
        Args:
            board: A 2D array representing the chess board
            
        Returns:
            A numpy array formatted for neural network input
        """
        # Create a 8x8x12 array (12 piece types, 8x8 board)
        input_array = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Map pieces to channels: 0-5 for white pieces, 6-11 for black pieces
        # Order: pawn, knight, bishop, rook, queen, king
        piece_to_channel = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # white pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # black pieces
        }
        
        # Convert the board to FEN for processing with python-chess
        fen = self._board_to_fen(board)
        chess_board = chess.Board(fen)
        
        # Fill the input array
        for square in chess.SQUARES:
            piece = chess_board.piece_at(square)
            if piece:
                row, col = 7 - (square // 8), square % 8  # chess.SQUARES goes from 0-63
                p_type = piece.piece_type - 1
                if piece.color:
                    channel = p_type
                else:
                    channel = p_type + 6
                input_array[row][col][channel] = 1
                
        return input_array.reshape(1, 8, 8, 12)  # Add batch dimension
    
    def _board_to_fen(self, board):
        """
        Convert our board representation to FEN string for python-chess.
        
        Args:
            board: A 2D array representing the chess board
            
        Returns:
            A FEN string representing the board
        """
        # Map our pieces to FEN characters
        piece_map = {
            'Pw': 'P', 'Nw': 'N', 'Bw': 'B', 'Rw': 'R', 'Qw': 'Q', 'Kw': 'K',
            'Pb': 'p', 'Nb': 'n', 'Bb': 'b', 'Rb': 'r', 'Qb': 'q', 'Kb': 'k'
        }
        
        fen_parts = []
        for row in board:
            empty_count = 0
            fen_row = ""
            
            for piece in row:
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    # Get piece symbol and color from our representation
                    symbol = piece.symbol + piece.color
                    fen_row += piece_map.get(symbol, '?')
            
            if empty_count > 0:
                fen_row += str(empty_count)
                
            fen_parts.append(fen_row)
        
        # Join with slashes and add other FEN components (turn, castling, etc.)
        # For simplicity, we'll assume some defaults for the non-piece parts
        fen = '/'.join(fen_parts)
        turn = 'w' if self.color == 'b' else 'b'  # Opposite of AI's color
        return f"{fen} {turn} KQkq - 0 1"
    
    def make_move(self, game):
        """
        Make a move based on current learning mode.
        
        Args:
            game: The Game object containing the current state
            
        Returns:
            tuple: (from_pos, to_pos) representing the move, or None if no moves available
        """
        # Check if there's a pawn waiting for promotion (AI always chooses Queen)
        if game.promotion_pawn and game.chessboard[game.promotion_pawn[1]][game.promotion_pawn[0]].color == self.color:
            from chess_server import Queen
            game._complete_promotion(Queen)
            return None
        
        # Use neural network if available and in neural mode
        if self.learning_mode == "neural" and self.model and self.TENSORFLOW_AVAILABLE:
            return self._make_neural_move(game)
        else:
            return self._make_random_move(game)
    
    def _make_random_move(self, game):
        """Make a random legal move"""
        possible_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = game.chessboard[row][col]
                if piece and piece.color == self.color:
                    moves = game.get_legal_moves((col, row))
                    for move in moves:
                        possible_moves.append(((col, row), move))
        
        # Return a random move if any are available
        if possible_moves:
            return random.choice(possible_moves)
        return None
    
    def _make_neural_move(self, game):
        """Make a move based on neural network evaluation"""
        if not self.model or not self.TENSORFLOW_AVAILABLE:
            return self._make_random_move(game)
            
        possible_moves = []
        for row in range(8):
            for col in range(8):
                piece = game.chessboard[row][col]
                if piece and piece.color == self.color:
                    moves = game.get_legal_moves((col, row))
                    for move in moves:
                        possible_moves.append(((col, row), move))
        
        if not possible_moves:
            return None
            
        # Evaluate each move with the neural network
        best_move = None
        best_score = float('-inf') if self.color == 'w' else float('inf')
        
        for from_pos, to_pos in possible_moves:
            # Create a copy of the board to simulate the move
            test_board = [row[:] for row in game.chessboard]
            from_col, from_row = from_pos
            to_col, to_row = to_pos
            
            # Make the move on the test board
            test_board[to_row][to_col] = test_board[from_row][from_col]
            test_board[from_row][from_col] = None
            
            # Evaluate the board
            board_input = self.board_to_input(test_board)
            score = self.model.predict(board_input, verbose=0)[0][0]
            
            # Choose the best move (highest score for white, lowest for black)
            if (self.color == 'w' and score > best_score) or (self.color == 'b' and score < best_score):
                best_score = score
                best_move = (from_pos, to_pos)
        
        # If neural network evaluation fails, fall back to random
        if best_move is None:
            return self._make_random_move(game)
            
        return best_move
    
    def create_model(self):
        if not self.TENSORFLOW_AVAILABLE:
            return None

        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Add

        inputs = Input(shape=(8, 8, 12))

        # Residual Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(64, (3, 3), activation='linear', padding='same')(x)
        shortcut = Conv2D(64, (1, 1), padding='same')(inputs)  # 1x1 projection
        x = Add()([x, shortcut])
        x = MaxPooling2D((2, 2))(x)

        # Residual Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='linear', padding='same')(x)
        shortcut = Conv2D(128, (1, 1), padding='same')(x)  # Identity shortcut
        x = Add()([x, shortcut])
        x = MaxPooling2D((2, 2))(x)

        # Final Layers
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
        return model
    
    def train_from_pgn(self, callback=None):
        if self.training_in_progress:
            print("[DEBUG] Training is already in progress, aborting.")
            return False
    
        self.training_in_progress = True
        self.training_log = []
        self.learning_progress = "Training from PGNs..."
    
        def safe_parse_evals(header_str):
            try:
                cleaned = header_str.strip().strip('"').replace('\\', '')
                return json.loads(cleaned)
            except Exception as e:
                print(f"[DEBUG] JSON parse failed: {e}")
                try:
                    return ast.literal_eval(cleaned)
                except Exception as e2:
                    print(f"[DEBUG] literal_eval also failed: {e2}")
                    return []
    
        def train_thread():
            try:
                print("[DEBUG] Starting training thread...")
    
                if not self.TENSORFLOW_AVAILABLE:
                    print("[DEBUG] TensorFlow not available.")
                    self.training_log.append("TensorFlow not available. Cannot train model.")
                    self.training_in_progress = False
                    return
    
                db_dir = "filtered_db"
                if not os.path.exists(db_dir):
                    print(f"[DEBUG] Directory {db_dir} not found.")
                    self.training_log.append(f"Directory {db_dir} not found.")
                    self.training_in_progress = False
                    return
    
                pgn_files = [f for f in os.listdir(db_dir) if f.endswith('.pgn')]
                print(f"[DEBUG] Found {len(pgn_files)} PGN files.")
                if not pgn_files:
                    self.training_log.append("No PGN files found in the filtered_db directory.")
                    self.training_in_progress = False
                    return
    
                if self.model is None:
                    print("[DEBUG] No model loaded, creating new model...")
                    self.model = self.create_model()
                    if self.model is None:
                        print("[DEBUG] Model creation failed.")
                        self.training_log.append("Failed to create model. Cannot train.")
                        self.training_in_progress = False
                        return
                    print("[DEBUG] Model created.")
    
                X, y = [], []
                games_processed = 0
                positions_collected = 0
    
                for pgn_file in pgn_files:
                    print(f"[DEBUG] Opening {pgn_file}")
                    file_path = os.path.join(db_dir, pgn_file)
    
                    with open(file_path, encoding='utf-8') as f:
                        while True:
                            game = chess.pgn.read_game(f)
                            if game is None:
                                print(f"[DEBUG] End of {pgn_file}")
                                break
                            
                            moves = list(game.mainline_moves())
                            print(f"[DEBUG] Game with {len(moves)} moves")
    
                            if len(moves) < 2:
                                print(f"[DEBUG] Skipping very short game.")
                                continue
                            
                            result = game.headers.get("Result", "*")
                            if result not in ["1-0", "0-1", "1/2-1/2"]:
                                print(f"[DEBUG] Invalid or missing result: {result}")
                                continue
                            
                            raw_eval_str = game.headers.get('StockfishEvals', '[]')
                            print(f"[DEBUG] Raw StockfishEvals header: {raw_eval_str[:80]}...")
    
                            evals = safe_parse_evals(raw_eval_str)
                            evals = [e if isinstance(e, (int, float)) and e != -1 else 0 for e in evals]
                            print(f"[DEBUG] Parsed {len(evals)} evals: {evals[:10]}")
    
                            if not evals:
                                print("[DEBUG] Skipping game due to no usable evaluations.")
                                continue
                            
                            board = chess.Board()
                            for i, move in enumerate(moves):
                                print(f"[DEBUG] Move {i}: {move}")
                                try:
                                    board.push(move)
                                except Exception as e:
                                    print(f"[DEBUG] Failed to push move {move}: {e}")
                                    break
                                
                                board_array = np.zeros((8, 8, 12), dtype=np.float32)
                                for sq in chess.SQUARES:
                                    piece = board.piece_at(sq)
                                    if piece:
                                        r, c = 7 - (sq // 8), sq % 8
                                        p_type = piece.piece_type - 1
                                        channel = p_type + (0 if piece.color else 6)
                                        board_array[r, c, channel] = 1
    
                                eval_idx = min(i, len(evals) - 1)
                                score = evals[eval_idx] / 100.0
                                print(f"[DEBUG] Eval idx {eval_idx}, Score {score}")
    
                                X.append(board_array)
                                y.append(score)
                                positions_collected += 1
    
                            print(f"[DEBUG] Finished game, collected {positions_collected} total positions so far.")
                            games_processed += 1
    
                print(f"[DEBUG] Training prep complete. {games_processed} games, {positions_collected} positions.")
    
                if not X or not y:
                    print("[DEBUG] No data collected. X or y is empty.")
                    self.training_log.append("No data collected. Training aborted.")
                    self.learning_progress = "No data collected. Training aborted."
                    self.training_in_progress = False
                    return
    
                X = np.array(X)
                y = np.array(y)
    
                self.learning_progress = f"Training on {len(X)} positions from {games_processed} games..."
    
                # === TRAINING CONFIGURATION ===
                from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
                early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                checkpoint = ModelCheckpoint("models/best_model.h5", save_best_only=True, monitor='val_loss')
    
                print("[DEBUG] Starting training...")
                history = self.model.fit(
                    X, y,
                    epochs=50,
                    batch_size=128,
                    validation_split=0.1,
                    callbacks=[early_stopping, checkpoint],
                    verbose=1
                )
    
                final_loss = history.history['val_loss'][-1]
                print(f"[DEBUG] Training complete: final val_loss={final_loss:.4f}")
                self.training_log.append(f"Training complete: final val_loss={final_loss:.4f}")
                self.learning_progress = f"Training complete: final val_loss={final_loss:.4f}"
    
            except Exception as e:
                print(f"[DEBUG] Training failed: {e}")
                self.training_log.append(f"Training failed: {e}")
                self.learning_progress = f"Training failed: {e}"
            finally:
                self.training_in_progress = False
                print("[DEBUG] Training thread finished.")
    
        threading.Thread(target=train_thread).start()
        return True






    def train_self_play(self, num_games=100, callback=None):
        """
        Train the neural network through self-play.
        
        Args:
            num_games: Number of games to play
            callback: Function to call with progress updates
        """
        if self.training_in_progress:
            return False
            
        self.training_in_progress = True
        self.training_log = []
        self.self_play_results = []
        
        def self_play_thread():
            if not self.TENSORFLOW_AVAILABLE:
                self.training_log.append("TensorFlow not available. Cannot train model.")
                self.training_in_progress = False
                return
                
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Create or load the model
            if self.model is None:
                self.model = self.create_model()
                if self.model is None:
                    self.training_log.append("Failed to create model.")
                    self.training_in_progress = False
                    return
            
            self.training_log.append(f"Starting self-play training with {num_games} games")
            
            # Experience replay buffer
            experiences = deque(maxlen=10000)
            
            for game_num in range(num_games):
                # Initialize a new game
                board = chess.Board()
                
                # Store positions and who-moved for this game
                game_positions = []
                mover_colors = []
                
                while not board.is_game_over():
                    # Current player color
                    current_color = chess.WHITE if board.turn else chess.BLACK
                    
                    # Get legal moves
                    legal_moves = list(board.legal_moves)
                    
                    if not legal_moves:
                        break
                        
                    # Choose a move with some exploration
                    if random.random() < 0.1:  # 10% exploration
                        move = random.choice(legal_moves)
                    else:
                        # Use the current model to evaluate moves
                        best_move = None
                        best_value = float('-inf') if current_color else float('inf')
                        
                        for move in legal_moves:
                            # Make the move on a copy of the board
                            board_copy = board.copy()
                            board_copy.push(move)
                            
                            # Convert to input format
                            board_array = np.zeros((8, 8, 12), dtype=np.float32)
                            
                            for square in chess.SQUARES:
                                piece = board_copy.piece_at(square)
                                if piece:
                                    row, col = 7 - (square // 8), square % 8
                                    p_type = piece.piece_type - 1
                                    if piece.color:
                                        channel = p_type
                                    else:
                                        channel = p_type + 6
                                    board_array[row][col][channel] = 1
                            
                            # Evaluate position
                            value = self.model.predict(board_array.reshape(1, 8, 8, 12), verbose=0)[0][0]
                            
                            # Update best move
                            if (current_color and value > best_value) or (not current_color and value < best_value):
                                best_value = value
                                best_move = move
                        
                        move = best_move if best_move else random.choice(legal_moves)
                    
                    # Store the current position
                    board_array = np.zeros((8, 8, 12), dtype=np.float32)
                    for square in chess.SQUARES:
                        piece = board.piece_at(square)
                        if piece:
                            row, col = 7 - (square // 8), square % 8
                            p_type = piece.piece_type - 1
                            if piece.color:
                                channel = p_type
                            else:
                                channel = p_type + 6
                            board_array[row][col][channel] = 1
                    
                    game_positions.append(board_array)
                    mover_colors.append(current_color)
                    
                    # For visualization in the UI
                    if game_num < 10:  # Only store details for a few games to save memory
                        self.self_play_results.append({
                            'game': game_num,
                            'move': str(move),
                            'board': board.fen()
                        })
                
                # Game is over, determine result
                if board.is_checkmate():
                    result = 1.0 if not board.turn else -1.0  # Winner is opposite of current turn
                else:
                    result = 0.0  # Draw
                
                # Add positions to experience replay with the game result
                for i, (position, color) in enumerate(zip(game_positions, mover_colors)):
                    # The "responsibility" for the result increases as the game progresses
                    progress = min(1.0, i / 40.0)
                    position_value = result if color else -result  # Flip for black
                    experiences.append((position, position_value * progress))
                
                self.training_log.append(f"Game {game_num+1}/{num_games} complete. Result: {result}")
                if callback:
                    callback(f"Game {game_num+1}/{num_games}")
                
                # Train on a batch of experiences every few games
                if (game_num + 1) % 5 == 0 and experiences:
                    # Sample batch from experiences
                    batch_size = min(256, len(experiences))
                    batch = random.sample(experiences, batch_size)
                    X_batch = np.array([exp[0] for exp in batch])
                    y_batch = np.array([exp[1] for exp in batch])
                    
                    # Train model
                    self.model.fit(X_batch, y_batch, epochs=1, verbose=0)
                    self.training_log.append(f"Trained on batch of {batch_size} positions")
            
            # Save the model
            self.model.save(os.path.join("models", "chess_model.h5"))
            
            self.learning_mode = "neural"
            self.training_in_progress = False
            
        # Start training in a separate thread
        threading.Thread(target=self_play_thread).start()
        return True

    def parse_pgn(self, pgn_path):
        import ast
        from chess import pgn

        games = []
        with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
            while True:
                game = pgn.read_game(pgn_file)
                if not game:
                    break

                # Extract headers
                stockfish_evals_str = game.headers.get('StockfishEvals', '[]')
                stockfish_best_moves_str = game.headers.get('StockfishBestMoves', '[]')

                # Clean and parse evals using ast.literal_eval
                stockfish_evals_str = stockfish_evals_str.strip().strip('\"')
                stockfish_evals_str = stockfish_evals_str.replace('\\', '')
                try:
                    evals = ast.literal_eval(stockfish_evals_str)
                except Exception as e:
                    print(f"[GAME] Failed to parse StockfishEvals in {pgn_path}: {e}")
                    continue

                # Clean and parse best moves using json.loads
                stockfish_best_moves_str = stockfish_best_moves_str.strip().strip('\"')
                stockfish_best_moves_str = stockfish_best_moves_str.replace('\\', '')
                try:
                    best_moves = json.loads(stockfish_best_moves_str)
                except json.JSONDecodeError as e:
                    print(f"[GAME] Failed to parse StockfishBestMoves in {pgn_path}: {e}")
                    continue

                # Validate lengths
                if len(evals) != len(best_moves):
                    print(f"[GAME] Mismatched lengths in {pgn_path}: evals={len(evals)}, moves={len(best_moves)}")
                    continue

                # Process game data
                board = game.board()
                positions = []
                for move, eval in zip(best_moves, evals):
                    try:
                        board.push_uci(move)  # Use push_uci for UCI notation
                    except ValueError as e:
                        print(f"[GAME] Invalid UCI move '{move}': {e}")
                        continue
                    positions.append(self.board_to_input(board))
                games.extend(positions)

        return games
