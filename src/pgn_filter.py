import chess
import chess.pgn
import chess.engine
import os
import json
import random
#from python_chess_utils import analyze_game_with_stockfish  # Placeholder for helper function

def filter_and_annotate_pgn(input_dir="unfiltered_db", output_dir="filtered_db", engine_path="src/stockfish/stockfish-windows-x86-64-avx2.exe", keep_unfiltered_ratio=0.1, callback=None):
    """
    Filter and annotate PGN files in the input directory.
    
    Args:
        input_dir: Directory with raw PGN files
        output_dir: Directory to save filtered and annotated PGNs
        engine_path: Path to Stockfish executable
        keep_unfiltered_ratio: Fraction of games to keep unfiltered for diversity
        callback: Optional function to report progress
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except Exception as e:
        print(f"Error initializing Stockfish: {e}")
        if callback:
            callback(f"Error initializing Stockfish: {e}")
        return

    total_games = 0
    processed_games = 0

    for pgn_file in os.listdir(input_dir):
        if not pgn_file.endswith('.pgn'):
            continue
        total_games += 1

    try:
        for idx, pgn_file in enumerate(os.listdir(input_dir)):
            if not pgn_file.endswith('.pgn'):
                continue

            input_path = os.path.join(input_dir, pgn_file)
            output_path = os.path.join(output_dir, pgn_file)

            with open(input_path, encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
                while True:
                    game = chess.pgn.read_game(f_in)
                    if game is None:
                        break

                    # Skip games with invalid results or too short
                    result = game.headers.get("Result", "*")
                    if result not in ["1-0", "0-1", "1/2-1/2"] or len(list(game.mainline())) < 10:
                        continue

                    # Randomly keep some unfiltered games for diversity
                    if random.random() < keep_unfiltered_ratio:
                        exporter = chess.pgn.FileExporter(f_out)
                        game.accept(exporter)
                        continue

                    # Analyze game with Stockfish
                    board = game.board()
                    evaluations = []
                    best_moves = []

                    for move in game.mainline_moves():
                        board.push(move)
                        result = engine.analyse(board, chess.engine.Limit(depth=5))  # Reduced from 15 to 5
                        score = result.get("score")
                        best_move = result.get("pv", [None])[0]

                        if score:
                            try:
                                if score.is_mate():
                                    evaluations.append(-score.white().mate())
                                else:
                                    evaluations.append(score.white().score())
                            except Exception as e:
                                print(f"Error extracting score: {e}")
                                evaluations.append(0)
                        else:
                            evaluations.append(0)

                        best_moves.append(str(best_move) if best_move else None)

                    # Store evaluations and best moves in custom headers
                    game.headers["StockfishEvals"] = json.dumps(evaluations)
                    game.headers["StockfishBestMoves"] = json.dumps(best_moves)

                    # Write the annotated game
                    exporter = chess.pgn.FileExporter(f_out)
                    game.accept(exporter)

                    processed_games += 1
                    if callback:
                        callback(f"Preprocessing PGNs: {processed_games}/{total_games} processed")

    except Exception as e:
        print(f"Error during PGN filtering: {e}")
        if callback:
            callback(f"Error during PGN filtering: {e}")

    finally:
        engine.quit()

    if callback:
        callback("PGN preprocessing complete.")
