# Chess AI Trainer with Pygame

This project is a self-learning chess AI built in Python. You can train it using PGN files or let it improve through self-play. It also includes a simple Pygame interface for playing against the AI.

## Features

- PGN-based training system  
- Self-play learning mode  
- Simple Pygame GUI to play against the AI  
- Neural network-based move selection  
- Optional Stockfish integration for evaluation  
- Organized codebase for easy extension or modification  

## Technologies Used

- Python 3  
- Pygame  
- NumPy  
- TensorFlow

## Project Structure

chess/  
├── .venv/                  # Virtual environment (ignored in repo)  
├── .vscode/                # VSCode settings  
├── filtered_db/            # Filtered PGN files  
│   └── lichess_db.pgn  
├── unfiltered_db/          # Raw PGN files  
│   └── lichess_db.pgn  
├── models/                 # Saved model weights (optional)  
├── src/                    # Source code  
│   ├── __pycache__/        # Cache files (ignored)  
│   ├── images/             # Pygame UI assets  
│   ├── stockfish/          # Stockfish integration or binary  
│   ├── chess_ai.py         # Neural network and AI logic  
│   ├── chess_server.py     # Game engine and interface  
│   ├── pgn_filter.py       # PGN file filtering  
│   └── zstdecomp.py        # PGN decompression (.zst files)  
├── requirements.txt        # All modules you need  
└── README.md               # Project info  

## Setup

1. Clone the repository:  
   `git clone https://github.com/your-username/chess-ai.git`  
   `cd chess-ai`  

2. Install dependencies:  
   `pip install -r requirements.txt`  

3. (Optional) Add a `models/` folder with your trained model if available.  

## Usage

- To launch the full game interface with AI:  
  `python src/chess_server.py`  
  - This runs the full Pygame UI  
  - Lets you play against the AI  
  - Includes built-in training options:  
    - Train from PGN file  
    - Train via self-play  
  - Uses and updates the model automatically  

- (Optional) Manual training from PGN:  
  `python src/chess_ai.py --train-from-pgn filtered_db/lichess_db.pgn`  

- (Optional) Manual self-play training:  
  `python src/chess_ai.py --self-play`  

- (Optional) Filter large PGN files:  
  `python src/pgn_filter.py --input unfiltered_db/lichess_db.pgn --output filtered_db/lichess_db.pgn`  

## Notes

- The repo does **not** include a pre-trained model due to size. You can train your own using the UI or scripts.  
- Lichess `.zst` files can be decompressed using `zstdecomp.py`.  

## License

MIT License — free to use and modify for personal or commercial projects.
