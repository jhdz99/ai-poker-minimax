# AI Poker (Texas Hold'em)

A Texas Hold'em poker game built in Python with a graphical interface and an AI opponent powered by **Determinized Minimax with Alpha-Beta pruning**.

## Features

- Heads-up Texas Hold'em
- AI opponent using game-tree search
- Determinization to handle hidden information
- Alpha-Beta pruning for efficient decision making
- Animated card dealing using Pygame
- Adjustable raise sizing and all-in support
- Persistent stacks between hands
- Showdown analysis explaining win/loss

## Technologies

- Python
- Pygame
- Minimax Algorithm
- Alpha-Beta Pruning

## Run

```bash
pip install -r requirements.txt
python -m src.main

## Project Structure

src/
  ai/        # AI logic (minimax, evaluation)
  game/      # Poker engine
  poker/     # Cards and hand evaluation
  ui/        # Pygame interface
assets/
  cards/     # Card images


## Future Improvements

  - Monte Carlo Tree Search AI
  - Poker hand strength evaluation model
  - Multiplayer networking
  - Chip animations