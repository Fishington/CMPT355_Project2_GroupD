# CMPT355_Project2_GroupD
This repo contains the code for Project 2 for CMPT 355 
## Overview
This project implements Konane (Hawaiian Checkers) with a weighted artificial intelligence system. The game uses minimax algorithm with alpha-beta pruning to evaluate board positions based on strategic piece weights.

## Features
- Full Konane game implementation
- Weighted piece evaluation system
- AI opponent using minimax with alpha-beta pruning
- Interactive command-line interface

## Project Structure
- `KonaneWithWeights.cpp` - Main game logic and board management
- `drivercheck.pl` - Game driver

## How to Run
```bash
./drivercheck.pl PlayerA PlayerB
```
Where Player A will play Black and Player B will play white.

## Game Rules
Konane is a two-player strategy game played on an 8×8 board. Players alternate removing opponent pieces by jumping over them, similar to checkers. The player who cannot move loses.

## Contributors
Wyatt Happer, Scott Pearson, Aidan Fisher, & Boden Smereka
