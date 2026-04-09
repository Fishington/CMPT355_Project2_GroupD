
# CMPT355_Project2_GroupD

This repo contains the code for Project 2 for CMPT 355

## Overview

This project implements Konane (Hawaiian Checkers) with a weighted artificial intelligence system. The game uses a minimax algorithm with alpha-beta pruning to evaluate board positions based on strategic piece weights.

## Features

- Full Konane game implementation

- Weighted piece evaluation system

- AI agent using minimax with alpha-beta pruning

## Project Structure

-  `KonaneWithWeights.cpp` - Main game logic and board management

-  `drivercheck.pl` - Game driver

-  `Demo` - Folder that holds all files that were used for the in-class demo

-  `makefile`- the file responsible for compiling our code

## Compiling our Program

```bash

make

```

This will result in a compiled version of our code called konane_GroupD

## How to Run

### Our Program With the Driver
```bash

perl drivercheck.pl PlayerA PlayerB

```
Where Player A will play Black and Player B will play white.

Replace either of the players with the konane_GroupD

### Our Standalone Program 
```bash
./konane_GroupD <board_state_file> <agent_colour>
```
Where board_state_file can be replaced with a file representing the game state

Where agent_colour can be either B (black) or W (white)

## Game Rules

Kōnane is a two-player strategy board game played on a 8x8 grid. One player controls the black pieces, while the other controls the white pieces. The objective of the game is to leave the opponent with no valid moves. When a player is unable to make a legal move on their turn, they lose the game, and their opponent is declared the winner.

A valid move in Kōnane consists of a player “jumping” one of their pieces over an adjacent opponent’s piece, provided that the square immediately beyond the opponent’s piece, in the same direction of the jump, is empty. For example, in chess notation, if a player has a piece on square A5 and the opponent has a piece on A6, the player can jump over A6 and land on A7, only if A7 is unoccupied. The jumped piece is then removed from the board. This jumping mechanic is similar to that found in checkers; however, unlike checkers, Kōnane only allows jumps in four directions: up, down, left, and right, as diagonal moves are not permitted.


## Contributors

Wyatt Happer, Scott Pearson, Aidan Fisher, & Boden Smereka