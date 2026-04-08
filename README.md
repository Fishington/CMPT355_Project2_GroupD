
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

```bash

./drivercheck.pl  PlayerA  PlayerB

```

Where Player A will play Black and Player B will play white.

Replace either of the players with the konane_GroupD

## Game Rules

Konane is a two-player strategy game played on an 8×8 board. Players alternate removing opponent pieces by jumping over them, similar to checkers. The player who cannot move loses.

## Contributors

Wyatt Happer, Scott Pearson, Aidan Fisher, & Boden Smereka