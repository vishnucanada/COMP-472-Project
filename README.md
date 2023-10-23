# COMP-472-Project

Overview

Two players alternately make movements on a grid in this text-based strategy game. The game may be played with AI opponents or with two human players. The aim of the game is to beat the rival team's units and take the victory.


HOW TO PLAY

Follow these instructions to play the game:
- Use the command python ai_wargame.py to run the Python script.
- The maximum number of turns and the maximum length of time (in seconds) that AI players can take between movements will both need to be entered.
- The game will begin, and the console's first game board will be seen.
- Each move consists of choosing a source and destination point on the grid, and movements are made by players in succession.


CODE STRUCTURE

The code is structured into a number of classes and functions, each with a distinct function. Here is a summary of the key elements:
- Unit: Describes the types, health, and skills of gaming units.
- Coord: Provides techniques for working with coordinates and represents coordinates on the game board.
- CoordPair: This object represents a pair of coordinates that are frequently used to define a move or a region on the board.
- Options: Provides options for configuring the game, including the type of game, the depth to which the AI may search, and time restrictions.
- Stats: Compiles and presents information about the game's statistics, including ratings for each depth and overall playtime.
- Game: Displays the game's current status, including the participants, game board, and available options. It also includes techniques for playing the game, moving, and dealing with AI turns.
- Player: Enumerates the two players in the game: Attacker and Defender.
- UnitType: Lists the many categories of in-game troops.
- GameType: Lists many game kinds, such as Attacker vs. Defender, Attacker vs. Computer, Defender vs. Computer, and Computer vs. Computer.
- MAX_HEURISTIC_SCORE and MIN_HEURISTIC_SCORE: constants that reflect the maximum and least heuristic scores that are used to rate games.

The game loop and user input processing are handled at the program's entry point, the main() function.


GAME RULES
+ Each participant makes a move in turn as the game is played on a square grid.
+ There are several sorts of troops available to each player, including AI, Tech, Virus, Programme, and Firewall.
+ Depending on the type of unit and the game's regulations, players can transfer their troops to neighboring cells, engage enemy units in combat, or repair their own units.
+ The game may be played by two human players, a human player, and an artificial intelligence (AI) player, or two AI players.
+ When a certain number of turns have been completed, or when one player completely eliminates all of the units of the opposition, the game is over.


ARGUMENTS 

You can modify the game using a number of command-line options that the program accepts:

- --max_depth: Defines the maximum depth of the search for AI moves.
- --max_time: Defines the most time (in seconds) that AI players may spend planning their movements.
- --game_type: Define the type of game. There are three options: "auto," "attacker," "defender," and "manual."
- --broker: Use an unfinished game broker to access the game.
To change the game settings, use these arguments when running the script.




D2 Read Me

The users now have the option of choosing between multiple heuristics of e0 e1 e2
By writing 0 1 2
