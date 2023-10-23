
from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests
from queue import PriorityQueue

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000

cumulative_evals = 0  # Initialize the counter
cumulative_evaluations = {}
class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker

class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health : int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table : ClassVar[list[list[int]]] = [
        [-3,-3,-3,-3,-1], # AI
        [-1,-1,-6,-1,-1], # Tech
        [-9,-6,-1,-6,-1], # Virus
        [-3,-3,-3,-3,-1], # Program
        [-1,-1,-1,-1,-1], # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table : ClassVar[list[list[int]]] = [
        [0,1,1,0,0], # AI
        [3,0,0,3,3], # Tech
        [0,0,0,0,0], # Virus
        [0,0,0,0,0], # Program
        [0,0,0,0,0], # Firewall
    ]

    e0_evaluation : ClassVar[list[int]]=[
        9999, # AI 
        3, # Tech 
        3, # Virus
        3, # Program
        3 # Firewall
    ]
    

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta : int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"
    
    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()
    
    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount


    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

    def e0_evaluation_amount(self) -> int:
        """Similar to  repair amount, determine the heursitic amount"""
        return self.e0_evaluation[self.type.value]

##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row : int = 0
    col : int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
                coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
                coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()
    
    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()
    
    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist,self.row+1+dist):
            for col in range(self.col-dist,self.col+1+dist):
                yield Coord(row,col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row,self.col-1) #left
        yield Coord(self.row-1,self.col) #up
        yield Coord(self.row+1,self.col) #down
        yield Coord(self.row,self.col+1) #right
    

    @classmethod
    def from_string(cls, s : str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip() 
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()
    
    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row,self.dst.row+1):
            for col in range(self.src.col,self.dst.col+1):
                yield Coord(row,col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0,col0),Coord(row1,col1))
    
    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0,0),Coord(dim-1,dim-1))
    
    @classmethod
    def from_string(cls, s : str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################000
@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth : int | None = 4
    min_depth : int | None = 2
    max_time : float | None = 5.0
    game_type : GameType = GameType.AttackerVsDefender
    alpha_beta : bool = True
    max_turns : int | None = 100
    randomize_moves : bool = True
    broker : str | None = None
    heuristic : int | None = 0
##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth : dict[int,int] = field(default_factory=dict)
    total_seconds: float = 0.0

##############################################################################################################
class Tree:
    move: CoordPair
    game_clone: Game;
    
    def __init__(self, heuristic, move, game_clone, children=None):
        self.heuristic = heuristic
        self.move = move
        self.game_clone = game_clone
        self.children = children if children is not None else []

    def add_child(self, child: Iterable[Tree]):
        self.children.append(child)
        
   
        






@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played : int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai : bool = True
    _defender_has_ai : bool = True
    def retrieve_heuristic(self, depth)-> int:
        if depth not in cumulative_evaluations and depth >= 0:
            cumulative_evaluations[depth] = cumulative_evals
        elif depth >= 0:
            current_val = cumulative_evaluations[depth]
            cumulative_evaluations[depth] = current_val + cumulative_evals
        if self.options.heuristic == 1:
            return self.heuristic_one()
        elif self.options.heuristic == 2:
            return self.heuristic_two()
        else:
            return self.heuristic_zero()
    def create_file(self,b,t,m) -> str:
        b = str(b)
        t = str(t)
        m = str(m)
        file_name = "gameTrace-"+str(b)+"-"+t+"-"+m+".txt"
        f = open(file_name,"a")
        f.close()
        return str(file_name)
    
    
    def write_to_file(self,file_name):
        f = open(file_name,"a")
        entire_output = self.to_string()
        f.write(entire_output)
        f.close()

    def write_to_file_string(self,string_to_write):
        b = str(self.options.alpha_beta)
        t = str(self.options.max_time)
        m = str(self.options.max_turns)
        file_name = "gameTrace-"+str(b)+"-"+t+"-"+m+".txt"
        f = open(file_name,"a")
        f.write(string_to_write)
        f.close()


    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim-1
        self.set(Coord(0,0),Unit(player=Player.Defender,type=UnitType.AI))
        self.set(Coord(1,0),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(0,1),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(2,0),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(0,2),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(1,1),Unit(player=Player.Defender,type=UnitType.Program))
        self.set(Coord(md,md),Unit(player=Player.Attacker,type=UnitType.AI))
        self.set(Coord(md-1,md),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md,md-1),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md-2,md),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md,md-2),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md-1,md-1),Unit(player=Player.Attacker,type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord : Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord : Coord, unit : Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord,None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord : Coord, health_delta : int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords : CoordPair) -> bool:
        """Validate a move expressed as a CoordPair"""
        unit_src = self.get(coords.src)
        unit_dst = self.get(coords.dst)

        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst): 
            #If coords are not in the board
            return False
        if unit_src is None or unit_src.player != self.next_player: 
            #If current player is not using its entity (source)
            return False
        if unit_dst is not None and unit_dst is unit_src:
            #self-destruct only when there is your oponent around yourself
            for coord in coords.src.iter_range(1):  
                    unit = self.get(coord)
                    if unit is not None and unit.player is not self.next_player:
                        return True  

        if unit_dst is not None and unit_dst.health >= 9 and unit_dst.player == unit_src.player:
            #You cant repair if youre above 9
            return False
        if unit_src.type._value_ in [0, 3, 4]: 
            #Entity of either type AI, program or firewall
            
            if unit_dst is None: 
                #Check when player wants to move to an empty spot
                for coord in coords.src.iter_adjacent():
                    adjacent_coord = self.get(coord)
                    if adjacent_coord != None and unit_src.player != adjacent_coord.player: 
                        #Check if the adjacent unit is an adversarial unit. If true, engaged in combat so you cannot move
                        return False
                        
                if unit_src.player._value_ == Player.Attacker._value_: #Player: Attacker
                    for coord in coords.src.iter_adjacent(): 
                        if (coord == coords.dst) and ((coord.row < coords.src.row) or (coord.col < coords.src.col)): 
                            #Accepted adjacent moves are only in the up or left direction
                            return True  
                    return False
                
                if unit_src.player._value_ == Player.Defender._value_: #Player: Defender
                    for coord in coords.src.iter_adjacent(): 
                        if (coord == coords.dst) and ((coord.row > coords.src.row) or (coord.col > coords.src.col)): 
                            #Accepted adjacent moves are only in the down or right direction
                            return True  
                    return False
            else: 
                #When player wants to attack or repair an entity
                for coord in coords.src.iter_adjacent():  
                    if coord == coords.dst: 
                        #Any adjacent moves are accepted 
                        return True  
                return False


        if unit_src.type._value_ in [1, 2]: 
            #Entity of either type Tech or virus
            for coord in coords.src.iter_adjacent():  
                if coord == coords.dst: 
                    #Any adjacent moves are accepted 
                    return True  
            return False
        
        return True
    


    def perform_move(self, coords : CoordPair) -> Tuple[bool,str]:
        """Validate and perform a move expressed as a CoordPair"""
        if self.is_valid_move(coords):
            unit_src = self.get(coords.src)
            unit_dst = self.get(coords.dst)
            if unit_dst != None and coords.dst != coords.src and unit_src.player == unit_dst.player and unit_dst.health > 9:
                health_delta = unit_src.repair_amount(unit_dst)
                unit_dst.mod_health(health_delta)

            elif unit_dst != None and unit_dst.player != self.next_player:
                health_delta = unit_src.damage_amount(unit_dst)
                unit_dst.mod_health(health_delta)
                health_delta = unit_dst.damage_amount(unit_src)
                unit_src.mod_health(health_delta)
            elif unit_dst != None and unit_src.player == unit_dst.player and unit_src == unit_dst: #Self destruction
                for coord in coords.src.iter_range(1):  # Adjust the range as needed
                    unit = self.get(coord)
                    if unit is not None:
                        damage_amount = unit.damage_amount(unit_dst)
                        unit.mod_health(damage_amount)   
                self.mod_health(coords.src, -9)
            else:
                self.set(coords.dst,self.get(coords.src))
                self.set(coords.src,None)
            return (True,"")    
        return (False,"invalid move")


    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()
    
    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            self.write_to_file_string('Player '+self.next_player.name+' played move '+s+'\n')
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')
    
    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success,result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ",end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success,result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ",end='')
                    print(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success,result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ",end='')
                print(result)
                
                self.next_turn()
            if not success:
                print(f"Computer has entered an Invalid Move and thus lost the game ")
                if self.next_player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False
                return 
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord,Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord,unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker    
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src,_) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> CoordPair:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (move_candidates[0])
        else:
            return (None)

    def is_time_up(self, start_time)-> bool: #THIS WORKS
        current_time = (datetime.now() - start_time).total_seconds()
        return current_time >= self.options.max_time
    


    def potential_move(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        #Still need to work on this! will have to remove the while loop and add the commented section!
        no_valid_move = True
        player_src = []
        for (src,_) in self.player_units(self.next_player):
            player_src.append(src)
        while(no_valid_move):
            src = random.choice(player_src)
            # Iterate through all adjacent coordinates and yield valid moves
            for dst in src.iter_adjacent():
                move = CoordPair(src, dst)  # Create a CoordPair with the chosen source and adjacent destination
                if self.is_valid_move(move):
                    no_valid_move = False
                    yield move
    
    
    
    def find_tree(self, root, depth, max_depth, start_time) -> Tree:
        if depth == max_depth:
            return root
        
        game_clone = self.clone()

        
        root = Tree(game_clone.retrieve_heuristic(depth), None, game_clone, None)  
        
        

            
        possible_moves = list(game_clone.find_all_moves(start_time))

        for move in possible_moves:
            game2 = game_clone.clone()
            (valid, _) = game2.perform_move(move)
            
            if valid:
                
                
                child = self.find_tree(root, depth + 1, max_depth, start_time)
                child.move = move
                root.add_child(child)
        
        return root
        
    
    def find_all_moves(self, start_time)-> Iterable[CoordPair]:
        game_clone = self.clone()
        player_src = []
        for (src,_) in self.player_units(self.next_player):
                player_src.append(src)
        random.shuffle(player_src)
        for src in player_src:
            if game_clone.is_time_up(start_time):
                break
            for dst in src.iter_adjacent():
                move = CoordPair(src, dst)  # Create a CoordPair with the chosen source and adjacent destination
                if self.is_valid_move(move):
                    yield move
        
    
    

    def minimax(self, maximize, start_time, depth, node, best_score, move: CoordPair)-> Tuple[int, CoordPair, int]:
        game_clone = self.clone()
        if depth < 0 or game_clone.is_time_up(start_time) or game_clone.has_winner() or len(node.children) == 0:
           
            
            return (game_clone.retrieve_heuristic(depth), move, depth)
        
        if maximize:
            max_eval = MIN_HEURISTIC_SCORE
            best_move = None
            for child in node.children:
                
                (eval, _,_) = self.minimax(not maximize, start_time, depth -1,child, max_eval, move)
                if eval > max_eval:
                    max_eval = eval
                    best_move = child.move
            return (max_eval, best_move, depth)
        else:
            min_eval = MAX_HEURISTIC_SCORE
            best_move = None
            for child in node.children:
                (eval,_, _) = game_clone.minimax(not maximize, start_time, depth-1, child, min_eval, move)
                if eval < min_eval:
                    min_eval = eval
                    best_move = child.move
            return (min_eval, best_move, depth)


    

    def alpha_beta_pruning(self, maximize, start_time):
        def alphabeta(node, depth, alpha, beta, is_maximizing):
            if depth == 0 or game_clone.is_time_up(start_time) or game_clone.has_winner():
                
                return game_clone.retrieve_heuristic(depth), node.move, depth

            if is_maximizing:
                max_eval = float("-inf")
                best_move = None

                for child in node.children:
                    eval, _, _ = alphabeta(child, depth - 1, alpha, beta, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = child.move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return max_eval, best_move, depth
            else:
                min_eval = float("inf")
                best_move = None

                for child in node.children:
                    eval, _, _ = alphabeta(child, depth - 1, alpha, beta, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_move = child.move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval, best_move, depth

        alpha = float("-inf")
        beta = float("inf")
        best_move = None
        best_move_score = MIN_HEURISTIC_SCORE if maximize else MAX_HEURISTIC_SCORE

        game_clone = self.clone()
        player_src = []
        valid_move = None

        for (src, _) in self.player_units(self.next_player):
            player_src.append(src)

        random.shuffle(player_src)

        for src in player_src:
            if game_clone.is_time_up(start_time):
                break

            for dst in src.iter_adjacent():
                move = CoordPair(src, dst)

                if self.is_valid_move(move):
                    valid_move = move
                    break

            if valid_move is not None:
                break

        (new_score, first_move, depth) = alphabeta(
            game_clone.find_tree(valid_move, 0, 3, start_time), 3, alpha, beta, maximize)

        if maximize and best_move_score < new_score:
            best_move = first_move
            best_move_score = new_score
        elif not maximize and best_move_score > new_score:
            best_move = first_move
            best_move_score = new_score

        if best_move is None and best_move_score is None:
            best_move = game_clone.random_move()
            
            best_move_score = game_clone.retrieve_heuristic(depth)

        return best_move_score, best_move, depth


  

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()
        maximize = (self.next_player is Player.Attacker)
        game_clone = self.clone()
        
        current_score = 0
        chosen_move = None
        
        if self.options.alpha_beta:
            (score, move, avg_depth) = self.alpha_beta_pruning(maximize, start_time)
            
        else:
            # WE RUN MINIMAX ON MULTIPLE MOVES AND STATES 
            for move in game_clone.find_all_moves(start_time):
                if game_clone.is_time_up(start_time):
                    move = chosen_move
                    score = current_score
                    break
                node = game_clone.find_tree(move, 0,3,start_time)
                #weve created a tree based on a move and run minimax on that
                (score, move, avg_depth) = self.minimax(maximize,start_time,4,node,None,node.move)
                if maximize and score > MIN_HEURISTIC_SCORE:
                    current_score = score
                    chosen_move = move
                elif not maximize and score < MAX_HEURISTIC_SCORE:
                    current_score = score
                    chosen_move = move
            move = chosen_move
            score = current_score


        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        
        print(f"Heuristic score: {score}")
        print(f"cumulative evals: "+str(sum(cumulative_evaluations.values())))
        print(f"Average recursive depth: {avg_depth:0.1f}")
        print(f"Evals per depth: ",end='')

        for key in sorted(cumulative_evaluations.keys()):
            print(f"{key}:{cumulative_evaluations[key]} ",end='')
        print()
        
        total_evals = sum(cumulative_evaluations.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        print("Computer played move "+str(move))
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'],data['from']['col']),
                            Coord(data['to']['row'],data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None
     
    
    
    
    def fastest_heuristic_you_ever_seen(self):
        count_number_valid_moves = 0
        for coord_pair in self.move_candidates():
            count_number_valid_moves += self.number_of_valid_moves_but_faster(coord_pair)
        return count_number_valid_moves

        
    def number_of_valid_moves_but_faster(self,coords: CoordPair) -> int:
        """
        This is a faster way to check number of valid moves, because it does not account for attacks and repairs,
        only for probable positions to move too
        """
        if coords is not None:

            if coords.dst is not None:
                coord_row = coords.src.row
                coord_col = coords.src.col        
                array_of_valid_moves = [self.is_valid_coord(Coord(coord_row,coord_col-1)), self.is_valid_coord(Coord(coord_row-1,coord_col)),self.is_valid_coord(Coord(coord_row+1,coord_col)),self.is_valid_coord(Coord(coord_row,coord_col+1))] 
                
                return sum(array_of_valid_moves)
        return 0
    def number_of_valid_moves(self,coords : CoordPair) -> int:
        """
        For one of the heuristics, We want to know how many legal moves are possible, which also equates to position
        """
        #Up down left right, so it will only run a max of 4 times
        valid_moves = 0
        iteratable_coords = list(coords.src.iter_adjacent())
        for coordinate_to_move in coords.src.iter_adjacent():
            if self.is_valid_move(coordinate_to_move) :
                valid_moves = valid_moves + 1
        return valid_moves
    
    def compare_units(self, unit_you: Unit, unit_enemy:Unit) -> int:
        if unit_you is None or unit_enemy is None:
            return 0
        if unit_you.type == UnitType.AI and unit_enemy.type == UnitType.Virus:
            return -9999
        elif unit_you.type == UnitType.Virus and unit_enemy.type == UnitType.Tech:
            return -10
        elif unit_you.type == UnitType.Virus and unit_enemy.type == UnitType.AI:
            return 9999
        elif unit_you.type == UnitType.Virus and unit_enemy.type == UnitType.Program:
            return 1000
        elif unit_you.type == UnitType.Tech and unit_enemy.type == UnitType.Virus:
            return 100
        elif unit_you.type == UnitType.Firewall and (unit_enemy.type == UnitType.AI or unit_enemy.type == UnitType.Program or unit_enemy.type == UnitType.Virus or unit_enemy.type == UnitType.Tech):
            return 1
        else:
            return 0
   
    def heuristic_zero(self):
        global cumulative_evals  # Declare it as global
        heuristic_value = 0
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None:
                
                if unit.player == Player.Defender:
                    heuristic_value -=  unit.e0_evaluation_amount()
                    cumulative_evals += 1  # Increment the counter
                else:
                    heuristic_value += unit.e0_evaluation_amount()
                    cumulative_evals += 1  # Increment the counter
        return heuristic_value

    def heuristic_one(self):
        """
        In the beginning of the game POSITION Matters more than piece trading, so assuming weve gone less than a depth of 5, 
        use this heuristic
        """
        heuristic_value = 0
        global cumulative_evals  # Declare it as global
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if coord is not None and unit is not None:
                
                a1 = Coord(coord.row,coord.col-1) #left
                a2 = Coord(coord.row-1,coord.col) #up
                a3 = Coord(coord.row+1,coord.col) #down
                a4 = Coord(coord.row,coord.col+1) #right
                iterable_coordinates = [a1,a2,a3,a4]
                for new_coord in iterable_coordinates:
                    if self.is_valid_coord(new_coord):
                        if unit.player == Player.Defender:
                            heuristic_value -=  1
                            cumulative_evals += 1  # Increment the counter
                        else:
                            heuristic_value += 1
                            cumulative_evals += 1  # Increment the counter
        return heuristic_value


    def heuristic_two(self):
        """
        Endgame heuristic, once we get deeper in the game, pieces matter more than position as well as
        those pieces in relation to the pieces they are near
        """
        heuristic_value = 0
        global cumulative_evals  # Declare it as global
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            
            unit_src = self.get(coord)
            unit_dst = self.get(coord)
            if unit_src is not None and unit_dst is not None:
                if unit_src.player == Player.Defender:
                    heuristic_value += self.compare_units(unit_src,unit_dst)
                if unit_dst.player == Player.Attacker:
                    heuristic_value -= self.compare_units(unit_src,unit_dst)
                cumulative_evals += 1  # Increment the counter        
        return heuristic_value


##############################################################################################################

def main():

    # parse command line arguments
    max_turns = int(input("Please enter a maximum amount of turns allowed: "))
    max_time = int(input("Please enter a maximum amount(in seconds) that AI is allowed to take: "))
    alpha_value = str(input("Please enter whether or not you are using Alpha-Beta Pruning (True or False): "))
    game_mode = str(input("Please enter a specified game mode: \n attacker \n defender \n manual \n computer \n"))
    heur = int(input("Please enter a valid heuristic to use (e0 e1 e2) as 0 1 2? \n"))
    
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    args = parser.parse_args()

    args.game_type = game_mode
    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp 
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp
    
    # set up game options
    options = Options(game_type=game_type)


    options.max_turns = max_turns
    options.max_time = max_time
    if alpha_value.lower() == "false":
        options.alpha_beta = False
    else:
        options.alpha_beta = alpha_value
    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker
    options.heuristic = heur

    # create a new game
    game = Game(options=options)

    #CREATING THE FILE THAT WILL BE WRITTEN TO
    b = options.alpha_beta
    t = options.max_time
    m = options.max_turns
    
    file_name = game.create_file(b,t,m)
    # the main game loop
    while True:
        print()
        print(game)
        game.write_to_file(file_name)

        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)

##############################################################################################################

if __name__ == '__main__':
    main()
