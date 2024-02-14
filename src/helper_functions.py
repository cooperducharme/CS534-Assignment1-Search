import csv
from pickle import NONE
import sys
import os
import copy
import itertools
import math
import random
import time

starting_temp = 100
temp_multiplier = 0.95


#Node
class Node_hillclimb():
    """A node class for Hill Climbing"""

    def __init__(self, parent, board, move, h, path_cost=0):
        """Node initialization
        Args:
            parent (Node or None): Parent Node.
            board (2D array): Board content.
            move (list): List of moves in form of [tile,(dx,dy)], pass in [] if no move
            prev_path_cost (int, optional): TODO _description_. Defaults to 0.
        """
        self.parent = parent
        self.board = board
        self.move = move
        self.heuristic = h
        if len(self.move) > 0:
            self.path_cost = path_cost
        else:
            self.path_cost = 0

    def board(self):
        return self.board

    def heuristic(self):
        return self.heuristic
    
    def path_cost(self):
        return self.path_cost

    def astar(self):
        return self.heuristic + self.path_cost
    
    def show_moves(self):
        moves = []
        cur_node = self
        while (cur_node.parent):
            cur_move_str = move_to_string(cur_node.move)
            moves = [cur_move_str] + moves
            cur_node = cur_node.parent
        print("Sequence of moves:",moves)
        print("Number of moves:", len(moves))
        return moves
    
    def __eq__(self, other):
        """Check if two Nodes have the same board
        Args:
            other (Node): Node to compare from
        Returns:
            Boolean: Return True if two boards are the same.
            False otherwise
        """
        if other == None:
            return False
        self_board = self.board
        other_board = other.board
        n = len(self_board)
        for row in range(n):
            for col in range(n):
                self_tile = self_board[row][col]
                other_tile = other_board[row][col]
                if self_tile != other_tile: return False
        return True

#read_file
def read_file(file_path):
    """Read the board file from csv to memory.
    Return the board content and number of B letters.
    Args:
        file_path (String): The path string for input board csv file
    Returns:
        data: 2D array of board file
        B_count: number of letter B in the board
    """
    datafile = open(file_path, 'r', encoding="utf-8-sig")
    datareader = csv.reader(datafile, delimiter=',')
    data = []
    B_count = 0
    for row in datareader:
        cur_row = []
        for element in row:
            if element.isnumeric():
                cur_row.append(int(element))
            else:
                cur_row.append(element)
                B_count += 1
        data.append(cur_row)    

    return data, B_count

#valid_moves
def valid_moves(board):
    """Get the valid moves for the current board.
    Args:
        board (2D array): content of the board
    Returns:
        list: list of valid moves on the board.
        Each entry in the list is a list of two parts.
        The first part is the number e.g. 1-9.
        The second part is the move direction in form of (dx,dy)
    """
    n = len(board)
    list_of_B_coords = []
    for row in range(n):
        for col in range(n):
            cur_tile = board[row][col]
            if not isinstance(cur_tile, str):
                continue
            list_of_B_coords.append((row,col))
    # print('list_of_B_coords',list_of_B_coords)
    list_of_moves = []
    for (row_cur,col_cur) in list_of_B_coords:
        for (dx, dy) in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            if inbound(row_cur + dx, col_cur + dy, n):
                if not isinstance(board[row_cur + dx][col_cur + dy],str):
                    list_of_moves.append([board[row_cur + dx][col_cur + dy], (0 - dx, 0 - dy)])
    return list_of_moves

#inbound
def inbound(row,col,n):
    """check if the coordinate is within the nxn matrix
    Args:
        row (int): row index
        col (int): col index
        n (int): n of the nxn board
    Returns:
        boolean : True if the coordinate is inbound,
        False otherwise
    """
    return 0<=row<n and 0<=col<n

#get_moved_board
def get_moved_board(board, move):
    """get a new board from the previous board
    and the given move of the tile
    Args:
        board (2D array): Current board layout
        move (list): Move is a list of two parts.
        The first part is the number e.g. 1-9.
        The second part is the move direction in form of (dx,dy)
    """
    tile = move[0]
    (dx,dy) = move[1]
    n = len(board)
    tile_x = -1
    tile_y = -1
    for row in range(n):
        break_flag = False
        for col in range(n):
            cur_tile = board[row][col]
            if not (tile == cur_tile):
                continue
            tile_x = row
            tile_y = col
            break_flag = True
            break
        if break_flag: break
    
    modified_board = copy.deepcopy(board)

    modified_board[tile_x][tile_y] = 'B'
    modified_board[tile_x+dx][tile_y+dy] = tile
    return modified_board

#move_to_string
def move_to_string(move):
    """change the list formate move to readable string
    Args:
        move (list): Move is a list of two parts.
        The first part is the number e.g. 1-9.
        The second part is the move direction in form of (dx,dy)
    Returns:
        string: string representation of move
    """
    (tile,direction) = move
    move_rep = {
        (0,1): 'right',
        (0,-1): 'left',
        (1,0): 'down',
        (-1,0): 'up'
    }
    direction_str = move_rep[direction]
    return str(tile) + ' ' + direction_str

#goal_states
def goal_states(board,B_count):
    """Generate the goal states for the current board.
    Also generate dictionaries of those goal states.
    Args:
        board (2D array): content of the board
        B_count (int): number of letter B in the board
    Returns:
        goal_board1_2d: goal board with B in the front
        goal_board2_2d: goal board with B at the end
        dict1: dictionary representation of goal_board1
        dict2: dictionary representation of goal_board2
    """
    n=len(board)
    board_1d = list(itertools.chain.from_iterable(board))
    board_1d = sorted([e for e in board_1d if e != 'B'])
    list_of_B = ['B'] * B_count
    goal_board1_1d = list_of_B + board_1d
    goal_board2_1d = board_1d + list_of_B
    goal_board1_2d = []
    goal_board2_2d = []
    for i in range(n):
        goal_board1_2d.append(goal_board1_1d[i*n:i*n+n])
        goal_board2_2d.append(goal_board2_1d[i*n:i*n+n])
    # print('goal_board1_2d',goal_board1_2d) 
    # print('goal_board2_2d',goal_board2_2d) 
    
    dict1 = goal_board_dicts(goal_board1_2d)
    dict2 = goal_board_dicts(goal_board2_2d)
    return goal_board1_2d, goal_board2_2d, dict1, dict2

#goal_board_dicts
def goal_board_dicts(goal_board):
    """Helper function to generate dictionary representation of goal board
    Args:
        goal_board (2D array): content of the goal board
    """
    n=len(goal_board)
    dict_goal = {}
    for row in range(n):
        for col in range(n):
            cur_tile = goal_board[row][col]
            if isinstance(cur_tile, str):
                continue
            dict_goal[cur_tile] = (row,col)
    # print('dict_goal',dict_goal)
    return dict_goal

#heuristic_function
def heuristic_function(board,goal_dict1,goal_dict2):
    """Get the heuristic value for the static board with no moves.
    Args:
        board (2D array): content of the board
        goal_dict (dictionary): key: tile of the board, value: corrdinate pair
    Returns:
        int: heuristic value for the current board
    """
    n = len(board)
    heuristic_1 = 0
    heuristic_2 = 0
    for row in range(n):
        for col in range(n):
            cur_tile = board[row][col]
            if isinstance(cur_tile, str):
                continue
            (dest_x,dest_y) = goal_dict1.get(cur_tile)
            heuristic_1 += cur_tile*(abs(dest_x-row) + abs(dest_y-col))
            (dest_x,dest_y) = goal_dict2.get(cur_tile)
            heuristic_2 += cur_tile*(abs(dest_x-row) + abs(dest_y-col))
    return min(heuristic_1,heuristic_2)

#same_board
def same_board(board1,board2):
    """Checks if two boards have the same configuration.
    Args:
        board (2D array): content of first board
        board (2D array): content of second board
    Returns:
        bool: True if two boards have same config, False if different config
    """
    n = len(board1)
    for row in range(n):
        for col in range(n):
            cur_tile = board1[row][col]
            other_cur_tile = board2[row][col]
            if cur_tile != other_cur_tile: return False
    return True

#print_board
def print_board(board):
    n = len(board)
    for x in range(n):
        print("[",end=" ")
        for y in range(n):
            if isinstance(board[x][y],str):
                print(' B',end=" ")
            else:
                if board[x][y] > 9:
                    print(str(board[x][y]),end=" ")
                else:
                    print(" " + str(board[x][y]),end=" ")
        print("]")

#probability_function
def probability_function(current_heuristic,neighbor_heuristic,temperature):
    number = math.e**((current_heuristic-neighbor_heuristic)/temperature)
    return number

#neighbor_analysis
def neighbor_analysis(cur_node,goal_dict1,goal_dict2,visited_boards):
    all_neighbors = []
    new_neighbors = []
    all_better_neighbors = []
    new_better_neighbors = []
    board = cur_node.board
    valid_move_list = valid_moves(board)
    for move in valid_move_list:
        moved_board = get_moved_board(board,move)
        heuristic = heuristic_function(moved_board,goal_dict1,goal_dict2)
        node = Node_hillclimb(cur_node,moved_board,move,heuristic,cur_node.path_cost + move[0])
        all_neighbors.append(node)
        if moved_board not in visited_boards:
            new_neighbors.append(node)
            if heuristic < cur_node.heuristic:
                new_better_neighbors.append(node)
        else:
            if heuristic < cur_node.heuristic:
                all_better_neighbors.append(node)
    return all_neighbors,new_neighbors,all_better_neighbors,new_better_neighbors

#anneal
def anneal(cur_node,neighbor_list,temperature):
    probability_list = []
    for neighbor in neighbor_list:
        probability_list.append(probability_function(cur_node.heuristic,neighbor.heuristic,temperature))
    return random.choices(neighbor_list,probability_list)[0]

#hill_climb
def hill_climb(board,time_limit,B_count):

    goal_board1, goal_board2, goal_dict1, goal_dict2 = goal_states(board,B_count)  
    min_path_cost = -1
    total_nodes_expanded = 0
    restart_count = 0
    start_h = heuristic_function(board,goal_dict1,goal_dict2)
    start_node = Node_hillclimb(None,board,[],start_h,0) 
    start_time = time.time()
    depth = 0
    while time.time() - start_time < time_limit:
        restart_count+=1
        result_node, nodes_expanded, final_depth, restarts = hill_climb_rec(start_node,time_limit-time.time()+start_time,B_count,goal_board1, goal_board2, goal_dict1, goal_dict2, depth)
        restart_count += restarts
        if result_node != None:
            if result_node.path_cost < min_path_cost or min_path_cost == -1:
                min_path_cost = result_node.path_cost
                min_path_node = result_node
                result_branching_factor = nodes_expanded**(1/final_depth)
                # print("Better Solution Found")
                # print(result_branching_factor)
                # print( result_node.path_cost)
            # else:
                # print("Solution Found")
            # print(time.time() - start_time)
        total_nodes_expanded += nodes_expanded

    if min_path_cost > 0:
        print("got a solution")
        print("Result board:", min_path_node.board)
        min_path_node.show_moves()
        print("Nodes expanded:",total_nodes_expanded)
        print("Result cost:",min_path_node.path_cost)
        print("Effective branching factor:", result_branching_factor)
    else:
        print("No result")
        print("Nodes expanded:",total_nodes_expanded)
    print("Total restart count:", restart_count)


def hill_climb_rec(start_node,time_limit,B_count,goal_board1, goal_board2, goal_dict1, goal_dict2, depth):
    depth+=1
    start_h = heuristic_function(start_node.board,goal_dict1,goal_dict2)
    board = start_node.board
    start_time = time.time()
    depth_limit = 500
    temperature_limit = 0.05
    visited_boards = []
    temperature = starting_temp
    number_attempts = 30
    depth_count = 0
    nodes_expanded = 0
    restart_count = 0
    
    cur_node = start_node
    
    while time.time() - start_time < time_limit and number_attempts > 0:
   
        restart = False
        nodes_expanded += 1
        depth_count += 1
        visited_boards.append(cur_node.board)
        temperature *= temp_multiplier

        if same_board(cur_node.board,goal_board1) or same_board(cur_node.board,goal_board2):
            return cur_node, nodes_expanded, depth_count, restart_count
        
        else:
            all_neighbors, new_neighbors, all_better_neighbors, new_better_neighbors = neighbor_analysis(cur_node,goal_dict1,goal_dict2,visited_boards)
            if temperature < temperature_limit:
                if len(new_better_neighbors) > 0:
                    cur_node = random.choice(new_better_neighbors)
                elif cur_node.heuristic<start_h/1.3:
                    # print("Found half heuristic. Making Checkpoint")
                    result_node, rec_nodes_expanded, rec_depth_count, rec_restart_count =  hill_climb_rec(cur_node,time_limit-time.time()+start_time, B_count, goal_board1, goal_board2, goal_dict1, goal_dict2, depth)
                    return result_node, rec_nodes_expanded+nodes_expanded, rec_depth_count+depth_count, rec_restart_count+restart_count
                else:
                    restart = True
            elif len(new_neighbors) > 0:
                cur_node = anneal(cur_node,new_neighbors,temperature)
            else:
                restart = True
        if restart: 
            number_attempts-=1
            visited_boards = []
            temperature = starting_temp
            depth_count = 0
            nodes_expanded = 0
            cur_depth = 0
            restart_count +=1
            cur_node = start_node
    return None, nodes_expanded, 0, restart_count



#########################################################################################################
#
#      Function Graveyard
#      These functions have been tested and aren't needed anymore
#
#########################################################################################################
"""
neighbor_boards
def neighbor_boards(board):
    boards_list = []
    valid_move_list = valid_moves(board)
    for move in valid_move_list:
        boards_list.append(get_moved_board(board,move))
    return boards_list

"""
"""


    goal_board1, goal_board2, goal_dict1, goal_dict2 = goal_states(board,B_count)  
    start_h = heuristic_function(board,goal_dict1,goal_dict2)
    temperature_time,restart_count,visited_boards,cur_nodes_expanded,cur_depth_count = reset_search(0,board,start_h)
    cur_node = Node_hillclimb(None,board,[],start_h,None)
    start_time = time.time()
    starter_board = board
    total_nodes_expanded = 0
    previous_nodes_expanded = 0
    previous_starter_node = None
    min_path_cost = -1


    #How deep to let the seach algorithm go
    depth_limit = 100

    #What temperature to transition from annealing to greedy
    #needs to be atleast 0.02 because 0.01 will cause the 
    #equation to evaluate to a number that is too large
    temperature_limit = 0.01
    while time.time() - start_time < time_limit:
        temperature = temperature_function(temperature_time)
        restart = False
        restart_current_node = False
        temperature_time += 1
        total_nodes_expanded += 1
        cur_nodes_expanded += 1
        cur_depth_count += 1
        visited_boards.append(cur_node.board)
    
        if cur_node.path_cost >= min_path_cost and min_path_cost!=-1:
            restart = True
            
        ##########################################################
        # Current board is a goal board
        ##########################################################
        elif same_board(cur_node.board,goal_board1) or same_board(cur_node.board,goal_board2):
            if (min_path_cost > cur_node.path_cost) or (min_path_cost == -1):
                result_node = cur_node
                min_path_cost = cur_node.path_cost
                result_branching_factor = cur_nodes_expanded**(1/cur_depth_count)
                print("Better solution found")
            restart = True
        ##########################################################
        # Reached depth limit or temperature limit
        ##########################################################
        elif cur_depth_count > depth_limit and depth_limit != -1:
            restart = True
        ##########################################################
        # Generate list of neighbors
        ##########################################################
        else:
            all_neighbors, new_neighbors, all_better_neighbors, new_better_neighbors = neighbor_analysis(cur_node,goal_dict1,goal_dict2,visited_boards)
            if temperature < temperature_limit:
                if len(new_better_neighbors) > 0:
                    cur_node = random.choice(new_better_neighbors)
                elif cur_node.heuristic<start_h/2:
                    restart_current_node = True
                    print("close")
                else:
                    print(cur_node.heuristic)
                    restart = True
            elif len(all_neighbors) > 0:
                cur_node = anneal(cur_node,all_neighbors,temperature)
            else:
                restart = True
        ##########################################################
        # If restart flag is True, then reset the search 
        ##########################################################
        if restart: 
            print("restarted")
            temperature_time, restart_count,visited_boards,cur_node,cur_nodes_expanded,cur_depth_count = reset_search(previous_starter_node,restart_count,starter_board,start)
            board = starter_board
            print("restarted")
        elif restart_current_node:
            start_h = cur_node.path_cost + cur_node.heuristic
            board = cur_node.board
            temperature_time = 0
            restart_count+=1
            visited_boards = []
            previous_starter_node = cur_node
            cur_depth_count = 0
            
    if min_path_cost > 0:
        print("Result board:", result_node.board)
        result_node.show_moves()
        print("Nodes expanded:",total_nodes_expanded)
        print("Result cost:",result_node.path_cost)
        print("Effective branching factor:", result_branching_factor)
    else:
        print("No result")
        print("Nodes expanded:",total_nodes_expanded)
    print("Total restart count:", restart_count)

"""