import copy
import math
import random
import time
from src.npuzzle_helper import get_moved_board, goal_states, move_to_string, valid_moves


class Node_hillclimb():
    """A node class for Hill Climbing"""

    def __init__(self, parent, board, move, h1, h2, path_cost=0):
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
        self.n = len(board)

        if len(self.move) > 0:
            self.path_cost = path_cost
        else:
            self.path_cost = 0
        
        self.heuristic_1 = h1
        self.heuristic_2 = h2

        
    def heuristic(self):
        return min(self.heuristic_1,self.heuristic_2)
    
    def astar(self):
        return min(self.heuristic_1,self.heuristic_2) + self.path_cost
    
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
    
    # def __format__(self, __format_spec: str) -> str:
    #     return self.show_moves(self)
    
    def __eq__(self, other):
        """Check if two Nodes have the same board

        Args:
            other (Node): Node to compare from

        Returns:
            Boolean: Return True if two boards are the same.
            False otherwise
        """
        self_board = self.board
        other_board = other.board
        n = len(self_board)
        for row in range(n):
            for col in range(n):
                cur_tile = self_board[row][col]
                other_cur_tile = other_board[row][col]
                if cur_tile != other_cur_tile: return False
        return True


def hill_climb(board,time_limit,B_count):
    goal_board1, goal_board2, goal_dict1, goal_dict2 = goal_states(board,B_count)
    start_h1 = heuristic_no_move(board,goal_dict1)
    start_h2 = heuristic_no_move(board,goal_dict2)
    
    cur_node = Node_hillclimb(None,board,[],start_h1,start_h2)
    result_node = None
    result_cost = -1
    start_time = time.time()
    restart_count = 0
    visited_boards = []
    TEMPERATURE = 10
    temperature = TEMPERATURE
    
    nodes_expanded = 0
    cur_nodes_expanded = 0
    cur_depth_count = 0
    
    """start time to begin comparing and locating nodes
    """
    while time.time() - start_time < time_limit:
        # If the current board being explored in the current node is in a goal state:
        #   Evaluate the circumstances
        if cur_node.board == goal_board1 or cur_node.board == goal_board2:
            # if the current board set up was achieved with no moves
            # OR if the result of the move costs more than the current path:
            #   Return the cost and state
            if (result_cost < 0) or (result_cost > cur_node.path_cost):
                result_node = cur_node
                result_cost = cur_node.path_cost
                result_branching_factor = cur_nodes_expanded**(1/cur_depth_count)
            # Make a new node to explore
            cur_node = Node_hillclimb(None,board,[],start_h1,start_h2)
            # Restart the search to find a new and potentially better solution based on new node
            visited_boards = []
            temperature = TEMPERATURE
            restart_count += 1
            cur_nodes_expanded = 0
            cur_depth_count = 0
            print("Restart because found a solution")
            
        # Explore a node
        nodes_expanded += 1
        cur_nodes_expanded += 1
        visited_boards.append(cur_node.board)
        moves = valid_moves(cur_node.board)
        cur_min_heuristic = 2147483647
        cur_board = copy.deepcopy(cur_node.board)
        cur_children = []
        # Explore the moves and compare values based on the heuristics and formulate a temperature
        for each_move in moves: 
            moved_board = get_moved_board(cur_board,each_move)
            moved_path_cost = cur_node.path_cost + each_move[0]
            # Find the lowest heuristic for the board to decide which goal state to go with for this node
            heuristic1 = heuristic_no_move(moved_board,goal_dict1)
            heuristic2 = heuristic_no_move(moved_board,goal_dict2)
            moved_heuristic = min(heuristic1,heuristic2)
            # Add this node to list of children for this node
            cur_children.append(Node_hillclimb(cur_node,moved_board,each_move,heuristic1,heuristic2,moved_path_cost))
            if (moved_heuristic < cur_min_heuristic):
                temp_best_moved = Node_hillclimb(cur_node,moved_board,each_move,heuristic1,heuristic2,moved_path_cost)
                cur_min_heuristic = moved_heuristic
        if (temp_best_moved.board not in visited_boards):
            cur_node = temp_best_moved
            cur_depth_count += 1
        else:
            # Annealing attempt with restart
            random_range = len(cur_children)
            random_node = cur_children[random.randrange(random_range)]
            random_heuristic = random_node.heuristic()
            if random_heuristic >= cur_node.heuristic():
                probability = math.e**((cur_node.heuristic()-random_heuristic)/temperature)
                if probability > random.uniform(0, 1):
                    cur_node = random_node
                    cur_depth_count += 1
                else: 
                    if cur_node.parent:
                        cur_node = cur_node.parent
                        cur_depth_count -= 1
                    else:
                        # Restart
                        cur_node = Node_hillclimb(None,board,[],start_h1,start_h2)
                        visited_boards = []
                        temperature = TEMPERATURE
                        restart_count += 1
                        cur_depth_count = 0
                        cur_nodes_expanded = 0
            else:
                cur_node = random_node
                cur_depth_count += 1
            temperature = temperature * 0.99

    if result_cost > 0:
        print("Result board:", result_node.board)
        result_node.show_moves()
        print("Nodes expanded:",nodes_expanded)
        print("Result cost:",result_cost)
        print("Effective branching factor:", result_branching_factor)
    else:
        print("No result")
        print("Nodes expanded:",nodes_expanded)
    print("Total restart count:",restart_count)
    
def greedier_hill_climb(board,time_limit,B_count):
    goal_board1, goal_board2, goal_dict1, goal_dict2 = goal_states(board,B_count)
    start_h1 = heuristic_no_move(board,goal_dict1)
    start_h2 = heuristic_no_move(board,goal_dict2)
    
    cur_node = Node_hillclimb(None,board,[],start_h1,start_h2)
    result_node = None
    result_cost = -1
    start_time = time.time()
    restart_count = 0
    TEMPERATURE = 10
    temperature = TEMPERATURE
    visited_boards = []
    nodes_expanded = 0
    cur_nodes_expanded = 0
    cur_depth_count = 0
    
    """start time to begin comparing and locating nodes
    """
    while time.time() - start_time < time_limit:
        # If the current board being explored in the current node is in a goal state:
        #   Evaluate the circumstances
        if cur_node.board == goal_board1 or cur_node.board == goal_board2:
            # if the current board set up was achieved with no moves
            # OR if the result of the move costs more than the current path:
            #   Return the cost and state
            if (result_cost < 0) or (result_cost > cur_node.path_cost):
                result_node = cur_node
                result_cost = cur_node.path_cost
                result_branching_factor = cur_nodes_expanded**(1/cur_depth_count)
            # Make a new node to explore
            cur_node = Node_hillclimb(None,board,[],start_h1,start_h2)
            # Restart the search to find a new and potentially better solution based on new node
            temperature = TEMPERATURE
            restart_count += 1
            cur_nodes_expanded = 0
            cur_depth_count = 0
            print("Restart because found a solution")
            
        # Randomly pick a child node
        moves = valid_moves(cur_node.board)
        random_range = len(moves)
        random_move = moves[random.randrange(random_range)]
        cur_board = copy.deepcopy(cur_node.board)
        moved_board = get_moved_board(cur_board,random_move)

        # Expand this node
        nodes_expanded += 1
        cur_nodes_expanded += 1
        visited_boards.append(moved_board)
        
        # Initialize this node
        moved_path_cost = cur_node.path_cost + random_move[0]
        heuristic1 = heuristic_no_move(moved_board,goal_dict1)
        heuristic2 = heuristic_no_move(moved_board,goal_dict2)
        moved_heuristic = min(heuristic1,heuristic2)
        random_node = Node_hillclimb(cur_node,moved_board,random_move,heuristic1,heuristic2,moved_path_cost)
        # Annealing attempt with restart
        if moved_heuristic >= cur_node.heuristic():
            temperature *= 0.999
            probability = math.e**((cur_node.heuristic()-moved_heuristic)/temperature)
            if probability > random.uniform(0, 1):
                cur_node = random_node
                cur_depth_count += 1
            else:
                # Restart
                cur_node = Node_hillclimb(None,board,[],start_h1,start_h2)
                temperature = TEMPERATURE
                restart_count += 1
                cur_nodes_expanded = 0
                cur_depth_count = 0
        else:
            cur_node = random_node
            cur_depth_count += 1

    if result_cost > 0:
        print("Result board:", result_node.board)
        result_node.show_moves()
        print("Nodes expanded:",nodes_expanded)
        print("Result cost:",result_cost)
        print("Effective branching factor:", result_branching_factor)
    else:
        print("No result")
        print("Nodes expanded:",nodes_expanded)
    print("Total restart count:",restart_count)
    
def heuristic_no_move(board,goal_dict):
    """Get the heuristic value for the static board with no moves.
    Args:
        board (2D array): content of the board
        goal_dict (dictionary): key: tile of the board, value: corrdinate pair

    Returns:
        int: heuristic value for the current board
    """
    n = len(board)
    heuristic_val = 0
    for row in range(n):
        for col in range(n):
            cur_tile = board[row][col]
            if isinstance(cur_tile, str):
                continue
            (dest_x,dest_y) = goal_dict.get(cur_tile)
            heuristic_val += cur_tile*(abs(dest_x-row) + abs(dest_y-col))
    # print('heuristic_val',heuristic_val)
    return heuristic_val


"""
    Author: Shun
    cur_node = start
    list_of_result_node = []
    list_of_result_cost = []
    while in time_limit:
        if cur_node = goal_board:
            update list_of_result_node & list_of_result_cost
        get potential_moves
        for each_move in potential_moves
            if better astar value:
                cur_node = Node(each_move)
                break
            get probability
            if probability:
                cur_node = Node(each_move)
                break
"""

"""
Author: Daniel Goto
    open file
    create board array from file
    find goal board array from board array
    current board = starting board
    start_time = current time
    while current board!=goal board and (current_time-start_time) < 12
        create array of potential boards and the probability of each depending on cost heuristic
        if there is a board in array with a probability of 1
            for board in array:
                if probability == 1:
                    current board = new board
                    exit loop
        else:
            if temperature < arbitrary threshold (maybe 0.5 or something)
                for board in potential boards:
                    if cost of potential board is same as cost current board and potential board!=previous board
                        previous board = current board
                        current board = potential board
                        exit for loop
            else 
                perform weighted random choice and set current board to chosen board
                    """