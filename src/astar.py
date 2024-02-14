import copy
import math
import random
import time
from src.npuzzle_helper import get_moved_board, goal_states, move_to_string, print_board, valid_moves


class Node_astar():
    """A node class for Hill Climbing"""

    def __init__(self, parent, depth, board, move, heuristic, path_cost=0):
        """Node initialization

        Args:
            parent (Node or None): Parent Node.
            board (2D array): Board content.
            move (list): List of moves in form of [tile,(dx,dy)], pass in [] if no move
            path_cost (int, optional): TODO _description_. Defaults to 0.
        """
        self.parent = parent
        self.depth = depth
        self.board = board
        self.move = move
        self.n = len(board)

        if len(self.move) > 0:
            self.path_cost = path_cost
        else:
            self.path_cost = 0
        
        self.heuristic = heuristic
        self.astar = heuristic + self.path_cost
    
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


def astar(board,B_count,weight):
    goal_board1, goal_board2, goal_dict1, goal_dict2 = goal_states(board,B_count)
    if weight:
        start_h1,_ = heuristic_weighted(board,goal_dict1)
        start_h2,_ = heuristic_weighted(board,goal_dict2)
    else:
        start_h1,_ = heuristic_unweighted(board,goal_dict1)
        start_h2,_ = heuristic_unweighted(board,goal_dict2)
    
    cur_node = Node_astar(None,0,board,[],min(start_h1,start_h2))
    result_node = None
    result_cost = -1
    start_time = time.time()
    restart_count = 0
    potential_nodes = []
    potential_astars = []
    heuristic_calc_time = []
    visited = []
    highest_depth = 0
    nodes_expanded = 0
    
    """start time to begin comparing and locating nodes
    """
    while time.time() - start_time < 30:
        # If the current board being explored in the current node is in a goal state:
        #   Evaluate the circumstances
        if cur_node.board == goal_board1 or cur_node.board == goal_board2:
            # if the current board set up was achieved with no moves
            # OR if the result of the move costs more than the current path:
            #   Return the cost and state
            result_node = cur_node
            result_cost = cur_node.path_cost
            result_branching_factor = nodes_expanded**(1/cur_node.depth)
            break
        # Explore a node
        nodes_expanded += 1
        moves = valid_moves(cur_node.board)
        cur_board = copy.deepcopy(cur_node.board)
        visited.append(cur_board)
        # Explore the moves and compare values based on the heuristics and formulate a temperature
        for each_move in moves:
            moved_board = get_moved_board(cur_board,each_move)
            if moved_board in visited: continue
            moved_path_cost = cur_node.path_cost + each_move[0]
            # Find the lowest heuristic for the board to decide which goal state to go with for this node
            if weight:
                heuristic1,t1 = heuristic_weighted(moved_board,goal_dict1)
                heuristic2,t2 = heuristic_weighted(moved_board,goal_dict2)
            else:
                heuristic1,t1 = heuristic_unweighted(moved_board,goal_dict1)
                heuristic2,t2 = heuristic_unweighted(moved_board,goal_dict2)
            
            heuristic_calc_time += [t1,t2]
            
            temp_node = Node_astar(cur_node,cur_node.depth + 1,moved_board,each_move,min(heuristic1,heuristic2),moved_path_cost)
            if temp_node.depth > highest_depth: highest_depth = temp_node.depth
            potential_nodes.append(temp_node)
            potential_astars.append(temp_node.astar)
        
        cur_min_astar = min(potential_astars)
        cur_min_astar_index = potential_astars.index(cur_min_astar)
        cur_min_astar_node = potential_nodes[cur_min_astar_index]
        potential_astars.pop(cur_min_astar_index)
        potential_nodes.pop(cur_min_astar_index)
        
        cur_node = cur_min_astar_node


    print("Astar results:")
    if result_cost > 0:
        print("Result board:", result_node.board)
        result_node.show_moves()
        print("Nodes expanded:",nodes_expanded)
        print("Nodes expanded per sec:",nodes_expanded/(time.time()-start_time))
        print("Result cost:",result_cost)
        print("Effective branching factor:", result_branching_factor)
        print("Avg heuristic calc time:", sum(heuristic_calc_time)/len(heuristic_calc_time))
        print("Total elapsed time:", time.time()-start_time)
    else:
        print("No result")
        print("Nodes expanded:",nodes_expanded)
        print("Nodes expanded per sec:",nodes_expanded/(time.time()-start_time))
        print("Total elapsed time:", time.time()-start_time)
        print("Effective branching factor:", nodes_expanded**(1/highest_depth))
        
    print("====================================")

def heuristic_weighted(board,goal_dict):
    """Get the heuristic value for the static board with no moves.
    Args:
        board (2D array): content of the board
        goal_dict (dictionary): key: tile of the board, value: corrdinate pair

    Returns:
        int: heuristic value for the current board
    float: time elapsed for this heuristic
    """
    n = len(board)
    start_time = time.time()
    heuristic_val = 0
    for row in range(n):
        for col in range(n):
            cur_tile = board[row][col]
            if isinstance(cur_tile, str):
                continue
            (dest_x,dest_y) = goal_dict.get(cur_tile)
            heuristic_val += cur_tile*(abs(dest_x-row) + abs(dest_y-col))
    # print('heuristic_val',heuristic_val)
    return heuristic_val,time.time()-start_time

#       |
#Cooper V
def heuristic_unweighted(board,goal_dict):
    """Get the heuristic value for the static board with no moves.
    Args:
        board (2D array): content of the board
        goal_dict (dictionary): key: tile of the board, value: corrdinate pair
    Returns:
        int: heuristic value for the current board
    float: time elapsed for this heuristic
    """
    n = len(board)
    start_time = time.time()
    heuristic_val = 0
    for row in range(n):
        for col in range(n):
            cur_tile = board[row][col]
            if isinstance(cur_tile, str):
                continue
            (dest_x,dest_y) = goal_dict.get(cur_tile)
            #heuristic_val += cur_tile*(abs(dest_x-row) + abs(dest_y-col))
            heuristic_val += (abs(dest_x-row) + abs(dest_y-col)) #removed cur_tile* from the line above
    # print('heuristic_val',heuristic_val)
    return heuristic_val,time.time()-start_time
#Cooper ^
#       |