import copy
import time
from src.astar import Node_astar, heuristic_weighted
from src.npuzzle_helper import get_moved_board, goal_states, valid_moves
import random
def astar_greedy_h(board,B_count):
    goal_board1, goal_board2, goal_dict1, goal_dict2 = goal_states(board,B_count)
    start_h1,_ = heuristic_weighted(board,goal_dict1)
    start_h2,_ = heuristic_weighted(board,goal_dict2)

    cur_node = Node_astar(None,0,board,[],start_h1,start_h2)
    result_node = None
    result_cost = -1
    start_time = time.time()
    restart_count = 0
    potential_nodes = []
    potential_astars = []
    heuristic_calc_time = []
    nodes_expanded = 0
    visited = []
    highest_depth = 0
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
            temp_node = Node_astar(cur_node,cur_node.depth + 1,moved_board,each_move,cur_node.heuristic,moved_path_cost)
            heuristic1,t1 = greedy_heuristic(temp_node,goal_board1,goal_dict1)
            heuristic2,t2 = greedy_heuristic(temp_node,goal_board2,goal_dict2)
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


    print("Astar with greedy heuristic results:")
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
        print("Avg heuristic calc time:", sum(heuristic_calc_time)/len(heuristic_calc_time))
        print("Effective branching factor:", nodes_expanded**(1/highest_depth))
        print("Total elapsed time:", time.time()-start_time)
    print("====================================")    
        

def greedy_heuristic(input_node,goal_board,goal_dict):
    """Get the greedy heuristic value for the static board with no moves.
    Args:
        input_node (Node_hillclimb): node indented to get greedy heuristic
        goal_board,goal_dict

    Returns:
        int: greedy heuristic value for the current board
        float: time elapsed for this heuristic
    """
    
    cur_node = input_node
    start_time = time.time()
    
    while 1:
        # If the current board being explored in the current node is in a goal state:
        #   Evaluate the circumstances
        if cur_node.board == goal_board:
            return cur_node.astar - input_node.path_cost, time.time() - start_time

        # Randomly pick a child node
        moves = valid_moves(cur_node.board)
        # random_range = len(moves)
        # random_move = moves[random.randrange(random_range)]
        cur_board = copy.deepcopy(cur_node.board)
        # moved_board = get_moved_board(cur_board,random_move)

        cur_min_heuristic = 2147483647
        for each_move in moves: 
            moved_board = get_moved_board(cur_board,each_move)
            moved_path_cost = cur_node.path_cost + each_move[0]
            # Find the lowest heuristic for the board to decide which goal state to go with for this node
            moved_heuristic,_ = heuristic_weighted(moved_board,goal_dict)
            # Add this node to list of children for this node
            if (moved_heuristic < cur_min_heuristic):
                temp_best_moved = Node_astar(cur_node,cur_node.depth + 1,moved_board,each_move,moved_heuristic,moved_path_cost)
                cur_min_heuristic = moved_heuristic
        # Check if heuristic got better
        if cur_min_heuristic >= cur_node.heuristic:
            # Worse heuristic, return prev astar and elapsed time
            return cur_node.astar - input_node.path_cost, time.time() - start_time
        else:
            # Better heuristic, continue greedy search
            cur_node = temp_best_moved
        
    return -1 # Something weird happens if it reaches this step