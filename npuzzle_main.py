import sys
import os
from src.astar import astar
from src.astar_greedy_h import astar_greedy_h
from src.helper_functions import hill_climb
from src.npuzzle_helper import read_file

def main(problem_flag,file_path,heuristic,weight_flag,time_limit):
    """The main structure of the script.

    Args:
        problem_flag (String): npuzzle or greedy, seperating problem 1 and 2
        file_path (String): The path string for input board csv file
        heuristic (String): sliding or greedy, Which heuristic to use problem 3
        weight_flag (Boolean): Whether the heuristic function will consider the weight of tile or not

    Returns:
        None
    """
    board,B_count = read_file(file_path)
    if problem_flag == "greedy":
        print(">>> Start greedy search >>>")
        hill_climb(board,time_limit,B_count)
    elif problem_flag == "npuzzle":
        if heuristic == "sliding":
            if weight_flag.lower() == "true" or weight_flag.lower() == "t":
                print(">>> Start astar search with weighted sliding heuristic >>>")
                astar(board,B_count,True)
            elif weight_flag.lower() == "false" or weight_flag.lower() == "f":
                print(">>> Start astar search with UN-weighted sliding heuristic >>>")
                astar(board,B_count,False)
            else:
                print("Wrong input for weight flag")
                print("Should be true or false")
                return -1
        elif heuristic == "greedy":
            print(">>> Start astar search with greedy heuristic >>>")
            astar_greedy_h(board,B_count)
        else:
            print("Wrong input for heuristic flag")
            print("Should be sliding or greedy")
            return -1
    return 0

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) <= 1:
        # argv = [sys.argv[0],'greedy','5by5_1.csv','30',''] # debug placeholder
        argv = [sys.argv[0],'greedy','board1.csv','10','t']
    problem_flag = argv[1] # basically npuzzle or greedy at the moment to seperat part1 and 2
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/boards/' + argv[2] # The file name of the board to read in
    
    if problem_flag == "greedy":
        time_limit = int(argv[3])
        heuristic = ''
        weight_flag = ''
    elif problem_flag == "npuzzle":
        time_limit = -1
        heuristic = argv[3] # Which heuristic to use (sliding, greedy (Part 3))
        weight_flag = argv[4] # Whether the heuristic should take into account tile weight (true vs false)?
    
    main(problem_flag,file_path,heuristic,weight_flag,time_limit)