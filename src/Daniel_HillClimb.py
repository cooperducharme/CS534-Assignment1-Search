import csv
import sys
import os
from src.helper_functions import read_file, hill_climb


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
        hill_climb(board,time_limit,B_count)
    
    #elif problem_flag == "npuzzle":
        #astar(board,B_count)
    
    return 0


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) <= 1:
        argv = [sys.argv[0],'greedy','board4.csv','12','']# debug placeholder
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