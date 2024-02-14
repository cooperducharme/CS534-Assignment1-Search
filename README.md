# CS534-Assignment1-Search

Link to [assignment requirements](https://docs.google.com/document/d/1zZrod7BBBQ-20dRqdRCLycND-w9uOaW90HbSq2EeR6w/edit):

 Link to [assignment writeup](https://docs.google.com/document/d/1QdQBOo-eeScezxPynIWe-Z665kZZUPlNhOgw7DdigkQ/edit?usp=sharing) (backup)

## Usage

The script can be run from the command line by passing in the following arguments:

```bash
python npuzzle_main.py [problem_flag] [file_path] [heuristic (npuzzle only)] [weight_flag (npuzzle only)] [time_limit (greedy only)]
```

### problem_flag

**greedy**: Indicates that the puzzle to be solved by greedy search.

**npuzzle**: Indicates that the puzzle to be solved by astar search.

### file_path

The file name of CSV file that represents the puzzle board. The file should be located in the boards directory.

### heuristic (npuzzle only)

The heuristic to use when solving an npuzzle. Can be either **sliding** or **greedy**.

### weight_flag (npuzzle only)

Whether the heuristic should take into account tile weight. Can be either true or false (or t or f).

### time_limit (greedy only)

The time limit in seconds for finding a solution to the greedy puzzle.

## Example

To run the script with the greedy puzzle board board1.csv with a time limit of 10 seconds, use the following command:

```bash
python npuzzle_main.py greedy board1.csv 10
```

To run the script with the npuzzle board board2.csv using the sliding heuristic and not taking into account tile weight, use the following command:

```bash
python npuzzle_main.py npuzzle board2.csv sliding f
```
