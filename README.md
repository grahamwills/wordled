 # Wordled
 
Build an optimal solution path for Wordle

## Prepare

Read the following data sources:
 * List of words believed to be used as the Wordle dictionary
 * List of known Wordle solutions (scraped from a web site)
 * List of word frequencies
 * Dictionary of words with parts of speech

With the above we create pickled data sets for use in the main analysis
program

## Analyze

Runs a tree search trying out candidate attempts at each node for all
possible candidates. We prune total nodes (to several million, typically) at
each tree depth to limit the size of the tree. Pruning the tree retains those
attempts which minimize the number of words all in one potential response
(a minimax approach).

The final tree runs through each node and looks at all possible candidates
that were in it, and selects the one that minimizes:

 * The number of words that the subtree starting at that guess fails to 
   guess in 6 guesses or fewer, weighted by word frequency
 * If that is tied, the one that minimizes the weighted average number of 
   guesses required


Here is the diagnostic output from the analyze run:

    READ 12949 WORDS
    READ 243 PRE_CALCULATED OUTCOMES
    READ 167676601 PRE_CALCULATED SCORES
    READ 781 KNOWN SOLUTIONS
    781 of 781 are allowable for this fraction of words
    Processing 1 nodes at depth = 0
    1 nodes evaluated in 21.212659 seconds
    Processing 3,058 nodes at depth = 1
    3,058 nodes evaluated in 23.468727 seconds
    Processing 496,634 nodes at depth = 2
    ... Pruning candidates from 3,373,500 to 3,000,000
    ... Adding candidates to node
    496,634 nodes evaluated in 118.000741 seconds
    Processing 9,276,792 nodes at depth = 3
    ... Pruning candidates from 30,166,226 to 3,000,000
    ... Adding candidates to node
    9,276,792 nodes evaluated in 506.135254 seconds
    Processing 5,455,609 nodes at depth = 4
    ... Pruning candidates from 12,968,906 to 3,000,000
    ... Adding candidates to node
    5,455,609 nodes evaluated in 226.64981 seconds
    Processing 2,352,715 nodes at depth = 5
    ... Pruning candidates from 4,896,457 to 3,000,000
    ... Adding candidates to node
    2,352,715 nodes evaluated in 145.183547 seconds
    Processing 1,955,127 nodes at depth = 6
    1,955,127 nodes evaluated in 20.200888 seconds
    Processing 1,204,338 nodes at depth = 7
    1,204,338 nodes evaluated in 176.804652 seconds
    Processing 2,323,503 nodes at depth = 8
    ... Pruning candidates from 5,560,080 to 3,000,000
    ... Adding candidates to node
    2,323,503 nodes evaluated in 373.330562 seconds
    Processing 2,712,782 nodes at depth = 9
    ... Pruning candidates from 3,915,032 to 3,000,000
    ... Adding candidates to node
    2,712,782 nodes evaluated in 29.306519 seconds
    Processing 1,278,709 nodes at depth = 10
    1,278,709 nodes evaluated in 9.184049 seconds
    Processing 437,966 nodes at depth = 11
    437,966 nodes evaluated in 3.08229 seconds
    Processing 50,400 nodes at depth = 12
    50,400 nodes evaluated in 0.421279 seconds
    Trying scoring method: Unsolved | Average (Weighted)
    
    Time taken = 3487.888373s (1767.302377 to select best paths)

# Show

Show builds a Markdown document that does two things:

 * Takes the known solutions and runs the tree on it, producing some summary
   statistics showing how well it worked
 * Prints out the first three levels of the tree into a document that
   can be referred to start a wordle solution.