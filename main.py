import agent3
import agent4
import numpy as np

dim=101
p=0.3
grid = [[0 for i in range(dim)] for j in range(dim)]


    ## Loop over inputted dimension
for i in range(dim):
    for j in range(dim):
        actual_prob = np.random.random_sample()  ## generating a random number
        if actual_prob > p:                      ## if the generated random number > p, assign it 1 (meaning it is
            grid[i][j] = 1                       ## traversable.
        else:
            grid[i][j] = 0

grid[0][0] = 1                                   ## start node and end node is always traversable.
grid[dim - 1][dim - 1] = 1

path1,kg=agent3.main(dim,"No",p,grid)
path2,kg=agent4.main(dim,"No",p,grid)
print("Length of path for agent 3",len(path1) )
print("Length of path for agent 4",len(path2) )

