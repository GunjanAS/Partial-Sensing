import agent3
import agent4
import agent1
import agent2
import astar
import numpy as np


dim=101
p=0.28
grid = [[0 for i in range(dim)] for j in range(dim)]


    
for i in range(dim):
    for j in range(dim):
        actual_prob = np.random.random_sample()  ## generating a random number
        if actual_prob > p:                      ## if the generated random number > p, assign it 1 (meaning it is
            grid[i][j] = 1                       ## traversable.
        else:
            grid[i][j] = 0

grid[0][0] = 1                                   ## start node and end node is always traversable.
grid[dim - 1][dim - 1] = 1

path1,kg1,bumps1=agent1.main(dim,"No",p,grid)
path2,kg2,bumps2=agent2.main(dim,"No",p,grid)


path3,kg3,bumps3=agent3.main(dim,"No",p,grid)
# print("path3",path3)
path4,kg4,bumps4=agent4.main(dim,"No",p,grid)

print("Length of path for agent 1",len(path1) )
print("Length of path for agent 2",len(path2) )
print("Length of path for agent 3",len(path3) )
print("Length of path for agent 4",len(path4) )
print("Number of bumps 1",bumps1 )
print("Number of bumps 2",bumps2 )
print("Number of bumps 3",bumps3 )
print("Number of bumps 4",bumps4 )

##----Calculate shortest path in the KG-----##
for i in range(0, dim):
    for j in range(0, dim):
        if(kg1[i][j] == 1):
            kg1[i][j] = 0
for i in range(0, dim):
    for j in range(0, dim):
        if(kg1[i][j] == 2):
            kg1[i][j] = 1
for i in range(0, dim):
    for j in range(0, dim):
        if(kg2[i][j] == 1):
            kg2[i][j] = 0
for i in range(0, dim):
    for j in range(0, dim):
        if(kg2[i][j] == 2):
            kg2[i][j] = 1
for i in range(0, dim):
    for j in range(0, dim):
        if(kg3[i][j] == -1):
            kg3[i][j] = 0
for i in range(0, dim):
    for j in range(0, dim):
        if(kg4[i][j] == -1):
            kg4[i][j] = 0
# print("kg1",kg1)
# print("kg3",kg3)
ll,path1_kg=astar.main(kg1,dim)
ll,path2_kg=astar.main(kg2,dim)
ll,path3_kg=astar.main(kg3,dim)
ll,path4_kg=astar.main(kg4,dim)

# # ll,path4_kg=astar.main(kg4,dim)
print("Length of path for agent 1 in kg",len(path1_kg) )
print("Length of path for agent 2 in kg",len(path2_kg) )
print("Length of path for agent 3 in kg",len(path3_kg) )
print("Length of path for agent 3 in kg",len(path4_kg) )





