import pandas as pd
import numpy as np
from math import sqrt
import heapq
from matplotlib import pyplot
from copy import deepcopy

def create_grid(p, dim):
    '''

    :param p: probability with which a node is blocked
    :param dim: dimension of the matrix
    :return: a 2d list denoting grid where 1 = traversable node, 0 = non traversable

    This function generates a random grid by taking into account probability 'p' and dimension 'dim'
    '''

    ## initialise a grid with all 0. 0 = cell blocked, 1 =  cell unblocked
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
    return grid


def print_grid(grid):
    for row in grid:
        for e in row:
            print(e, end=" ")

        print()


# print_grid(create_grid(0.4, 5))


class Node:
    '''
    A node class that stores 5 things for a node - position, parent, g(n), h(n), f(n)
    '''
    def __init__(self, parent=None, position=None):
        '''
        This function initalises a node by setting parent , position and heuristic values as 0
        :param parent: parent of the current code
        :param position: position of the current code
        '''
        self.parent = parent
        self.position = position
        self.g = 0
        self.f = 0
        self.h = 0

    def __eq__(self, node):
        '''
        This function is overload for == operator in python. It is used to compare if two nodes are equal.

        :param node: node to compare with
        :return: 1 if two self and node is equal, otherwise 0

        '''
        if (self.position[0] == node.position[0] and self.position[1] == node.position[1]) :
            return True
        else:
            return False
    
    def __lt__(self, other):
        '''
        This function is overload for < operator in python. It is used to compare if one node is less than other.

        :param other: node to compare with
        :return: 1 if self's f value is less than other's f value, otherwise 0

        '''
        return self.f < other.f


def generate_children(grid, knowledge_grid, fringe, visited_list, current, all_moves, end_node,is_gridknown):

    '''
    This function uses a grid (be it grid or knowledge) and generates all valid children of the current node.

    :param grid: the original actual grid
    :param knowledge_grid: the knowledge grid
    :param fringe: list of nodes in the priority queue
    :param visited_list: a dictionary of nodes already visited
    :param current: current node in the queue
    :param all_moves: array of all valid moves
    :param end_node: end position/node in the grid
    :param is_gridknown: parameter to switch between grid and knowledge grid
    :return: array of relevant children
    '''
    current_x, current_y = current.position
    relevant_children = []
    dim = len(grid)

    for a_move in all_moves:                                            ## looping over all valid moves
        child_x = current_x + a_move[0]
        child_y = current_y + a_move[1]
        if child_x > dim-1 or child_x < 0 or child_y > dim-1 or child_y < 0:   ## condition to check if node is in within
                                                                               ## boundaries of the grid
            continue
        children_node = Node(current, (child_x, child_y))                       ## initalising children node with current
                                                                                ## as parent and child_x, child_y as position

        if(is_gridknown=="No"):                                                 ## isgridknown checks whether to we have grid
                                                                                ## loaded in the memory, if not we use knowledge
                                                                                ## grid
            grid_for_pathcalculation = knowledge_grid
        else:
            grid_for_pathcalculation = grid
        if (grid_for_pathcalculation[child_x][child_y] != 0) and (visited_list.get(children_node.position) != "Added"  ): ## condition to check is current node
                                                                                                                          ## is not blocked and current node is
                                                                                                                          ## not in visited list

            children_node.g = current.g + 1                                                                          ## assigining current g = g(parent) + 1
            children_node.h = abs(children_node.position[0] - end_node.position[0]) + abs(                           ## using manhattan distance as our heuristic
                children_node.position[1] - end_node.position[1])
           
            children_node.f = children_node.g + children_node.h                                                      ## f(n) = g(n) + f(n)
            relevant_children.append(children_node)
    return relevant_children


def search(grid, fringe,knowledge_grid, start_position, end_position,is_gridknown):
    '''

    :param grid: the original actual grid
    :param fringe: list of all processed nodes
    :param knowledge_grid: the knowledge grid
    :param start_position: start position in grid
    :param end_position: end position in grid
    :param is_gridknown: parameter to switch between grid and knowledge grid
    :return: the path from start node to end node
    '''
    startNode = Node(None, start_position)
    endNode = Node(None, end_position)

    fringe=[]
    visited_nodes = {}
    already_fringed = {}                                    ## a hashmap to keep track of fringed nodes and its lowest cost
    already_fringed[startNode.position] = startNode.f
    heapq.heappush(fringe,(startNode.f,startNode))          ## pushing start node in fringe
    all_moves = [[1, 0],                                    ## defined all moves -
                 [0, 1],                                    ##[1,0] - move right
                 [-1, 0],                                   ## [0,1] - move down
                 [0, -1]]                                   ## [0,-1] - move up
                                                            ## [-1,0] - move left

    path = []
    while fringe:                                           ## while fringe is not empty
        current = heapq.heappop(fringe)                     ## popping node from fringe
        current=current[1]
        visited_nodes[current.position]="Added"             ## assigning current node to visited

        if current.position== endNode.position:

            i = current
            while(i is not None):                           ## traversing path if current=goal to get the path from start to goal
                path.append(i.position)
                i = i.parent

            return "Solvable", path
        children = generate_children(                       ## otherwise generate children
            grid, knowledge_grid, fringe, visited_nodes, current, all_moves, endNode,is_gridknown)
        if children:
            for node in children:
                if node.position in already_fringed:                    ## checking if the children is already fringed,
                    if already_fringed[node.position] > node.f:         ## if yes update and push the moinimum cost one
                        already_fringed[node.position] = node.f         ## otherwise ignore the child
                        heapq.heappush(fringe, (node.f, node))
                else:
                    heapq.heappush(fringe, (node.f, node))              ## if the child is not already fringed, push it
                    already_fringed[node.position] = node.f             ## to priority queue and assign in the hashmap
                

    return "Unsolvable", path



def main(dim,is_gridknown,density,grid):
    '''
    This function execuated repeated A* algorithm and uses above functions to do so.
    :return: None
    '''
    fringe=[]
    bumps=0


    
    #grid = create_grid(density, dim)                ## create a grid with entered density and dim values.

    
    # assuming unblocked for all cells
    knowledge_grid = [[1 for _ in range(dim)] for _ in range(dim)]          ## intialise knowledge grid to all 1's
    im = None

    # print_grid(grid)
    start = (0, 0)
    end = (dim-1, dim-1)
    all_moves = [[1, 0],
                 [0, 1],
                 [-1, 0],
                 [0, -1]]
    for a_move in all_moves:
        child_x = start[0] + a_move[0]
        child_y = start[1] + a_move[1]
        if (child_x > dim-1 or child_x < 0 or child_y > dim-1 or child_y < 0):
                continue
        else:
            if(grid[child_x][child_y] == 0):
                knowledge_grid[child_x][child_y] = 0                               ## update the knowledge grid with field of view
    ll, path = search(grid,fringe, knowledge_grid, start, end,is_gridknown)
    final_path=[]
    if(ll!="Unsolvable" and is_gridknown=="No"):
        while(len(path) > 1 and ll!="Unsolvable"):
            count=0
            flag=0

            # traverse the path obtained from search function to see if blocked cell exists or not.
            # If blocked cell exists, run search function again to calculate the path
            #  Continue in this while loop -1) either path returned is 0 that means nothing left in fringe and no path to reach goal 2) or path exists to reach goal

            for i in path[::-1]:
                count+=1
                for a_move in all_moves:
                    child_x = i[0] + a_move[0]
                    child_y = i[1] + a_move[1]
                    if (child_x > dim-1 or child_x < 0 or child_y > dim-1 or child_y < 0):
                        continue
                    else:
                        if(grid[child_x][child_y] == 0):
                            knowledge_grid[child_x][child_y] = 0
                        elif(grid[child_x][child_y] == 1):
                            knowledge_grid[child_x][child_y] = 2
                final_path.append((i[0],i[1]))

                if(grid[i[0]][i[1]] == 0):  # blocked in grid
                    bumps+=1
                    final_path.pop()
                    knowledge_grid[i[0]][i[1]] = 0  # updating knowledge_grid
                    new_start_position = path[path.index(i)+1][0], path[path.index(i)+1][1]
                    ll, path = search(grid, fringe, knowledge_grid,
                                  new_start_position, end, is_gridknown)
                    finalresult=ll
                    break

                if(count==len(path)):
                    print("Solved")
                    flag=1
                    # print("final_path",final_path)
                    break
            if(flag==1):
                return final_path, knowledge_grid,bumps
                break        
        if(ll=="Unsolvable"):
            print("Unsolvable")
            return [],knowledge_grid,bumps
        if(flag!=1):
            # print("finalresult",finalresult)
            return [],knowledge_grid,bumps

    elif(is_gridknown=="Yes"):
        print(ll)
        print("path",path)

    else:
        print("Unsolvable")
        return [],knowledge_grid,bumps

    # for (i, j) in final_path:
    #     grid[i][j] = 2
    # pyplot.figure(figsize=(dim, dim))
    # pyplot.imshow(grid)
    # pyplot.grid()
    # pyplot.xticks(size=14, color="red")
    # pyplot.show()





