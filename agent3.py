import pandas as pd
import numpy as np
from math import sqrt
import heapq
from matplotlib import pyplot
from copy import deepcopy



class Cell:
    def __init__(self, x, y):
        self.xpos = x
        self.ypos = y
        self.n = None
        self.isvisited = None
        self.status = "hidden"
        self.c = None
        self.b = None
        self.e = None
        self.h = None

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

def sense(currcell, grid,dim):
    currcell.n = 0
    currcell.c = 0
    currcell.e=0
    currcell.b=0
    
    all_moves = [[1, 0],
                 [0, 1],
                 [-1, 0],
                 [0, -1],
                 [1, 1],
                 [-1, -1],
                 [-1, 1],
                 [1, -1]]
    for a_move in all_moves:
        child_x = currcell.xpos + a_move[0]
        child_y = currcell.ypos + a_move[1]
        if (child_x > dim-1 or child_x < 0 or child_y > dim-1 or child_y < 0):
            continue
        else:
            currcell.n += 1
            if(grid[child_x][child_y] == 0):
                currcell.c += 1
    # currcell.h=currcell.n-currcell.c
    currcell.h=currcell.n

def infer(currcell,knowledge_grid,celldetailsgrid,dim):
    blocked_inferred=set()
    # if currcell.h:

    if (currcell.h):
        if currcell.c==currcell.b:

            all_moves = [[1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],
                    [1, 1],
                    [-1, -1],
                    [-1, 1],
                    [1, -1]]
            for a_move in all_moves:
                child_x = currcell.xpos + a_move[0]
                child_y = currcell.ypos + a_move[1]
                if (child_x > dim-1 or child_x < 0 or child_y > dim-1 or child_y < 0):
                    continue
                else:
                    if celldetailsgrid[child_x][child_y].status=="hidden":
                        celldetailsgrid[child_x][child_y].status="empty"
                        knowledge_grid[child_x][child_y]=1
                        currcell.e+=1
                        currcell.h-=1
            return knowledge_grid,blocked_inferred
        elif currcell.n-currcell.c==currcell.e:
            all_moves = [[1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],
                    [1, 1],
                    [-1, -1],
                    [-1, 1],
                    [1, -1]]
            for a_move in all_moves:
                    child_x = currcell.xpos + a_move[0]
                    child_y = currcell.ypos + a_move[1]
                    if (child_x > dim-1 or child_x < 0 or child_y > dim-1 or child_y < 0):
                        continue
                    else:
                        if celldetailsgrid[child_x][child_y].status=="hidden":
                            
                            celldetailsgrid[child_x][child_y].status="blocked"
                            blocked_inferred.add((child_x,child_y))
                            # print("Agent 3 current "+str(currcell.xpos)+str(currcell.ypos)+"changing "+str(child_x)+str(child_y))
                            knowledge_grid[child_x][child_y]=0
                            currcell.b+=1
                            currcell.h-=1
            return knowledge_grid,blocked_inferred
    return knowledge_grid,blocked_inferred

def main(dim,is_gridknown,density,grid):
    '''
    This function execuated repeated A* algorithm and uses above functions to do so.
    :return: None
    '''
    fringe=[]
    # dim=5
    # is_gridknown="No"
    # density=0.1



    
    #grid = create_grid(density, dim)                ## create a grid with entered density and dim values.

    
    # assuming unblocked for all cells
    knowledge_grid = [[-1 for _ in range(dim)] for _ in range(dim)]          ## intialise knowledge grid to all 1's
    start = (0, 0)
    end = (dim-1, dim-1)
    knowledge_grid[start[0]][start[1]] = 1
    knowledge_grid[end[0]][end[1]] = 1
    celldetailsgrid=[[Cell(i,j) for i in range(dim)] for j in range(dim)]
    celldetailsgrid[0][0].status="empty"
    sense(celldetailsgrid[0][0],grid,dim)
    knowledge_grid,blocked_inferred=infer(celldetailsgrid[0][0],knowledge_grid,celldetailsgrid,dim)
   


    bump_counter = 0
    ll, path = search(grid,fringe, knowledge_grid, start, end,is_gridknown)
    # print(path)
    final_path=[]
    if(ll!="Unsolvable" and is_gridknown=="No"):
        while(len(path) > 1 and ll!="Unsolvable"):
            count=0
            flag=0
            flag2=0
            pathset=set(path)
            # traverse the path obtained from search function to see if blocked cell exists or not.
            # If blocked cell exists, run search function again to calculate the path
            #  Continue in this while loop -1) either path returned is 0 that means nothing left in fringe and no path to reach goal 2) or path exists to reach goal

            for i in path[::-1]:
                count+=1
                final_path.append((i[0],i[1]))
                if(count==1):continue
                
                
                

                if(grid[i[0]][i[1]] == 0):  # blocked in grid
                    celldetailsgrid[i[0]][i[1]].status="blocked"
                    celldetailsgrid[path[path.index(i)+1][0]][path[path.index(i)+1][1]].b+=1
                    celldetailsgrid[path[path.index(i)+1][0]][path[path.index(i)+1][1]].h-=1

                    final_path.pop()
                    knowledge_grid[i[0]][i[1]] = 0  # updating knowledge_grid
                    new_start_position = path[path.index(i)+1][0], path[path.index(i)+1][1]
                    bump_counter+=1
                    for i in range(0, dim):
                        for j in range(0, dim):
                            knowledge_grid,blocked_inferred= infer(celldetailsgrid[i][j],knowledge_grid,celldetailsgrid,dim)
                    ll, path = search(grid, fringe, knowledge_grid,
                                  new_start_position, end, is_gridknown)
                    finalresult=ll
                    break
                elif (grid[i[0]][i[1]] == 1 ):
                    celldetailsgrid[i[0]][i[1]].status="empty"
                    # print("i",i)
                    # print("path[path.index(i)+1][0]", path[path.index(i)+1][0])
                    celldetailsgrid[path[path.index(i)+1][0]][ path[path.index(i)+1][1]].e+=1
                    celldetailsgrid[path[path.index(i)+1][0]][ path[path.index(i)+1][1]].h-=1
                    knowledge_grid[i[0]][i[1]] = 1
                    sense(celldetailsgrid[i[0]][i[1]],grid,dim)

                    for i in range(0, dim):
                        for j in range(0, dim):
                            knowledge_grid,blocked_inferred= infer(celldetailsgrid[i][j],knowledge_grid,celldetailsgrid,dim)
                    # print("knowledge_grid",knowledge_grid)
                    if(len(blocked_inferred)>0):
                        for k in blocked_inferred:
                            if (k  in  pathset):
                                flag2=1
                                break
                    if flag2==1:
                        new_start_position = path[path.index(i) ][0], path[path.index(i)][1]
                        final_path.pop()
                        # print("new_start_position",new_start_position)
                        # print("knowledge_grid")
                        # print_grid(knowledge_grid)

                        ll, path = search(grid, fringe, knowledge_grid,
                                      new_start_position, end, is_gridknown)
                        # print("path",path)
                        finalresult = ll
                        break  
                if(count==len(path)):
                    print("Solved")
                    flag=1
                    # print("final_path",final_path)
                    break
            if(flag==1):
                
                

                return final_path, knowledge_grid, bump_counter
                break        
        if(ll=="Unsolvable"):
            print("Unsolvable")
            return [],knowledge_grid, bump_counter
        if(flag!=1):
            # print("finalresult",finalresult)
            return [], knowledge_grid, bump_counter

    elif(is_gridknown=="Yes"):
        print(ll)
        print("path",path)

    else:
        print("Unsolvable")
        return [],knowledge_grid, bump_counter

    # for (i, j) in final_path:
    #     grid[i][j] = 2
    # pyplot.figure(figsize=(dim, dim))
    # pyplot.imshow(grid)
    # pyplot.grid()
    # pyplot.xticks(size=14, color="red")
    # pyplot.show()


# grid = create_grid(0.2, 5)
# grid=[[1,1,0,1,1],[1,0,1,]]
# start = (0,0)
# end = (4,4)

# solved, path2 = search(knowledge_grid,[],[],start, end,"Yes")

# print(set(path), len(set(path)))
# print(path2[::-1], len(path2))