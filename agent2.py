from PIL.Image import NONE
import pandas as pd
import numpy as np
from math import sqrt
import heapq
import matplotlib.pylab as plt
import time
from matplotlib import pyplot
from copy import deepcopy





class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.f = 0
        self.h = 0

    def __eq__(self, node):
        if (self.position[0] == node.position[0] and self.position[1] == node.position[1]):
            return True
        else:
            return False

    def __lt__(self, other):
        return self.f < other.f


class PriorityQueue:
    def __init__(self, ):
        self.fringe = {}

    def insert(self, insertedNode):
        self.fringe[insertedNode] = ""

    def pop(self, ):
        min_index = 0
        min_v = self.fringe[0].f
        for index, node in enumerate(self.fringe):
            if node.f < min_v:
                min_index = index
                min_v = node.f
        item_to_ret = self.fringe[min_index]
        del self.fringe[min_index]
        return item_to_ret

    def isEmpty(self, ):
        return len(self.fringe) == 0





def generate_children(grid, knowledge_grid, fringe, visited_list, current, all_moves, end_node, is_gridknown):
    current_x, current_y = current.position
    relevant_children = []
    dim = len(grid)
    # print("Current:- "  + str(current.position[0]) + " " + str(current.position[1]))
    for a_move in all_moves:
        child_x = current_x + a_move[0]
        child_y = current_y + a_move[1]
        if child_x > dim-1 or child_x < 0 or child_y > dim-1 or child_y < 0:
            continue
        children_node = Node(current, (child_x, child_y))

        if(is_gridknown == "No"):
            grid_for_pathcalculation = knowledge_grid
        else:
            grid_for_pathcalculation = grid
        if (grid_for_pathcalculation[child_x][child_y] != 0) and (visited_list.get(children_node.position) != "Added"):
            children_node.g = current.g + 1
            
            children_node.h = abs(children_node.position[0] - end_node.position[0]) + abs(
                    children_node.position[1] - end_node.position[1])
            children_node.f = children_node.g + children_node.h
            relevant_children.append(children_node)
    return relevant_children


def search(grid, fringe, knowledge_grid, start_position, end_position, is_gridknown,):
    startNode = Node(None, start_position)
    endNode = Node(None, end_position)

    # fringe = PriorityQueue()
    fringe = []
    visited_nodes = {}
    already_fringed = {}
    already_fringed[startNode.position] = startNode.f
    # fringe.insert(startNode)
    heapq.heappush(fringe, (startNode.f, startNode))
    all_moves = [[1, 0],
                 [0, 1],
                 [-1, 0],
                 [0, -1]]
    path = []
    while fringe:
        current = heapq.heappop(fringe)
     
        current = current[1]
        # print("current",current.position)
        visited_nodes[current.position] = "Added"
        if current.position == endNode.position:

            i = current
            while(i is not None):
                path.append(i.position)
                i = i.parent

            return "Solvable", path,
        children = generate_children(
            grid, knowledge_grid, fringe, visited_nodes, current, all_moves, endNode,is_gridknown)
        if children:
            for node in children:
                if node.position in already_fringed:
                    if already_fringed[node.position] > node.f:
                        already_fringed[node.position] = node.f
                        heapq.heappush(fringe, (node.f, node))
                else:
                    heapq.heappush(fringe, (node.f, node))
                    already_fringed[node.position] = node.f

    return "Unsolvable", path



    
    

def main(dim,is_gridknown,density,grid):
    fringe = []
    knowledge_grid = [[1 for _ in range(dim)] for _ in range(dim)]
    bumps=0

    # pltGrid = deepcopy(grid)
    # pltGrid[0][0] = 3
    # prevPos = [0,0]
    # pyplot.figure(figsize=(20, 20))
    # pyplot.grid()
    # im = None
    # print_grid(grid)
    
    start = (0, 0)
    end = (dim-1, dim-1)
    all_moves = [[1, 0],
                 [0, 1],
                 [-1, 0],
                 [0, -1]]
    

    ll, path, = search(grid, fringe, knowledge_grid,
                                                 start, end, is_gridknown)
    final_path = []
    if(ll != "Unsolvable" and is_gridknown == "No"):
        print("is_gridknown",is_gridknown)
        while(len(path) > 1 and ll!="Unsolvable"):
            count = 0
            flag = 0
            # traverse the path obtained from search function to see if blocked cell exists or not.
            # If blocked cell exists, run search function again to calculate the path
            # Continue in this while loop -1) either path returned is 0 that means nothing left in fringe and no path to reach goal 2) or path exists to reach goal
            for i in path[::-1]:
                count += 1
                
                final_path.append((i[0], i[1]))
                if(grid[i[0]][i[1]] == 0):  # blocked in grid
                    bumps+=1
                    final_path.pop()

                    # updating knowledge_grid
                    knowledge_grid[i[0]][i[1]] = 0
                    new_start_position = path[path.index(
                        i)+1][0], path[path.index(i)+1][1]

                    ll, path, = search(grid, fringe, knowledge_grid,
                                                                 new_start_position, end, is_gridknown, )
                    finalresult = ll
                    break
                elif(grid[i[0]][i[1]] == 1):
                    knowledge_grid[i[0]][i[1]] = 2
                # pltGrid[prevPos[0]][prevPos[1]] = 2
                # pltGrid[i[0]][i[1]] = 3
                # prevPos = [i[0], i[1]]
                # # pyplot.imshow(pltGrid)
                # if (im is None) :
                #     im = pyplot.imshow(pltGrid)
                # else:
                #     im.set_data(pltGrid)
                # pyplot.pause(0.5)

                if(count == len(path)):
                    print("Solved")
                    flag = 1
                # print("final_path",final_path)
                    break
            if(flag == 1):
                return final_path, knowledge_grid,bumps
                break
        if(ll=="Unsolvable"):
            return [],knowledge_grid,bumps
        if(flag != 1):
            print("finalresult", finalresult)
            return [],knowledge_grid,bumps
    elif(is_gridknown == "Yes"):
        print(ll)
        return path
        # print("path",path)
    else:
        print("Unsolvable")
        return [],knowledge_grid,bumps
    # pyplot.show()

    
    
            
    
    
    
