import pandas as pd
import numpy as np
from math import sqrt
import heapq,time
from matplotlib import pyplot
from copy import deepcopy
import matplotlib.pylab as plt




class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.f = 0
        self.h = 0

    def __eq__(self, node):
        if (self.position[0] == node.position[0] and self.position[1] == node.position[1]) :
            return True
        else:
            return False
    
    def __lt__(self, other):
        return self.f < other.f


def generate_children(grid, visited_list, current, all_moves, end_node):
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

       
        if (grid[child_x][child_y] != 0) and (visited_list.get(children_node.position) != "Added"  ):
            children_node.g = current.g + 1
            children_node.h = abs(children_node.position[0] - end_node.position[0]) + abs(
                children_node.position[1] - end_node.position[1])
           
            children_node.f = children_node.g + children_node.h
            relevant_children.append(children_node)
    return relevant_children


def search(grid, fringe, start_position, end_position):
    startNode = Node(None, start_position)
    endNode = Node(None, end_position)

    fringe=[]
    visited_nodes = {}
    already_fringed = {}
    already_fringed[startNode.position] = startNode.f
    heapq.heappush(fringe,(startNode.f,startNode))
    all_moves = [[1, 0],
                 [0, 1],
                 [-1, 0],
                 [0, -1]]
    path = []
    while  fringe:
        current = heapq.heappop(fringe)
        current=current[1]
        visited_nodes[current.position]="Added"
        if current.position== endNode.position:

            i = current
            while(i is not None):
                path.append(i.position)
                i = i.parent

            return "Solvable", path
        children = generate_children(
            grid, visited_nodes, current, all_moves, endNode)
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



def main(grid,dim):
    fringe=[]
    

    
    # assuming unblocked for all cells
   
    # pltGrid = deepcopy(grid)
    # pltGrid[0][0] = 3
    # prevPos = [0,0]
    # pyplot.figure(figsize=(20, 20))
    # pyplot.grid()
    im = None

    
    start = (0, 0)
    end = (dim-1, dim-1)
    
    
    
    
    ll, path = search(grid,fringe, start, end)
    return (ll,path)
    
    

   
    
    