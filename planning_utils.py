from enum import Enum
from queue import PriorityQueue
import numpy as np
from bresenham import bresenham


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    # (Northing, Easting, Cost)
    WEST = (0, -1, 1.)
    EAST = (0, 1, 1.)
    NORTH = (1, 0, 1.)
    SOUTH = (-1, 0, 1.)
    NORTHWEST = (1, -1, 2.**0.5)
    NORTHEAST = (1, 1, 2.**0.5)
    SOUTHWEST = (-1, -1, 2.**0.5)
    SOUTHEAST = (-1, 1, 2.**0.5)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    all_actions = list(Action)
    valid_actions = []
    
    n, m = grid.shape[0], grid.shape[1]
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle
    
    for action in all_actions:
        # get the tuple representation
        da = action.delta
        if (x + da[0] >= 0) and (x + da[0] < n) and (y + da[1] >= 0) and (y + da[1] < m) and (grid[x + da[0], y + da[1]] == 0):
            valid_actions.append(action)

    return valid_actions


def a_star(grid, h, start, goal):

    path = []
    path_cost = 0.0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
            
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost



def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


    
def combine_segments_colinear(path):
    if len(path) < 3:
        # if the path is less than 3 points, then there is nothign to prune....
        return path
        
    # start new path list with first (start) point    
    new_path = []
    new_path.append(path[0])
    
    dx_last = path[1][0] - path[0][0]
    dy_last = path[1][1] - path[0][1]
    
    # check if intermediate points are elbows/turns
    for k in range(1,len(path)-1):
        dx = path[k + 1][0] - path[k][0]
        dy = path[k + 1][1] - path[k][1]
        if dx_last != dx or dy_last!= dy:
            # this is an elbow/turn, append point to new path list
            dx_last = dx
            dy_last = dy
            new_path.append(path[k])
     
    # add last points to new path list
    new_path.append(path[-1])
    
    return new_path
    
def combine_segments_bresenham(path, grid):
    if len(path) < 3:
        # if the path is less than 3 points, then there is nothign to prune....
        return path
        
    # start new path list with first (start) point    
    new_path = []
    new_path.append(path[0])
    p_last = path[0]
    k_last = 0
    
    for k in range(1, len(path) - 1):
        # use bresenham to find grid cell between points
        line_cells = list(bresenham(p_last[0], p_last[1], path[k + 1][0], path[k + 1][1]))
        
        # check if next point causes a collision
        collision = False
        for c in line_cells:
            # if next points causes a collision then use current points  
            if (1 == grid[c[0]][c[1]]):
                new_path.append(path[k])
                p_last = path[k]
                k_last = k
                break
            # else try next points...
    
    # ensure that last point is included in new path list...regardless
    new_path.append(path[-1])    

    return new_path
    
    
    
    
    