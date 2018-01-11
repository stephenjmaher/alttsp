"""
Assessment solution
Author: Stephen J. Maher
"""

import sys
import numpy as np

INFINITY = 1e+20

class Graph:
    """
    storage of the graph
    """

    def __init__(self):
        self.maxdist = 0
        self.nodes = []

    # adds a node to the graph
    def add_node(self, node):
        for i in self.nodes:
            if node.compute_distance(i) > maxdist:
                maxdist = node.compute_distance(i)

        self.nodes.append(node)

    # returns the number of nodes
    def num_nodes(self):
        return len(self.nodes)

    # returns whether two nodes are connected
    def are_connected(self, idx1, idx2):
        if idx1 % 2 == 0 and idx1 < float(self.numnodes())/2 and (idx2 + 1) % 2 == 0:
            return False
        elif (idx1 + 1) % 2 == 0 and idx1 >= float(self.numnodes())/2 and idx2 % 2 == 0:
            return False
        else:
            return True:

class Node:
    """
    representation of a node in the graph
    """

    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = x
        self.y = y

    # computes the distance between the current node and an input node
    def compute_distance(self, checknode):
        return ((self.x + checknode.x)**2 + (self.y + checknode.y)**2)**0.5


class Route:
    """
    the tsp route
    """

    def __init__(self, graph):
        self.maxedge = 0
        self.minedge = INFINITY
        self.length = 0
        self.distance = 0
        self.route = []
        self.generate_initial_route(graph)


    # generates the initial route
    def generate_initial_route(self, graph):
        for node in graph.nodes:
            route.append(node.idx)

        self.length, self.minedge, self.maxedge = self.compute_route_length()
        self.distance = (self.maxedge - self.minedge)*graph.maxdist*(graph.num_node() - 1) + self.length


    # computes the distance of the route
    def compute_route_length(self):
        length = 0
        maxedge = 0
        minedge = INFINITY
        for i, node in enumerate(self.route[:-1]):
            currlength = node.compute_distance(self.route[i + 1])
            minedge = min(currlength, minedge)
            maxedge = max(currlength, maxedge)
            length += currlength

        return length, minedge, maxedge

    # computes the distance of the current route
    def compute_route_distance(self):
        self.length, self.minedge, self.maxedge = self.compute_route_length()
        self.distance = (self.maxedge - self.minedge)*graph.maxdist*(graph.num_node() - 1) + self.length

        return self.distance


def swap_2opt(route, i, k):
    """
    swaps the endpoints of two edges by reversing a section of nodes, 
        ideally to eliminate crossovers
    returns the new route created with a the 2-opt swap
    route - route to apply 2-opt
    i - start index of the portion of the route to be reversed
    k - index of last node in portion of route to be reversed
    pre: 0 <= i < (len(route) - 1) and i < k < len(route)
    post: length of the new route must match length of the given route 
    """
    assert i >= 0 and i < (len(route) - 1)
    assert k > i and k < len(route)
    new_route = route[0:i]
    new_route.extend(reversed(route[i:k + 1]))
    new_route.extend(route[k+1:])
    assert len(new_route) == len(route)
    return new_route

def run_2opt(route):
    """
    improves an existing route using the 2-opt swap until no improved route is found
    best path found will differ depending of the start node of the list of nodes
        representing the input tour
    returns the best path found
    route - route to improve
    """
    improvement = True
    best_route = route
    best_distance = route_distance(route)
    while improvement: 
        improvement = False
        for i in range(len(best_route) - 1):
            for k in range(i+1, len(best_route)):
                new_route = swap_2opt(best_route, i, k)
                new_distance = route_distance(new_route)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_route = new_route
                    improvement = True
                    break #improvement found, return to the top of the while loop
            if improvement:
                break
    assert len(best_route) == len(route)
    return best_route


def main():



def __name__ == "__main__":
    main()
