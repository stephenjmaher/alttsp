"""
Assessment solution
Author: Stephen J. Maher
"""

import sys
import os
import timeit
import numpy as np

INFINITY = 1e+20

class Graph:
    """
    storage of the graph
    """

    def __init__(self):
        self.maxdist = 0    # the largest distance between nodes
        self.nodes = []     # the number of nodes

    def add_node(self, node):
        """
        adds a node to the graph
        """
        # updates the distance between the nodes
        for i in self.nodes:
            if node.compute_distance(i) > self.maxdist:
                self.maxdist = node.compute_distance(i)

        self.nodes.append(node)

    def num_nodes(self):
        """
        returns the number of nodes
        """
        return len(self.nodes)

    def are_connected(self, idx1, idx2):
        """
        returns whether two nodes are connected
        """
        return self.nodes[idx1].is_connected(idx2, self.num_nodes())

    def compute_distance(self, idx1, idx2):
        return self.nodes[idx1].compute_distance(self.nodes[idx2])

class Node:
    """
    representation of a node in the graph
    """

    def __init__(self, idx, x, y):
        self.idx = idx      # the node index
        self.x = x          # the x coordinate
        self.y = y          # the y coordinate

    def compute_distance(self, checknode):
        """
        computes the distance between the current node and an input node
        """
        return ((self.x - checknode.x)**2 + (self.y - checknode.y)**2)**0.5

    def is_connected(self, idx, numnodes):
        """
        returns whether two nodes are connected
        """
        if self.idx % 2 == 0 and self.idx < float(numnodes)/2 and (idx + 1) % 2 == 0:
            return False
        elif (self.idx + 1) % 2 == 0 and self.idx >= float(numnodes)/2 and idx % 2 == 0:
            return False
        else:
            return True


class Route:
    """
    the tsp route
    """

    def __init__(self, graph):
        self.maxedge = (-1, -1, 0)              # tuple representing the edge with the largest distance
        self.minedge = (-1, -1, INFINITY)       # tuple representing the edge with the smallest distance
        self.length = 0                         # the length of the route (not the objective function)
        self.distance = 0                       # the objective value of the route
        self.sequence = []                      # the route
        self.graphmaxdist = graph.maxdist       # graph data:
        self.numnodes = graph.num_nodes()

        # generate the initial route from the input graph
        self.generate_initial_route(graph)


    def generate_initial_route(self, graph):
        """
        generates the initial route
        """
        nodelist = [x for x in graph.nodes]
        self.sequence.append(nodelist[0])
        nodeidx = 0
        currnode = nodelist[nodeidx]
        nodelist[nodeidx] = nodelist[-1]
        nodelist.pop()

        # iterates through the list and pops nodes when added to the route.
        # if not all nodes are added in a pass, then loop is restarted.
        # node will not be added if the connections are infeasible
        while len(nodelist) > 0:
            assert(nodeidx < len(nodelist))
            nextnode = nodelist[nodeidx]
            if currnode.is_connected(nextnode.idx, self.numnodes):
                self.sequence.append(nextnode)
                nodelist[nodeidx] = nodelist[-1]
                nodelist.pop()
                currnode = nextnode
                if nodeidx == len(nodelist):
                    nodeidx = 0
            else:
                if nodeidx == len(nodelist) - 1:
                    nodeidx = 0
                else:
                    nodeidx += 1


        self.length, self.minedge, self.maxedge = self.compute_route_length(self.sequence)
        self.distance = (self.maxedge[2] - self.minedge[2])*self.graphmaxdist*(self.numnodes - 2) + self.length


    def compute_route_length(self, sequence):
        """
        computes the distance of the route
        """
        length = 0
        maxedge = (-1, -1, 0)
        minedge = (-1, -1, INFINITY)
        for i, node in enumerate(sequence):
            checkidx = i + 1
            if i == len(sequence) - 1:
                checkidx = 0
            currlength = node.compute_distance(sequence[checkidx])
            if currlength < minedge[2]:
                minedge = (node.idx, sequence[checkidx].idx, currlength)
            if currlength > maxedge[2]:
                maxedge = (node.idx, sequence[checkidx].idx, currlength)
            length += currlength

        return length, minedge, maxedge

    def compute_route_distance(self):
        """
        computes the distance of the current route
        """
        self.length, self.minedge, self.maxedge = self.compute_route_length(self.sequence)
        self.distance = (self.maxedge[2] - self.minedge[2])*self.graphmaxdist*(self.numnodes - 2) + self.length

        return self.distance

    def compute_sequence_distance(self, sequence):
        """
        computes sequence distance
        """
        length, minedge, maxedge = self.compute_route_length(sequence)
        distance = (maxedge[2] - minedge[2])*self.graphmaxdist*(self.numnodes - 2) + length

        return distance

    def check_swap(self, i, j):
        """
        check potential swap candidates
        """
        # check whether the new connections are feasible
        if self.sequence[i - 1].is_connected(self.sequence[j].idx, self.numnodes) and \
                ((j + 1 < self.numnodes and self.sequence[i].is_connected(self.sequence[j + 1].idx, self.numnodes)) or
                        (j + 1 == self.numnodes and self.sequence[i].is_connected(self.sequence[0].idx,
                            self.numnodes))):
            subsequence = []
            subsequence.extend(reversed(self.sequence[i:j+1]))
            # checking whether a subsequence is feasible once reversed
            return self.check_subsequence(subsequence)
        else:
            return False

        return True

    def check_subsequence(self, sequence):
        """
        checks whether a subsequence is feasible
        """
        for i, node in enumerate(sequence):
            checkidx = i + 1
            if checkidx >= len(sequence):
                checkidx = 0

            # checks whether the nodes are connected
            if not node.is_connected(sequence[checkidx].idx, self.numnodes):
                return False

        return True

    def distance(self):
        """
        helper function returning the distance
        """
        return self.distance

    def perform_twoopt_swap(self, i, j):
        """
        performs a two-opt swap
        """
        swap = False
        # only perform the swap if the new connections are feasible
        if self.check_swap(i, j):
            currdist = self.distance
            newsequence = self.sequence[0:i]
            newsequence.extend(reversed(self.sequence[i:j + 1]))
            newsequence.extend(self.sequence[j+1:])
            assert(len(self.sequence) == len(newsequence))
            # complete the swap if the new route is less than the old route
            if currdist > self.compute_sequence_distance(newsequence):
                self.sequence = newsequence
                self.compute_route_distance()
                swap = True

        return swap

    def print_route(self):
        """
        prints the new route
        """
        print "Route:",
        for i, node in enumerate(self.sequence):
            if i < self.numnodes - 1:
                print "%d -"%int(node.idx),
            else:
                print int(node.idx)

    def print_solution(self, instance, runtime):
        """
        prints the solution to a file
        """
        dirname = os.path.dirname(instance)
        filename = os.path.splitext(os.path.basename(instance))[0]
        solfile = "%s/result%s.txt"%(dirname,filename)
        f = open(solfile, 'w')
        f.write("Route:")


        for i, node in enumerate(self.sequence):
            if i < self.numnodes - 1:
                f.write("%d-"%int(node.idx))
            else:
                f.write("%d\n"%int(node.idx))

        f.write("Total Distance: %f\n"%self.length)
        f.write("Delta value: %f\n"%(self.maxedge[2] - self.minedge[1]))
        f.write("Run time: %f"%(runtime))

        f.close()


    def check_route(self):
        """
        checks whether the route is feasible
        """
        self.print_route()
        if not len(self.sequence) == self.numnodes:
            return False

        maxedge = 0
        minedge = INFINITY
        for i, node in enumerate(self.sequence[:-1]):
            maxedge = max(maxedge, node.compute_distance(self.sequence[i + 1]))
            minedge = min(minedge, node.compute_distance(self.sequence[i + 1]))
            if not node.is_connected(self.sequence[i + 1].idx, self.numnodes):
                print "(%d, %d) is not a feasible connection\n"%(int(node.idx), int(self.sequence[i + 1].idx))
                return False

        maxedge = max(maxedge, self.sequence[-1].compute_distance(self.sequence[0]))
        minedge = min(minedge, self.sequence[-1].compute_distance(self.sequence[0]))
        if not self.sequence[-1].is_connected(self.sequence[0].idx, self.numnodes):
            print "(%d, %d) is not a feasible connection\n"%(int(self.sequence[-1].idx), int(self.sequence[0].idx))
            return False

        if not maxedge == self.maxedge[2]:
            print "Max Edge: %f != %f\n"%(maxedge, self.maxedge[2])
            return False

        if not minedge == self.minedge[2]:
            print "Min Edge: %f != %f\n"%(minedge, self.minedge[2])
            return False

        return True

def run_twoopt(graph, route):
    """
    improves an existing route using the 2-opt swap until no improved route is found
    best path found will differ depending of the start node of the list of nodes
        representing the input tour
    returns the best path found
    graph -  the graph that the route is being found for
    route - route to improve
    """
    improvement = True
    while improvement:
        improvement = False
        for i in range(1, graph.num_nodes() - 1):
            for j in range(i+1, graph.num_nodes()):
                if route.perform_twoopt_swap(i, j):
                    improvement = True
                    break #improvement found, return to the top of the while loop

            if improvement:
                break
    return route


def readInstance(filename):
    """
    reads the input instance
    """
    instance = np.genfromtxt(filename, delimiter=",", skip_header=1)

    graph = Graph()

    for n in instance:
        node = Node(n[0], n[1], n[2])
        graph.add_node(node)

    return graph


def main():
    """
    the main function (very informative)
    """

    if len(sys.argv) == 1:
        print "No file has been input. Please input a TSP instance."
        return

    start = timeit.default_timer()

    graph = readInstance(sys.argv[1])

    # create the initial route
    route = Route(graph)

    newroute = run_twoopt(graph, route)

    assert(newroute.check_route())

    stop = timeit.default_timer()

    newroute.print_solution(sys.argv[1], stop - start)

    print "Run time: %f"%(stop - start)






if __name__ == "__main__":
    main()
