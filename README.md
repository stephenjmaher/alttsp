# Travelling Salesman Problem with forbidden arcs

The problem provided as a version of the travelling salesman problem. In the presented setting, not all nodes in the
graph are connected, but are restricted by rules based upon their node number. In addition, the objective function is a
linear combination of two objectives: minimise the difference between the largest and smallest used edge and minimise
the path length.

I have implemented a version of two-opt to solve this problem. Given the additional constraints, prior to each two-opt
swap the feasibility of the new connections is evaluated.
