import sumolib

net = sumolib.net.readNet('data/grid.net.xml')
print([i.getCoord() for i  in net.getNodes()])
print(net.getEdges())