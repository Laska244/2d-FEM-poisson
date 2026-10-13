import math

def funkce3(x,y):
    return math.sin(x)

def funkce122(x,y):
    return 1

def right_side(x,y):
    return 0

input_data = {
    "nodes" : [[0,0],[1,0],[2,1],[-0.5,0.3]],

    
    "dirichlet" : [
        [3, funkce3],[1,funkce122]],


    "right_side" : right_side,
    "mesh_size" : 0.05
}
