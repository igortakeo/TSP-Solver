import numpy as np
import math
from solver_galaxy import solve, INF
import sys

#sys.stdin = open('teste.txt', 'r')

precision = 3

#Funcao responsavel pelo calculo preenchimento da matrix de adjacencia das galaxias com a distancia entre cada uma
def gen_matrix_dist(coords):
    n = len(coords)
    matrix = np.zeros((n,n))
    
    for i in range(n):          #Calcula a distancia euclidiana da galaxia i a galaxia j
        for j in range(i, n):
            matrix[i,j] = matrix[j,i] =  math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2) 
    
    return matrix


ngalaxys = int(input())         #Input do numero de galaxias do problema em questao

coords = []

for i in range(ngalaxys):       #Input das coordenadas de cada galaxia
    x = float(input())
    y = float(input())
    coords.append((x,y))

matrix = gen_matrix_dist(coords)
matrix_int = np.zeros((ngalaxys,ngalaxys), dtype=np.int64)

for i in range(ngalaxys):
    for j in range(ngalaxys):
        matrix_int[i,j] = int(matrix[i,j] * (10 ** precision))       

path, distance = solve(matrix_int)
print(path)
print(float(distance)/(10 ** precision))