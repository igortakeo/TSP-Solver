import numpy as np
import math
from solver_galaxy import solve, INF
import sys
import matplotlib.pyplot as plt

#Funcao responsavel por plotar os pontos do problema e o caminho entre eles encontrado pelo solver
def printPath(coords, path, name):
  coords_unzip = list(zip(*coords)) 

  xp = []
  yp = []
  for p in path:
    xp.append(coords_unzip[0][p-1])
    yp.append(coords_unzip[1][p-1])  

  plt.figure(figsize=(20,20),facecolor='white')
  plt.plot(xp, yp,label = 'Solution path',linewidth = 2) #Plota o caminho encontrado
  plt.scatter(coords_unzip[0], coords_unzip[1],label = 'Problem points',linewidth = 2, c = 'red') #Plota os pontos do sistema
  plt.scatter(coords_unzip[0][0], coords_unzip[1][0],label = 'Initial point',linewidth = 2, c = 'orange') #Plota o ponto inicial
  plt.xlabel('x',fontsize='large') 
  plt.ylabel('y',fontsize='large') 
  plt.title('Solution for ' + name) 
  plt.legend() 
  plt.show()


#Funcao responsavel por obter os dados dado um arquivo .tsp de entrada, conforme os arquivos da especificacao
def input_parser(input_name):
    coords = []

    try:
        f = open(input_name, 'r')
    except OSError:
        print('Could not read file ' + input_name)      #Caso haja algum problema na abertura (Como arquivo inexistente)
        sys.exit()

    name = ''
    IsPointInput = False

    for line in f:                  
        line_info = line.split()

        if not IsPointInput:
            if line_info[0] == 'NAME:':
                name = line_info[1]     
            
            if line_info[0] == 'NODE_COORD_SECTION':    #Indica que abaixo virao os dados de cada ponto do problema
                IsPointInput = True
        else:
            coords.append((float(line_info[1]), float(line_info[2])))

    f.close()

    return name, coords



#Funcao responsavel pelo calculo preenchimento da matrix de adjacencia das galaxias com a distancia entre cada uma
def gen_matrix_dist(coords):
    n = len(coords)
    matrix = np.zeros((n,n))
    
    for i in range(n):          #Calcula a distancia euclidiana da galaxia i a galaxia j
        for j in range(i, n):
            matrix[i,j] = matrix[j,i] =  math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2) 
    
    return matrix


#input do programa
test_file_name = input('Insert test file name: ')
name, coords = input_parser(test_file_name)
ngalaxys = len(coords)

matrix = gen_matrix_dist(coords)

path, distance = solve(matrix)       #soluciona o problema dada a matriz de distancias do problema
print('Solution path (Points visited):')
print(path)                              #caminho percorrido (pontos visitados)
print('Distance travelled: %f' %(distance)) #distancia total percorrida
printPath(coords, path, name)           #plota o caminho percorrido