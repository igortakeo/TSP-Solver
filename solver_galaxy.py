from gurobipy import gurobipy as gp
from gurobipy import GRB
import numpy as np

INF = 0x3f3f3f3f

def greedy(matrix_ori):
    # copia a matriz original do problema
    matrix = np.copy(matrix_ori)
   
    # inicia as variaveis
    n = np.shape(matrix)[0]
    m = np.zeros((n,n),dtype=np.int64)
    matrix_aux = np.zeros((n+1,n),dtype=np.int64)
    galaxy_visited = 0
    local = 0
    ind_y = n+1

    # marca para n poder ir de um no para ele mesmo
    for i in range(n):
        matrix[i,i] = INF

    # conecta as galaxias
    while galaxy_visited < n:

        # calcula o y
        matrix_aux[n,local] = ind_y
        ind_y = ind_y - 1

        # acha a mais próxima
        galaxy_visited = galaxy_visited + 1
        ind = np.argmin(matrix[local,:])    

        # marca que nao pode voltar para nenhuma tentar voltar para ela (ja visitada)
        for i in range(n):
            matrix[i,local] = INF
        
        # marca por qual galaxia foi visitada
        matrix_aux[local,ind] = 1
        local = ind

    # gera o vetor a ser usado no solver como resposta inicial
    vector = []
    for i in range(n+1):
        for j in range(n):
            vector.append(int(matrix_aux[i,j]))

     # copia a matrix aux
    for i in range(n):
        for j in range(n):
            m[i,j] = matrix_aux[i,j]

    return vector, m

def cost(route,matrix):
    sum = 0
    for i in range(1, len(route)):
        sum += matrix[route[i]-1,route[i-1]-1]
    return sum


# Heuristica de Melhoria 2-opt 
# usada para melhorar o path encontrado no algoritmo guloso
# Referência: http://pedrohfsd.com/2017/08/09/2opt-part1.html 
def two_opt(initial_path, matrix):
    best_path = initial_path
    flag = True
    size_path = len(initial_path)
    while flag:
        flag = False
        i = 1
        while i < size_path-2:
            j = i+1
            while j < size_path:
                if j-1 != 1:
                    new_path = initial_path[:]
                    new_path[i:j] = initial_path[j-1:i-1:-1]
                    if cost(new_path, matrix) < cost(best_path, matrix):
                        best_path = new_path
                        flag = True
                j+=1
            i+=1
        initial_path = best_path

    return matrix_from_path(best_path)

#
#
def matrix_from_path(path):
    n = len(path) - 1
    matrix = np.zeros((n+1,n), dtype=np.int64)
    ind_y = n

    for i in range(1, n + 1):
        matrix[path[i-1] - 1, path[i] - 1] = 1
        matrix[n, path[i] - 1] = ind_y
        ind_y = ind_y - 1

    return matrix


#dados os valores que o solver encontrou para as variaveis binarias x_ij
#encontra o caminho percorrido (pontos visitados)
def path_tracer(ans_matrix):
    index = 0
    path = [1]

    n = np.shape(ans_matrix)[0]
    while True:
        for j in range(n):
            if ans_matrix[index,j] == 1:    #encontra o proximo ponto visitado a partir do atual
                index = j
                break
        
        if index == 0:          #se voltou a origem, entao fim de caminho
            path.append(1)
            break

        path.append(index+1)


    return path

def solve(matrix, flag, flag2):

    model = gp.Model("MPI_GALAXY") # Define o solver
    model.setParam('TimeLimit', 60*10) # limita o tempo em 10 minutos

    if(flag2 == 1):
        model.setParam('MIPFocus', 1) # Seta o estregia para explorar os nós da arvore
    
    if(flag2 == 2):
        model.setParam('MIPFocus', 3) # Seta o estregia para explorar os nós da arvore

    n = np.shape(matrix)[0]
    
    #Cria as variaveis
    x = [[model.addVar(0, 1, 0, vtype=GRB.INTEGER, name='x[%d,%d]' % (i,j)) for j in range(n)] for i in range(n)] #Define as variaveis binarias inteiras x_ij que assume valores 0 ou 1
    # x_ij assume 1 caso o caminho da galaxia i a galaxia j seja escolhido
    # x_ij assume 0 caso contrario

    y = [model.addVar(0, GRB.INFINITY, 0, vtype=GRB.INTEGER, name='y[%d]' % (i)) for i in range(n)] #define a variavel auxiliar y
    
    #Define as restricoes 1 e 2
    for i in range(n): 
        #Restricao 1: sum em j de {x_ij} = 1 para todo i 
        model.addConstr(sum([x[i][j] for j in range(n) if j != i]) == 1)
        #Restricao 2: sum em i de {x_ij} = 1 para todo j
        model.addConstr(sum([x[j][i] for j in range(n) if j != i]) == 1)

    #Define a Restricao 3 para verificar um subciclo ja encontrado
    for i in range(1, n):
        for j in range(1, n):
            if j != i:
                model.addConstr((y[i] - n * x[i][j]) >= y[j] - (n - 1))

    #Define a funcao objetivo - indica que desejamos MINIMIZAR a funcao objetivo
    model.setObjective(sum((matrix[i,j] * x[i][j]) for i in range(n) for j in range(n)), GRB.MINIMIZE)

    model.update()

    # heuristica gulosa
    if(flag == 2):
        vector, m = greedy(matrix)
        vars = model.getVars()
        #coloca os x e os y
        for i in range(n+1):
            for j in range(n):
                vars[(i*n)+j].start = vector[(i*n)+j]
        #coloca os y
        #for j in range(n):
        #    vars[(n*n)+j].start = GRB.UNDEFINED

    # heuristica gulosa + heuristica de melhoria 2-OPT
    if(flag == 3):
        _ , m = greedy(matrix)
        matrix_adj = two_opt(path_tracer(m), matrix)
        #print(matrix_adj)
        #exit()
        vars = model.getVars()
        #coloca os x e os y
        for i in range(n+1):
            for j in range(n):
                vars[(i*n)+j].start = matrix_adj[i, j]
        



    model.optimize()     #Chamada do solver para o modelo desenvolvido

    results = model.getVars() # pega as variaveis resolvidas

    #Obtem os valores das variaveis binarias x_ij que o solver encontrou
    ans_matrix = np.zeros((n,n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            ans_matrix[i,j] = results[(i*n)+j].x

    return path_tracer(ans_matrix)