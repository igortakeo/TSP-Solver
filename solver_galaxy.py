from ortools.linear_solver import pywraplp
import numpy as np

INF = 0x3f3f3f3f

def greedy(matrix_ori):

    matrix = np.copy(matrix_ori)

    n = np.shape(matrix)[0]
    m = np.zeros((n,n),dtype=np.int64)
    matrix_aux = np.zeros((n+1,n),dtype=np.int64)
    galaxy_visited = 0
    local = 0
    ind_y = n+1

    for i in range(n):
        matrix[i,i] = INF

    while galaxy_visited < n:

        matrix_aux[n,local] = ind_y
        ind_y = ind_y - 1

        galaxy_visited = galaxy_visited + 1
        ind = np.argmin(matrix[local,:])    
        
        for i in range(n):
            matrix[i,local] = INF
        
        matrix_aux[local,ind] = 1
        local = ind

    result = 0
    for i in range (n):
        for j in range(n):
            if matrix_aux[i,j] == 1:
                result += matrix_ori[i,j]


    vector = []
    for i in range(n+1):
        for j in range(n):
            vector.append(int(matrix_aux[i,j]))

 
    for i in range(n):
        for j in range(n):
            m[i,j] = matrix_aux[i,j]

    return vector, result, m


def cost(route,matrix):
    sum = 0
    for i in range(1, len(route)):
        sum += matrix[route[i]-1,route[i-1]-1]
    return sum

def two_opt(route, matrix):
     best = route
     improved = True
     while improved:
          improved = False
          for i in range(1, len(route)-2):
               for j in range(i+1, len(route)):
                    if j-i == 1: continue # changes nothing, skip then
                    new_route = route[:]
                    new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
                    if cost(new_route,matrix) < cost(best,matrix):
                         best = new_route
                         improved = True
          route = best
     return best

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


def solve(matrix, flag):
    solver = pywraplp.Solver.CreateSolver('SCIP') #Define o modelo

    n = np.shape(matrix)[0]
    
    #Cria as variaveis
    x = [[solver.IntVar(0, 1, 'x[%d,%d]' % (i,j)) for j in range(n)] for i in range(n)] #Define as variaveis binarias inteiras x_ij que assume valores 0 ou 1
    # x_ij assume 1 caso o caminho da galaxia i a galaxia j seja escolhido
    # x_ij assume 0 caso contrario

    infinity = solver.infinity()

    y = [solver.IntVar(0, infinity, 'y[%d]' % (i)) for i in range(n)] #define a variavel auxiliar y
    
    #Define as restricoes 1 e 2
    for i in range(n): 
        #Restricao 1: sum em j de {x_ij} = 1 para todo i 
        solver.Add(sum([x[i][j] for j in range(n) if j != i]) == 1)
        #Restricao 2: sum em i de {x_ij} = 1 para todo j
        solver.Add(sum([x[j][i] for j in range(n) if j != i]) == 1)

    #Define a Restricao 3 para verificar um subciclo ja encontrado
    for i in range(1, n):
        for j in range(1, n):
            if j != i:
                solver.Add((y[i] - n * x[i][j]) >= y[j] - (n - 1))

    if(flag == 2):
        vector, result, m = greedy(matrix)
        print(result)
        path = path_tracer(m)
        print(path)
        path = two_opt(path, matrix)
        result = cost(path, matrix)
        print(result)
        print(path)
        solver.SetHint(solver.variables(), vector)

    #Define a funcao objetivo - indica que desejamos MINIMIZAR a funcao objetivo
    #Também define a funcao objetivo
    solver.Minimize(sum((matrix[i,j] * x[i][j]) for i in range(n) for j in range(n)))

    solver.SetTimeLimit(600000)  #Limite de tempo de 10 minutos
    solver.Solve()     #Chamada do solver para o modelo desenvolvido

    #Obtem os valores das variaveis binarias x_ij que o solver encontrou
    ans_matrix = np.zeros((n,n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            ans_matrix[i,j] = x[i][j].solution_value() 

    print("Nós:" + str(solver.nodes()))

    return path_tracer(ans_matrix), solver.Objective().Value()