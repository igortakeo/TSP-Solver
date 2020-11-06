from ortools.sat.python import cp_model
import numpy as np

INF = 0x3f3f3f3f

def path_tracer(ans_matrix):
    index = 0
    path = [1]

    n = np.shape(ans_matrix)[0]
    while True:
        for j in range(n):
            if ans_matrix[index,j] == 1:
                index = j
                break
        
        if index == 0:
            break

        path.append(index+1)


    return path

def solve(matrix):
    model = cp_model.CpModel() #Define o modelo

    n = np.shape(matrix)[0]
    
    #Cria as variaveis
    x = [[model.NewIntVar(0, 1, 'x[%d,%d]' % (i,j)) for j in range(n)] for i in range(n)] #Define as variaveis binarias inteiras x_ij que assume valores 0 ou 1
    # x_ij assume 1 caso o caminho da galaxia i a galaxia j seja escolhido
    # x_ij assume 0 caso contrario

    y = [model.NewIntVar(0, INF, 'y[%d]' % (i)) for i in range(n)] #define a variavel auxiliar y

    func_obj = model.NewIntVar(0, INF, 'func_obj') #Define a variavel da funcao objetivo com intervalo [0, INF] 
    
    #Define as restricoes 1 e 2
    for i in range(n): 
        #Restricao 1: sum em j de {x_ij} = 1 para todo i
        model.Add(sum([x[i][j] for j in range(n) if j != i]) == 1)
        #Restricao 2: sum em i de {x_ij} = 1 para todo j
        model.Add(sum([x[j][i] for j in range(n) if j != i]) == 1)

    #Define a Restricao 3 para verificar um subciclo ja encontrado
    for i in range(1, n):
        for j in range(1, n):
            if j != i:
                model.Add((y[i] - n * x[i][j]) >= y[j] - (n - 1))

    #Define a funcao objetivo
    model.Add(func_obj == sum((matrix[i,j] * x[i][j]) for i in range(n) for j in range(n)))
    model.Minimize(func_obj) #indica que desejamos MINIMIZAR a funcao objetivo

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0 * 60.0
    solver.Solve(model)

    ans_matrix = np.zeros((n,n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            ans_matrix[i,j] = solver.Value(x[i][j])

    return path_tracer(ans_matrix), solver.Value(func_obj)