from gurobipy import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import heapq as pq 

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

# Funcao que retorna uma matriz de adjacencias
# a partir de um caminho (lista de visitados) encontrado
def matrix_from_path(path):
    n = len(path) - 1
    matrix = np.zeros((n+1,n), dtype=np.int64)
    ind_y = n

    for i in range(1, n + 1):
        matrix[path[i-1] - 1, path[i] - 1] = 1
        matrix[n, path[i] - 1] = ind_y
        ind_y = ind_y - 1

    return matrix

#########################################################

#Heuristica construtiva de christofides
def christofides(matrix):
    map_odd_vertices = dict()
    set_max_matching = set()
    n = np.shape(matrix)[0]
    matrix_ret = np.zeros((n,n),dtype=np.int64)
    
    G = nx.from_numpy_array(matrix) #constroi um grafo G completo a partir da matriz de distancias
    #G = nx.eulerize(G) # ***NECESSARIO***
    G_MST = nx.minimum_spanning_tree(G) #encontra a arvore geradora minima do grafo
    
    #acha todos os vertices de grau impar na arvore geradora minima
    odd_vertices = [] 
    even_vertices = [] 
    for vertice in G_MST.nodes():
        degree = len(G_MST.adj[vertice])
        if degree % 2 == 1:
            odd_vertices.append(vertice)
            map_odd_vertices[len(odd_vertices)-1] = vertice
        else:
            even_vertices.append(vertice)

    #monta a matriz de adjacencias contendo apenas os vertices com grau impar na arvore geradora minima
    matrix_sub = np.delete(matrix, even_vertices, axis = 1)         #remove todos os vertices de grau par
    matrix_sub = np.delete(matrix_sub, even_vertices, axis= 0)      

    #como a funcao da biblioteca networkx trabalha com matching perfeito de maximo custo
    #constroi o grafo de peso de arestas complementar para achar o matching perfeito de minimo custo
    max_weight_edge = np.max(matrix_sub)                #acha o maior peso de aresta
    matrix_sub = (matrix_sub*(-1)) + max_weight_edge    #acha a matriz de pesos complementar
    matrix_sub = matrix_sub + 1

    G_matching = nx.from_numpy_array(matrix_sub)    #constroi o grafo com a matriz de adjacencias complementar
    set_max_matching = nx.max_weight_matching(G_matching, maxcardinality = True)
    
    for x, y in set_max_matching:
        u = map_odd_vertices[x]
        v = map_odd_vertices[y] 
        G_MST.add_edge(u, v, weight=matrix[u, v])

    G_MST = nx.eulerize(G_MST)

    flag_break = True

    #Se todos os nos tiverem grau 2, terminou o algoritmo
    for vertice in G_MST.nodes(): 
        degree = len(G_MST.adj[vertice])
        if degree != 2:
            flag_break = False 
            break
    
    #Encontrando o caminho euleriano e terminando o algoritmo se flag_break = True 
    if flag_break: 
        eulerian_route = list(nx.eulerian_circuit(G_MST, source=0))
        for x, y in eulerian_route:
            matrix_ret[x,y] = 1
        return matrix_ret

    heap = []
    degree_nodes = [0]*n

    #Loop para pegar o grau de cada no de acordo com as arestas do grafo
    for edge in G_MST.edges(): 
        degree_nodes[edge[0]]+=1
        degree_nodes[edge[1]]+=1

    #Inserindo os nos na fila de prioridades
    for i in range(n): 
        pq.heappush(heap, (-degree_nodes[i], i))


    #Realizando o shortcut
    while heap[0][0]*-1 > 2:  
        node = pq.heappop(heap)
        w = node[1] #No com o maior grau
        degree_w = node[0]*-1 
        map_adj = G_MST.adj[w]
        list_adj = list() #Lista de adjacentes do no w
        for i in map_adj:
            list_adj.append(i)
        flag = False
        for i in range(len(list_adj)):
            for j in range(i+1, len(list_adj)):
                x = list_adj[i]
                y = list_adj[j]
                if matrix[x,y] <= matrix[x,w] + matrix[w,y]: #Desigualdade triangular
                    Graph_copy = G_MST.copy()
                    Graph_copy.remove_edge(x,w) 
                    Graph_copy.remove_edge(w,y)
                    Graph_copy.add_edge(x,y, weight=matrix[x,y])
                    if nx.is_connected(Graph_copy): #Verificando se o grafo permaneceu conexo 
                        G_MST = Graph_copy.copy()
                        flag = True
                        break
            if flag: 
                break
        degree_w -= 2 
        pq.heappush(heap, (-degree_w, w))
    
    #Encontrando o caminho euleriano    
    eulerian_route = list(nx.eulerian_circuit(G_MST, source=0))

    #Montando a matriz de adjacencia
    for x, y in eulerian_route:
        matrix_ret[x,y] = 1
    
    return matrix_ret    



    
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
    model.setParam('TimeLimit', 60*30) # limita o tempo em 10 minutos

    if(flag2 == 2):
        model.setParam('MIPFocus', 1) # Seta o estregia para explorar os nós da arvore
    
    if(flag2 == 3):
        model.setParam('MIPFocus', 2) # Seta o estregia para explorar os nós da arvore

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
    
    # heuristica do algoritmo de christofides
    if(flag == 4):
        matrix_adj = christofides(matrix)
        exit()


    model.optimize()     #Chamada do solver para o modelo desenvolvido

    results = model.getVars() # pega as variaveis resolvidas

    #Obtem os valores das variaveis binarias x_ij que o solver encontrou
    ans_matrix = np.zeros((n,n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i != j:
                ans_matrix[i,j] = results[(i*n)+j].x

    return path_tracer(ans_matrix)
