import random
import numpy as np

# Input function for describing genetic algorithm
def func(x,y):
    result = x * np.sin(4 * x) + 1.1 * y * np.sin(2 * y)
    return result

# Generate random points function
def generate_points(n, x_array, y_array):
    #random.seed()
    for i in range (n):
        x_array[i] = 10 * random.random()
        y_array[i] = 10 * random.random()       
    return x_array, y_array

#Selection function for choosing survival populations
def sort_array(array):
    array_1 = array.copy()
    result = np.zeros([len(array)])
    index = np.zeros([len(array)], dtype = int)
    ind = 0
    while (ind<n):
        min = np.min(array_1)
        ind_min = np.where(array == min)
        index[ind] = ind_min[0][0]
        result[ind] = min
        array_1 = np.delete(array_1, np.where(array_1 == min)[0][0])
        ind += 1
    return result, index

#Sorting corresponding input parameters according to output variable sorting result 
def rearrange_array(x_array,y_array,index_child):
    x_array_child = x_array.copy()
    y_array_child = y_array.copy()
    for i in range(len(x_array)):
        x_array_child[i] = x_array[index_child[i]]
        y_array_child[i] = y_array[index_child[i]]
    x_array = x_array_child
    y_array = y_array_child
    return x_array, y_array

#Crossover function for generating childhood population
def crossover_chromosomes(x_array,mate_index_1,mate_index_2):
    n = len(x_array)
    N_keep = n//2
    #random.seed()
    beta = random.random()
    x_new_1 = (1-beta) * x_array[mate_index_1] + beta * x_array[mate_index_2]
    x_new_2 = (1-beta) * x_array[mate_index_2] + beta * x_array[mate_index_1]
    return x_new_1, x_new_2

#Calculation function for probability rate and ranking number for further crossover operation
def ranking_calculation(z_array):
    probability_rate = np.zeros(z_array.shape)
    N = len(probability_rate)
    N_keep = N//2
    for i in range(N_keep):
        probability_rate[i] = (N_keep - i) / (N_keep * (N_keep + 1) / 2)
    ranking = np.zeros(z_array.shape)
    for i in range(1,N_keep+1):
        for j in range (0,i):
            ranking[i] += probability_rate[j]
    return probability_rate, ranking

#Function for generation childhood chromosomes by crossover operation
def generate_child_chromosomes(x_array,y_array,ranking):
    N = len(x_array)
    N_keep = N//2
    for m in range(0,N_keep,2):
        #random.seed()
        rand_1 = np.random.uniform()
        rand_2 = np.random.uniform()
        for i in range (N_keep+1):
            if rand_1 > ranking[i] and rand_1 <= ranking[i+1]:
                mate_index_1 = i
                break
        for i in range (N_keep+1):
            if rand_2 > ranking[i] and rand_2 <= ranking[i+1]:
                mate_index_2 = i
                break
        while (mate_index_1 == mate_index_2):
            mate_index_2 = random.randint(0,N_keep)    
        x_array[N_keep+m], x_array[N_keep+m+1] = crossover_chromosomes(x_array,mate_index_1,mate_index_2)
    return x_array, y_array
#Duplication check for mutation process
def duplication_check_mutation(index):
    N = index.shape[1]
    logic = False
    for i in range(N):
        for j in range (N):
            index_1 = index[:,i]
            index_2 = index[:,j]
            if j!=i and (index_1 == index_2).all():
                logic = True
                break
    return logic
    
#Function for assigning value for chosen input variables due to mutation process
def mutation_function(x_array, y_array, mutation_rate):
    N = len(x_array)
    mutation_count = np.round((N-1) * mutation_rate * 2)
    mutation_count = mutation_count.astype(int)
    combined_array = np.vstack((x_array,y_array))
    index_mute = np.zeros([2,mutation_count], dtype = int)
    #random.seed()
    while (duplication_check_mutation(index_mute)):
        for i in range(0,mutation_count):
            index_mute[0,i] = random.randint(0,1)
            index_mute[1,i] = random.randint(1,mutation_count-1)
    for i in range(index_mute.shape[1]):
        location = index_mute[:,i]
        combined_array[location[0], location[1]] = 10 * random.random()
    x_array = combined_array[0,:]
    y_array = combined_array[1,:]
    return x_array, y_array
    




#Declare number  of known values for input function
n = 12
index = np.arange(0,n)

#Generate input chromosomes in main program
x_array = np.zeros(n, dtype = float)
y_array = np.zeros(n, dtype = float)
generate_points(n, x_array, y_array)
z_array = func(x_array, y_array)

print(x_array)
print(y_array) 
print(z_array)
min = np.min(z_array)
print("Minimum value of function is: ", min)
for q in range (10000):
    #Sorting value of known variables and their corresponding input parameters
    (z_array, index_child) = sort_array(z_array)
    x_array, y_array = rearrange_array(x_array,y_array,index_child)
    #Calculating probability rate and ranking number in main program for further crossover operation
    probability_rate, ranking = ranking_calculation(z_array)
    #Generating childhood chromosomes using defined crossover function
    x_array, y_array = generate_child_chromosomes(x_array,y_array,ranking)
    #Mutating generated chromosomes after crossover operation
    x_array, y_array = mutation_function(x_array, y_array, mutation_rate=0.2)
    #Calculating output after selection and reproduction process
    z_array = func(x_array, y_array)
    #print(z_array)
    min = np.min(z_array)
    print("Minimum value of function is: ", min)


