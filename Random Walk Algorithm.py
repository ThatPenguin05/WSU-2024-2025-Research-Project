
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math as math
from matplotlib.colors import LinearSegmentedColormap
import sys
import time

#Assumptions
#X1, X2, ... Xn axes have the same length for an n-cube region in R^n space
#The number of subdivisions along each axes will be the same for the n-cube region
#The probability field in R^n will be at each point an average of randomized discrete normal distributions along each axis
start_time = time.time()

#Initialization Variables (Global)
N = 10 #int >= 2 : number of samples, must be >= 2 or else divide by 0 error
T = 150 #int >= 1 : number of iterations
S = 1 #int >= 0 : step size
simulations = 1 #int >= 1 : number of simulations to be performed at once with the same initial conditions and distributions
dimensions = 2 #int >= 1 : number of dimensions n for R^n space that numpy can handle
axis_length = 1 #float >= 1 : defined region length along each axis of n-cube
axis_subdivide = 100 #int >= 1 : lattice resolution along each axis of n-cube
cluster_subdivide = 5 #int >= 1 : number of subdivisions of axis_subdivide, c ** dim number of clusters will be generated
normal_mean_range = [0, 1] #floats [a,b] such that 0<=a<b<=1: range for normal curve randomized mean values per axis
normal_deviation_range = [0.1, 1] #floats [a,b] such that 0<=a<b<=1: range for normal curve randomized standard deviation values per axis

enable_cluster = True #bool : enable or disable the cluster sampling measurement of strategy accuracy
show_paths = True #bool : enable or disable the visualization of paths for generated random walks
show_environment_data = False #bool : enable or disable showing the environment each simulation was run in
show_simulation_data = False #bool : enable or disable showing the best sample locations and p values
show_simulation_progress = False #bool : enable or disable showing how many simulations have completed
show_normal_curves = False #bool : enable or disable showing the normal curve information

sample_generation_type = 1 #int >= 0 : sample generation type
sample_generation_dictionary = {
    0 : 'Random Sampling', #random generation of samples
    1 : 'Latin Hypercube Sampling' #uniform distribution of random generation by Latin Hypercube Sampling method
}

#---------------------------------------------------------------------------------------#

#Color Map Generation (Global)
if dimensions == 2 and simulations == 1:
    #R^n Probability Distribution Color Map (Global)
    def p_field_cmap(subdivide):
        color = np.linspace(0, 1, subdivide) #low density = white, high density = red
        rgba = np.array([[0, 0, c, 1.0] for c in color])
        return rgba
    #Sample Color Map (Global)
    def sample_cmap(iteration, n):
        color = np.linspace(0.3, 1, iteration)
        rgba = np.tile([[c, c*0.15, c*0.1, 1.0] for c in color], n).reshape(iteration*n, 4)
        return rgba
    #Best Sample Color Map (Global)
    def best_cmap(n):
        rgba = np.tile([1, 1, 1, 1], n).reshape(n, 4)
        return rgba
    p_cmap = LinearSegmentedColormap.from_list("", p_field_cmap(axis_subdivide))

#R^n Probability Distribution Coordinate Grid and Values (Global)
def generate_p_values(subdivide, length, dim, mean_range, deviation_range):
    #randomized normal curve per axis, average all p values per axis to get p value per coordinate
    #dim number of random mean values, repeated subdivide times
    mean = np.repeat(np.random.default_rng().uniform(low=mean_range[0], high=mean_range[1], size=dim), repeats=subdivide).reshape(dim, subdivide)
    #dim number of random standard deviation values, repeated subdivide times
    standard_deviation = np.repeat(np.random.default_rng().uniform(low=deviation_range[0], high=deviation_range[1], size=dim), repeats=subdivide).reshape(dim, subdivide)
    #generate a random normal curve value for every subdivide mean and standard deviation, repeated across each dimension
    d_axis_p = np.absolute(np.random.normal(loc=mean.ravel(),scale=standard_deviation.ravel(),size=dim*subdivide)).reshape(dim, subdivide)
    #sort the curves according to where the mean falls
    for d in np.arange(dimensions):
        #scale the mean point by the number of subdivisions
        mean_point = np.int64(np.round(subdivide*mean[d, 0]))
        #everything up to the mean location, ascending
        d_axis_p[d, :mean_point] = np.sort(d_axis_p[d, :mean_point])
        #everything past the mean location, descending
        d_axis_p[d, mean_point:] = np.sort(d_axis_p[d, mean_point:])[::-1]
    
    probability_field_axis = length*d_axis_p
    
    return probability_field_axis, mean, standard_deviation
#generate lattice grid for probability distribution by generating each axis
lattice_coordinate = np.arange(axis_subdivide)/axis_subdivide
p_value, p_mean, p_standard_deviation = generate_p_values(subdivide=axis_subdivide, length=axis_length, dim=dimensions,
                                                          mean_range=normal_mean_range, deviation_range=normal_deviation_range)

#R^n Sample Generation (Global)
def sample_generation(method, dim, subdivide, n, coordinate):
    if method == 'Random Sampling':
        #generate random index values and array of corresponding coordinates
        d_axis_index = np.array([np.random.choice(np.arange(subdivide), n, replace=False) for d in np.arange(dim)]).transpose() #indicies of random starting samples
        #generate random samples along each axis based on randomized index values
        d_axis_coordinate = coordinate[d_axis_index]
    if method == 'Latin Hypercube Sampling':
        #fail if more than axis_subdivide samples are provided
        if n > subdivide: sys.exit("Latin Hyper Cube must have at most " + str(subdivide) + " samples. " + str(n) + " provided.")
        #generate axis_divide approximately evenly spaced integer interval bounds across each axis
        axis_index_interval = np.round(np.linspace(0, subdivide, n+1))
        #generate randomized array of sample index values along each axis for the final initial lhs sample placement
        d_axis_index = np.array([np.random.choice(np.random.randint(axis_index_interval[:-1], axis_index_interval[1:]), size=n, replace=False) for d in np.arange(dim)]).transpose()
        #generate random coordinates along each axis based on randomized index values
        d_axis_coordinate = coordinate[d_axis_index]
    return d_axis_index, d_axis_coordinate

#---------------------------------------------------------------------------------------#

#Strategy Class (Local)
class LatticeRandomWalk:
    '''
    Generate n samples on probability coordinate grid (lattice in n-cube in R^n).
    Run for t iterations with s step size.
    During each iteration, each sample has a random probability assigned
    to the lattice direction it will move s distance in (hence, strict).
    
    When t iterations have concluded, each sample will have generated a
    point history. The entire history is n*t number of points.
    Each point in the history will collect various bits of data.
    Root Mean Square Displacement: s*sqrt(t).
    
    The history of a point will be indicated by a particular colormap, with white
    being the first entry in history, and blue being the end point in history.
    All such points will be place on the scatter plot.
    '''
    def __init__(self, name, n, iteration, step, p, subdivide, dim, lattice_axis, axis_sample_index, axis_sample_coordinate):
        #initialize the variables, arrays, and dictionary (local)
        self.name = name #name of this strategy class
        self.n = n #number of samples
        self.iteration = iteration #number of iterations
        self.step = step #step size
        self.p = p #p values generated
        self.subdivide = subdivide #axis subdivisions
        self.dim = dim #number of dimensions
        self.lattice_axis = lattice_axis #all lattice axes
        self.axis_sample_index = axis_sample_index #list of index values for samples per axis upon initialization
        self.axis_sample_coordinate = axis_sample_coordinate #list of coordinates for samples per axis upon initialization
        
        #initialize the empty n*t history arrays
        self.sample_history = np.zeros([self.n, self.iteration, self.dim]) #n rows, t columns, dim dimensions
        self.sample_history[:, 0, :] = self.axis_sample_coordinate #add the initial coordinates as the first columns of sample history
        self.p_history = np.zeros([self.n, self.iteration]) #n rows, t columns
        self.p_history[:, 0] = np.sum(self.p[np.arange(self.dim), self.axis_sample_index[:, np.arange(self.dim)]], axis=1)/self.dim #add the initial p values as the first columns of p history
        self.direction_probability = np.tile(1/(2*self.dim),2*self.dim) #sample direction probabilities 1 / (2*dim)
        
        #initialize the direction double matrix
        self.direction = np.zeros((2*self.dim,self.dim)).ravel() #direction matrix made of identity and negative identity matricies
        self.direction[:self.dim*self.dim:self.dim+1] = 1 #identity matrix for the first dim*dim space
        self.direction[self.dim*self.dim::self.dim+1] = -1 #negative indentity matrix for the second dim*dim space
        self.direction = self.direction.reshape([2*self.dim, self.dim]) #reshaping to two back to back identity and negative identity matricies
    
    def generate_walk(self): #(local)
        #the lattice random walk for t-1 iterations with s step size (the first iteration was initialized at the start)
        #get one random direction for each sample, based on direction probabilities per sample
        random_direction = np.array(np.random.choice(np.arange(2*self.dim), size=(self.iteration-1)*self.n, p=self.direction_probability)).reshape(self.n, self.iteration-1)
        #the randomly selected directions now make up the direction history
        direction_history = self.direction[random_direction]
        
        #the index history is the initial points plus each step of random directions taken, all combined in one list
        index_history = np.transpose(np.array([np.absolute(self.axis_sample_index+np.sum(direction_history[:, :t], axis=1)) for t in np.arange(self.iteration)], dtype=np.int64), [1, 0, 2])
        index_history[index_history > self.subdivide-1] = self.subdivide-1 #correction for index values that go over bounds
        #the p values found per sample history
        self.p_history[:, 1:] = np.sum([self.p[d, index_history[:, 1:, d]] for d in np.arange(self.dim)], axis=0)/self.dim
        #the sample coordinate found per sample history
        self.sample_history[:, 1:, :] = np.transpose(np.array([self.lattice_axis[index_history[:, 1:, d]] for d in np.arange(self.dim)]), [1, 2, 0])
        
        #the index of the best p value per sample history
        p_best_index = np.argmax(self.p_history, axis=1)
        #the best p value per sample history
        best_history_p = np.max(self.p_history, axis=1)
        #the best sample index per sample history
        best_history_index = np.array([index_history[i, p_best_index[i], :] for i in np.arange(self.n)])
        #the best sample coordinate per sample history
        best_history_coordinate = np.array([self.sample_history[i, p_best_index[i], :] for i in np.arange(self.n)])
        
        return self.sample_history, best_history_coordinate, best_history_index, best_history_p

#Simulation Initialization with Strategy (Global)
if simulations:
    #initial arrays for gathering information across multiple simulations
    sim_sample_history = np.array([]) #sample history per simulation
    sim_best_coordinate = np.array([]) #best sample coordinate per simulation
    sim_best_index = np.array([]) #best sample index per simulation
    sim_best_p_value = np.array([]) #best sample p value per simulation
    for sim in np.arange(simulations):
        #conduct each simulation
        axis_initial_index, axis_initial_coordinate = sample_generation(method=sample_generation_dictionary[sample_generation_type],
                                                              dim=dimensions, subdivide=axis_subdivide, n=N, coordinate=lattice_coordinate)
        strategy = LatticeRandomWalk(name='Lattice Random Walk', n=N, iteration=T, step=S, p=p_value,
                                     subdivide=axis_subdivide, dim=dimensions, lattice_axis=lattice_coordinate,
                                     axis_sample_index=axis_initial_index, axis_sample_coordinate=axis_initial_coordinate)
        sample_history, best_coordinate, best_index, best_p_value = strategy.generate_walk()
        #append current simulation information to the total information gathered
        sim_sample_history = np.append(sim_sample_history, sample_history)
        sim_best_coordinate = np.append(sim_best_coordinate, best_coordinate)
        sim_best_index = np.append(sim_best_index, best_index)
        sim_best_p_value = np.append(sim_best_p_value, best_p_value)
        if show_simulation_progress: print("Simulation " + str(sim+1) + " Done")
    #reshape the simulation lists accordingly
    sim_sample_history = sim_sample_history.reshape([simulations, N, T, dimensions])
    sim_best_coordinate = sim_best_coordinate.reshape([simulations, N, dimensions])
    sim_best_index = sim_best_index.reshape([simulations, N, dimensions])
    sim_best_p_value = sim_best_p_value.reshape([simulations, N])

#---------------------------------------------------------------------------------------#

#Strategy Accuracy of Random Walk Vs. Expected Non-Uniform Sample from Probability Distribution (Global)
strategy_error = np.zeros([simulations])
strategy_accuracy = np.zeros([simulations])

if enable_cluster:
    #ensure the number of clusters requested is valid
    if cluster_subdivide > axis_subdivide: sys.exit("Cluster must have at most " + str(axis_subdivide) + " subdivisions. " + str(cluster_subdivide) + " provided.")
    #total number of clusters
    total_cluster = cluster_subdivide**dimensions
    #cluster coordinate axis components that, in a linear combination with each axis index value, make up the current cluster index
    cluster_axis_components = cluster_subdivide**np.arange(dimensions+1)
    #best sample coordinate axis components that, in a linear combination with each axis index value, make up the current best sample index
    best_axis_components = axis_subdivide**np.arange(dimensions+1)
    
    #generate cluster_subdivide approximately evenly spaced integer interval bounds across each axis
    cluster_axis_index_interval = np.round(np.linspace(0, axis_subdivide, cluster_subdivide+1))
    #generate values along each axis for which all clusters are defined
    d_cluster_axis_index = np.array([np.random.randint(cluster_axis_index_interval[:-1], cluster_axis_index_interval[1:]) for d in np.arange(dimensions)])
    #convert the index values into the coordinate counter parts
    d_cluster_axis_coordinate = lattice_coordinate[d_cluster_axis_index]
    #generate all cluster coordinate, index values
    cluster_coordinate_index = [x for x in np.ndenumerate(np.arange(total_cluster).reshape(np.repeat([cluster_subdivide], repeats=dimensions)))]
    cluster_coordinate = np.array([cluster_coordinate_index[d][0] for d in np.arange(total_cluster)])
    #convert the index values into the p value of each cluster
    cluster_p_value = np.sum([p_value[d, cluster_coordinate[:, d]] for d in np.arange(dimensions)], axis=0)/dimensions
    
    for sim in np.arange(simulations):
        observed_sample_best = np.zeros([total_cluster]) #list of number of best samples found per cluster
        #for each sample, gather the number of local best samples found per cluster
        for sample in sim_best_index[sim]:
            for cluster in np.arange(total_cluster):
                #check that the current best sample index is within the corresponding cluster bounds across each axis
                if np.all(sample >= cluster_axis_index_interval[cluster_coordinate[cluster]]) and np.all(sample <= cluster_axis_index_interval[cluster_coordinate[cluster]+1]):
                    observed_sample_best[cluster] += 1
        
        #get the index values, sample counts, and p values of clusters which detected samples
        cluster_of_detected = np.array([x for x in np.arange(total_cluster) if observed_sample_best[x] != 0])
        observed_best_of_detected = observed_sample_best[cluster_of_detected]
        p_of_detected = cluster_p_value[cluster_of_detected]
        #sum of all detected cluster probability values
        p_cluster_sum_of_detected = np.sum(p_of_detected)
        #normalized version of detected cluster p values
        normalized_p_cluster_of_detected = p_of_detected/p_cluster_sum_of_detected
        
        #once all local information is gathered and observed, compute the best distribution error and the strategy accuracy
        duplicate_N = 0 #samples are often double counted along multiple cluster boundaries
        if np.sum(observed_best_of_detected) > N: duplicate_N = np.sum(observed_best_of_detected) - N
        error = np.sum(np.absolute(observed_best_of_detected - (duplicate_N+N)*normalized_p_cluster_of_detected))
        #calculate the maximum error as the worst case scenario when all total_N_counted points are in the highest probability value cluster
        maximum_error = 2*(duplicate_N+N)*(1-np.max(cluster_p_value)/np.sum(cluster_p_value))
        strategy_error[sim] = error #error between the observed and expected number of best samples across all clusters
        strategy_accuracy[sim] = 1 - strategy_error[sim]/maximum_error #percentage of how close error is to 0, with 100% being error = 0

#---------------------------------------------------------------------------------------#

#Visualization of Samples and Probability Distribution (Global)
#environment information
if show_environment_data:
    axis_normal_curve_information = ['Dimension ' + str(d+1) + ': Mean=' + f"{p_mean[d]:.2f}" + ' | Standard Deviation=' + f"{p_standard_deviation[d]:.2f}" for d in np.arange(dimensions)]
    print('\n-------------------------------------------------------------+'
        + '\nNumber of Dimensions: ' + str(dimensions)
        + '\nProbability Field Resolution: ' + str(axis_subdivide) + '^' + str(dimensions)
        + '\nSample Region Dimensions: ' + str(axis_length) + '^' + str(dimensions)
        + '\nSample Region N-Volume: ' + str(axis_length**dimensions)
        + '\nN-Cube Cluster Dimensions: ' + str(cluster_subdivide) + '^' + str(dimensions)
        + '\nNumber of Clusters: ' + str(cluster_subdivide**dimensions)
        + '\n\nNormal Curve Information per Axis'
    )
    if show_normal_curves:
        for d in np.arange(dimensions): print(f'{axis_normal_curve_information[d]}')
    print('\nNumber of Simulations: ' + str(simulations)
        + '\nPoint Generation Method: ' + sample_generation_dictionary[sample_generation_type]
        + '\n' + strategy.name + ': N=' + str(N) + ', T=' + str(T) + ', S=' + str(S)
    )
#simulation information
if show_simulation_data:
    for sim in np.arange(simulations):
        print('\n-------------------------------------------------------------+'
            + '\nSimulation ' + str(sim+1)
            + '\nStrategy Error: ' + f"{strategy_error[sim]:.2f}"
            + '\nStrategy Accuracy: ' + f"{strategy_accuracy[sim]*100:.2f}" + "%"
        )
        sample_coordinates = np.transpose([(sim_best_index[sim,:]%best_axis_components[d+1]-sim_best_index[sim,:]%best_axis_components[d])/(best_axis_components[d]) for d in np.arange(dimensions)])
        sample_data = [str(sample_coordinates[i]) + ' | ' + f'{sim_best_p_value[sim,i]:.2f}' for i in np.arange(N)]
        print('\nCoordinate | Probability Found')
        for i in np.arange(N): print(sample_data[i])
print('\n-------------------------------------------------------------+'
      + '\nRun Time: ' + f'{time.time()-start_time:.6f}' + ' Seconds'
      + '\nAverage Strategy Accuracy: ' + f'{100*np.sum(strategy_accuracy)/simulations:.2f}%'
      + '\nStrategy Accuracy per Simulation: ' 
      + '\n' + ' '.join([f'{x*100:.2f}%' for x in strategy_accuracy]))
print('\n-------------------------- Complete -------------------------+\n')
#visualization available for 2 dimensions and 1 simulation
if dimensions == 2 and simulations == 1:
    visualize_p_coords_x1 = np.repeat(lattice_coordinate, repeats=axis_subdivide).reshape(axis_subdivide, axis_subdivide).T #x1 axis for 2d coordinate visualization
    visualize_p_coords_x2 = np.repeat(lattice_coordinate, repeats=axis_subdivide).reshape(axis_subdivide, axis_subdivide) #x2 axis for 2d visualization
    visualize_p_values_x1 = np.repeat(p_value[0], repeats=axis_subdivide).reshape(axis_subdivide, axis_subdivide).T #x1 axis for 2d p value visualization
    visualize_p_values_x2 = np.repeat(p_value[1], repeats=axis_subdivide).reshape(axis_subdivide, axis_subdivide) #x2 axis for 2d p value visualization
    visualize_p_values = np.sum([visualize_p_values_x1, visualize_p_values_x2], axis=0)/dimensions
    fig, ax = plt.subplots()
    if show_paths: ax.scatter(sim_sample_history[0,:,:,0], sim_sample_history[0,:,:,1], s=3, c=sample_cmap(T,N), marker='s', zorder=1) #sample distribution
    ax.scatter(sim_best_coordinate[0,:,0], sim_best_coordinate[0,:,1], s=3, c=best_cmap(N), marker='s', zorder=2) #best sample distribution
    plt.scatter(visualize_p_coords_x1, visualize_p_coords_x2, s=20, c=visualize_p_values, cmap=p_cmap, marker='s', zorder=0) #p distribution
    plt.title(strategy.name + ': N=' + str(N) + ', D=' + str(dimensions) + ', C=' + str(cluster_subdivide) + ', T=' + str(T)
            + '\nStrategy Error: ' + f"{strategy_error[0]:.2f}" + ' | Strategy Accuracy: ' + f"{strategy_accuracy[0]*100:.2f}" + "%", fontsize=12)
    plt.colorbar()
    plt.show()




'''
--- index "boundary" bias for direction changes ---

self.direction['N'] = 0 #add the no direction option to the direction list

#for i in np.arange(self.n):
    #adjust direction if direction change would yield index error
    if (self.sample_index[i]+self.history_direction[i,t+1]<=0) or (self.sample_index[i]+self.history_direction[i,t+1]>=(self.subdivide**2-1)):
        self.history_direction[i,t+1] = 0

--- bias code removed ---

bias_type = 0 #int >= 0 : bias type
bias_dictionary = { #types of bias to choose from
    0 : None,
    1 : 'Average Increase' #adjustment of direction probabilities based on p value increase history, per sample
}

bias, bias_interval

bias_interval = 10 #int >= 0 : apply the average increase bias to the previous bias_interval points, disregard all history prior to that

self.bias = bias #bias type for sample direction
self.bias_interval = bias_interval #look at the previous bias_interval number of directions for average increase bias

if self.bias: self.sample_direction = np.ones([self.n, self.iteration, 2*self.dim]) * np.float64(1/(2*self.dim)) #direction if bias is present

#initialize variables for bias types
if self.bias == 'Average Increase':
    self.best_direction_tally = np.ones([self.n, self.iteration, 2*self.dim]) #2*dim direction tally

#adjust direction according to bias type
    if self.bias == 'Average Increase': #average increase bias
        #add the previous step direction probability row (subtract 1) to the current
        self.best_direction_tally[:, t+1] += self.best_direction_tally[:, t] - 1
        #array of truth values: true if the current p value is greater than the previous, false otherwise
        truth_list = np.array([1 if (self.history_p_index[i, t+1] > self.history_p_index[i, t]) else 0 for i in np.arange(self.n)])
        #compare truth value to corresponding direction
        for i in np.arange(self.n):
            if truth_list[i]:
                #if a corresponding direction led to an increase, add one to that column tally
                self.best_direction_tally[i, t+1, rand_directions[i]] += 1
            else:
                #otherwise, average 1 across all directions
                self.best_direction_tally[i, t+1, :] += (1/(2*self.dim))
        if t+1 < self.bias_interval+1:
            #adjust the step direction probability set to be the current tally per sample divided by 2*self.dim+t+1
            self.sample_direction[:, t+1, :] = self.best_direction_tally[:, t+1, :] / (2*self.dim+t+1)
        else:
            #subtract the t-bias_interval row to disregard all history prior to bias_interval history interval
            adjusted_direction_tally = self.best_direction_tally[:, t+1, :] - self.best_direction_tally[:, t-self.bias_interval, :]
            #add the 1 back in that was discluded
            adjusted_direction_tally += 1
            #adjust the step direction probability set to be the average of the last bias_interval samples divided by 2*dim+bias_interval+1
            self.sample_direction[:, t+1, :] = adjusted_direction_tally / (2*self.dim+self.bias_interval+1)

bias=bias_dictionary[bias_type], bias_interval=bias_interval

+ '\nBias Type: ' + str(bias_dictionary[bias_type])

+ ' | Bias Type: ' + str(bias_dictionary[bias_type])
'''
