'''
Machine learning + PSO using concurrent loop for parallel process
By Prof. Shin-Pon Ju at NSYSU on 5/12/2024

'''
#PSO
#https://medium.com/@KunduSourodip/using-particle-swarm-optimization-optimizer-for-your-ml-model-e45a98dab601
#LGSCPSOA
#Mathematical Problems in Engineering,Volume 2014, Article ID 905712, 11 pages,http://dx.doi.org/10.1155/2014/905712

#Concurrent
#https://www.squash.io/how-to-parallelize-a-simple-python-loop/

import concurrent.futures
import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

start = time.perf_counter()

file = open('SVR_report.txt', 'w')
data = pd.read_csv('BostonHousing.csv')

# 打亂資料集
current_time = int(time.time())
data = data.sample(frac=1, random_state=current_time).reset_index(drop=True)

# 分割特徵和目標變量
X = data.drop('medv', axis=1)
y = data['medv']

# Define the objective function with cross-validation
def objective_function(params):
    c, gamma, epsilon = params
    mse_scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=current_time)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        svr = SVR(C=c, gamma=gamma, epsilon=epsilon)
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores)

# SLR for LGSPSO
def get_neighborhood(k, total_particles):
    k_minus_2 = (k - 2 - 1) % total_particles
    k_minus_4 = (k - 4 - 1) % total_particles
    k_plus_1 = (k + 1 - 1) % total_particles  
    k_plus_3 = (k + 3 - 1) % total_particles 
    return [k_plus_1, k_minus_2, k_plus_3, k_minus_4]

# Define the bounds for each hyperparameter
lower_bound = [1e-8, 1e-8, 1e-8]
upper_bound = [50, 5, 5]

# Define the PSO algorithm
n_dimensions = 3
n_particles = os.cpu_count()  
max_iter = 1001  
w = 1.2  # to 0.4 at end
c1 = 1.49445
c2 = 1.49445
alpha = 0.6
alpha1 = 1 - alpha 

# Define the PSO function
def pso(objective_function, lower_bound, upper_bound, n_particles, n_dimensions, max_iter, w, c1, c2):
    
    SLR = []
    for k in range(1, n_particles + 1):  # need to start from 1
        SLR.append(get_neighborhood(k, n_particles))
    SLR = np.array(SLR)
    
    # Initialize the particles randomly within the search space
    particles = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_particles, n_dimensions))

    # Initialize the personal best positions and global best position
    personal_best_positions = particles

    # Use multiprocessing to calculate the initial objective function values
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(objective_function, particles))
    results = np.array(results)

    particles_bestVal = results 
    global_best_position = particles[np.argmin(results)]
    global_bestVal = results[np.argmin(results)]

    # Initialize local data
    local_best_positions = particles.copy()
    local_bestVal = results.copy()
    for i in range(n_particles):
        # Ensure indices are within bounds
        local_best_index = SLR[i][np.argmin(results[SLR[i]])]
        local_best_positions[i] = particles[local_best_index]
        local_bestVal[i] = results[local_best_index]

    # Initialize the velocities
    velocities = np.zeros((n_particles, n_dimensions))
   
    # Perform optimization
    inc = ((w - 0.2) / (max_iter - 1))  # w from w to 0.2, descending
    for i in range(max_iter):
        # Update the velocities
        r1 = np.random.rand(n_particles, n_dimensions)
        r2 = np.random.rand(n_particles, n_dimensions)
        velocities = (w - i * inc) * velocities + \
                     c1 * r1 * (alpha * (personal_best_positions - particles) + alpha1 * (local_best_positions - particles)) + \
                     c2 * r2 * (global_best_position - particles)

        # Update the positions
        particles = particles + velocities

        # Enforce the bounds
        particles = np.clip(particles, lower_bound, upper_bound)

        # Use multiprocessing to calculate the objective function values
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(objective_function, particles))
        results = np.array(results)

        # Update the personal best positions, local best, and global best positions
        for j in range(n_particles):
            local_best_index = SLR[j][np.argmin(results[SLR[j]])]
            # Update local info
            if results[local_best_index] < local_bestVal[j]:
                local_best_positions[j] = particles[local_best_index]
                local_bestVal[j] = results[local_best_index]

            if results[j] < particles_bestVal[j]:
                personal_best_positions[j] = particles[j]
                particles_bestVal[j] = results[j]

            if particles_bestVal[j] < global_bestVal:
                global_best_position = personal_best_positions[j]
                global_bestVal = particles_bestVal[j]
                file.write(f"***iter {i} particle {j} for value {global_bestVal}\n")
                file.write(f"hyperparameters for c, gamma, epsilon:{global_best_position}\n")

    return global_best_position

# Call the PSO function with the specified hyperparameters
hyperparameters = pso(objective_function, lower_bound, upper_bound, n_particles, n_dimensions, max_iter, w, c1, c2)

# Train the SVM model with the best hyperparameters
c, gamma, epsilon = hyperparameters
svr = SVR(C=c, gamma=gamma, epsilon=epsilon)
svr.fit(X, y)

# Evaluate the performance with cross-validation
mse_scores = []
kf = KFold(n_splits=5, shuffle=True, random_state=current_time)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

mean_mse = np.mean(mse_scores)
file.write("\n\n*** Final training with the most optimal hyperparameters\n")
file.write(f"Mean Squared Error (Cross-Validation):{mean_mse}\n")
file.write(f"Fine-tuning hyperparametes for c, gamma, epsilon:, {c}, {gamma}, {epsilon}\n")
finish = time.perf_counter()
file.write(f'\n**Finished in {round(finish-start, 2)} second(s)')
file.close()
