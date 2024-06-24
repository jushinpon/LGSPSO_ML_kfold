'''
Machine learning + PSO using concurrent loop for parallel process
By Prof. Shin-Pon Ju at NSYSU on 5/12/2024
MLP:
https://blog.csdn.net/weixin_38278334/article/details/83023958

adam hyperparameters:

https://blog.csdn.net/nyist_yangguang/article/details/121603917

https://blog.csdn.net/weixin_44491423/article/details/116711606

https://blog.csdn.net/weixin_38278334/article/details/83023958


alpha 0~0.1
beta1 0~0.9
beta2 0.950 ~0.999999
epsilon 10^-4 ~ 10^-9

All hyperparameters can be found in:
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#
'''
# PSO
# https://medium.com/@KunduSourodip/using-particle-swarm-optimization-optimizer-for-your-ml-model-e45a98dab601
# LGSCPSOA
# Mathematical Problems in Engineering,Volume 2014, Article ID 905712, 11 pages,http://dx.doi.org/10.1155/2014/905712

# Concurrent
# https://www.squash.io/how-to-parallelize-a-simple-python-loop/

import concurrent.futures
import numpy as np
import pandas as pd
import time
import os
import copy
import sys
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random


start = time.perf_counter()

file = open('MLP_report.txt', 'w')
data = pd.read_csv('merged_data.csv')
heatmap_filename = 'heatmap.png'
model_filename = 'bcc_classifier_model.pkl'  # 訓練好的模型檔名

max_fun = 15000 #only for lbfgs
max_iter4ML = 5000 #epoch step number
n_iter_no_change=20,  # Increased from a lower value,10
tol=4e-5,  # Consider adjusting,1e-4
validation_fraction=0.2  # Increased from a lower value,0.1
verbose = False

# 打亂資料集
current_time = int(time.time())
data = data.sample(frac=1, random_state=current_time).reset_index(drop=True)

# 分割特徵和目標變量
data_x = data.drop(['Composition', 'ifBCC', 'BCC%'], axis=1)

#corr = data_x.corr()
#plt.figure(figsize=(12, 10))
#sns.heatmap(corr, annot=True, cmap='coolwarm')
#plt.savefig(heatmap_filename)

#Normalize the data (good for relu)
#scaler_x = MinMaxScaler()
#sX = scaler_x.fit_transform(data_x)

# Standardize the data (good for relu and logistic)
scaler_x = StandardScaler()
sX = scaler_x.fit_transform(data_x)

X = pd.DataFrame(sX, columns=data_x.columns)
y = data['ifBCC']

#X = np.array(X)
#y = np.array(y)
#print(np.bincount(y))
#sys.exit()
#----------------------------------以上不動
#---------------------------------------參考
# 定義文字類型的超參數選項
#activation_options = ['logistic','relu']#,'identity', 'tanh'
#solver_options = ['lbfgs', 'sgd', 'adam']
#learning_rate_options = ['constant', 'invscaling', 'adaptive']

activation_options = ['relu']#choose one only
solver_options = ['adam']
learning_rate_options = ['adaptive']


random_activation = random.choice(activation_options)
random_solver = random.choice(solver_options)
random_learning_rate = random.choice(learning_rate_options)
random_nesterovs_momentum = random.choice([True]) #for sgd only
random_shuffle = random.choice([True])
random_early_stopping = random.choice([False])
#random_warm_start = random.choice([True, False])
random_verbose = random.choice([False])

def objective_function(params):
    (num_neuron1, num_neuron2, num_neuron3, batch_size, alpha, beta_1, beta_2, epsilon) = params
    #print (num_neuron1, num_neuron2, num_neuron3, alpha, beta_1, beta_2, epsilon)
    #sys.exit()
    accuracy_scores = []
    mlp = MLPClassifier(
            hidden_layer_sizes= (int(num_neuron1), int(num_neuron2), int(num_neuron3)),
            batch_size= int(batch_size),
            alpha=alpha,
            max_iter=max_iter4ML,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=20,
            activation=random_activation,
            solver=random_solver,
            learning_rate=random_learning_rate,
            early_stopping=random_early_stopping,
            tol=1e-4,
            validation_fraction=0.1,
            verbose = random_verbose
        )
    # Initialize the MLPClassifier with parameters
    #mlp = MLPClassifier(
    #    hidden_layer_sizes=(int(num_neuron1), int(num_neuron2), int(num_neuron3)),#hidden_layer_sizes=(100, 50, 25),
    #    batch_size= int(batch_size),#batch_size=32,
    #    alpha=0.0001,
    #    max_iter=5000,
    #    beta_1=0.9,
    #    beta_2=0.999,
    #    epsilon=1e-8,
    #    n_iter_no_change=20,
    #    activation='relu',
    #    solver='adam',
    #    learning_rate='adaptive',
    #    early_stopping=False,
    #    tol=1e-4,
    #    validation_fraction=0.2,
    #    verbose=False
    #)

    kf = KFold(n_splits=2, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Convert to numpy arrays if they are not already
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        #print(type(X_train),type(y_train))
        #print(type(X_test),type(y_pred))
        #sys.exit()
        accuracy = accuracy_score(y_test, y_pred)
        #print (f'accuracy score: {accuracy}\n')
        #print (f'y_test: {y_test[0:30]}\n')
        #print (y_test[0:30])
        #print (f'y_pred: {y_pred[0:30]}\n')
        #print (y_pred[0:30])
        accuracy_scores.append(accuracy)

    return -np.mean(accuracy_scores)  # 取負值以最小化

# 定義哪些超參數需要四捨五入為整數，1 表示需要四捨五入，0 表示不需要
#round_mask = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1])
#num_hidden_layers, num_neuron, alpha, batch_size, learning_rate_init, power_t, max_iter,
#tol, momentum, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun
# 定義PSO參數的上下界
#num_neuron1, num_neuron2, num_neuron3, batch_size, alpha, beta1, beta2, epsilon 
round_mask = np.array([  1,   1,   1,   1,   0,   0,      0,     0])
lower_bound =         [  5,   5,   5,   4,   0,   0,  0.950,  1e-9]
upper_bound =         [100, 100, 100, 200, 0.1, 0.9,  0.999,  1e-5]
file.write(f"***Hyperparameters for PSO:\nnum_neuron1, num_neuron2, num_neuron3, batch_size, alpha, beta1, beta2, epsilon\n")
file.write(f"lower_bound: {lower_bound}\n")
file.write(f"upper_bound: {upper_bound}\n\n")
file.flush()

# 檢查lower_bound、upper_bound、round_mask的長度是否相等
if not (len(lower_bound) == len(upper_bound) == len(round_mask)):
    raise ValueError(f"Length mismatch: lower_bound ({len(lower_bound)}), upper_bound ({len(upper_bound)}), round_mask ({len(round_mask)})")

# 定義PSO算法參數
n_dimensions = len(lower_bound)
#n_particles = 8
n_particles = os.cpu_count()
max_iter = 1000 #pso step
w = 0.9  # to 0.2 at end
c1 = 1.49445
c2 = 1.49445
alphapso = 0.6
alphapso1 = 1 - alphapso 

# SLR for LGSPSO
def get_neighborhood(k, total_particles):
    k_minus_2 = (k - 2 - 1) % total_particles
    k_minus_4 = (k - 4 - 1) % total_particles
    k_plus_1 = (k + 1 - 1) % total_particles
    k_plus_3 = (k + 3 - 1) % total_particles
    return [k_plus_1, k_minus_2, k_plus_3, k_minus_4]

# 定義PSO函數
def pso(objective_function, lower_bound, upper_bound, n_particles, n_dimensions, max_iter, w, c1, c2):
    SLR = []
    for k in range(1, n_particles + 1):  # 需要從1開始
        SLR.append(get_neighborhood(k, n_particles))
    SLR = np.array(SLR)

    # 隨機初始化粒子在搜索空間中的位置
    particles = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_particles, n_dimensions))
     # 將標記為1的超參數四捨五入為整數
    particles[:, round_mask == 1] = np.round(particles[:, round_mask == 1]).astype(int) ## seems not work
    #print (particles)
    #print(int(particles[:, round_mask == 1]))  # Print only the converted parts
    #print([int(particle) for particle in particles[0, round_mask == 1]])  # Check types of the first row’s integer-converted parts
    #for i in range(particles.shape[1]):
    #    if round_mask[i] == 1:
    #        particles[:, i] = np.round(particles[:, i]).astype(int)
#
    #print(particles)

    #sys.exit()
    # 強制約束邊界
    particles = np.clip(particles, lower_bound, upper_bound)
    # 初始化個體最優位置和全局最優位置
    personal_best_positions = copy.deepcopy(particles)
   
    # 使用多進程計算初始目標函數值
    #results = []
    #for i in range(len(particles)):
    #    print (f'{i} {particles[i]}')
    #    print (f'result: {objective_function(particles[i])}\n')
    #sys.exit()    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(objective_function, particles))
    results = np.array(results)
    #sys.exit()    
    particles_bestVal = copy.deepcopy(results)
    global_best_position = np.copy(particles[np.argmin(results)])
    global_bestVal = results[np.argmin(results)]

    file.write(f"Initial particle best value: {global_bestVal}\n")
    file.write(f"Initial best hyperparameters: {global_best_position}\n")
    file.flush()
    
    print(f"Initial particle best value: {global_bestVal}\n")
    print(f"Initial best hyperparameters: {global_best_position}\n")
    #file.write(f"upper_bound: {upper_bound}\n\n")

    # 初始化局部數據
    local_best_positions = np.copy(particles)
    local_bestVal = results[:]

    for i in range(n_particles):
        local_best_index = SLR[i][np.argmin(results[SLR[i]])]
        local_best_positions[i] = np.copy(particles[local_best_index])
        local_bestVal[i] = results[local_best_index]

    # 初始化速度
    #velocities = np.zeros((n_particles, n_dimensions))
    velocities = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_particles, n_dimensions))
    
    #print (f"Initial for paticles:\n")
    #print (particles)
    #print (f"Initial for velocities:\n")
    #print (velocities)        
    #print (f"Initial for results:\n")
    #print (results)
    #print (f"PSO parameters,w,c1,c2:\n")
    #print (w,c1,c2)
    # 進行優化
    inc = ((w - 0.2) / (max_iter - 1))  # w從w到0.2，遞減
    for i in range(max_iter):
        # 更新速度
        r1 = np.random.rand(n_particles, n_dimensions)
        r2 = np.random.rand(n_particles, n_dimensions)
        velocities = (w - i * inc) * velocities + \
                     c1 * r1 * (alphapso * (personal_best_positions - particles) + alphapso1 * (local_best_positions - particles)) + \
                     c2 * r2 * (global_best_position - particles)

        # 更新位置
        particles = particles + velocities

        # 強制約束邊界
        particles = np.clip(particles, lower_bound, upper_bound)

        # 將標記為1的超參數四捨五入為整數
        particles[:, round_mask == 1] = np.round(particles[:, round_mask == 1])

        # 使用多進程計算目標函數值
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(objective_function, particles))
        results = np.array(results)
        #print (f"it {i} for paticles:\n")
        #print (particles)
        #print (f"it {i} for velocities:\n")
        #print (velocities)        
        #print (f"it {i} for results:\n")
        #print (results)
        #particles_bestVal = results
        iter_best_position = np.copy(particles[np.argmin(results)])
        iter_bestVal = results[np.argmin(results)]

        file.write(f"\nThe best value at PSO iteration {i}: {iter_bestVal}, particle {np.argmin(results)}\n")
        file.write(f"hyperparameters: {iter_best_position}\n")
        file.flush()
        print (f"\n**The best value at PSO iteration {i}: {iter_bestVal}, particle {np.argmin(results)}:\n")
        print (f"hyperparameters: {iter_best_position}\n")
        # 更新個體最優位置和全局最優位置
        for j in range(n_particles):
            local_best_index = SLR[j][np.argmin(results[SLR[j]])]
            # 更新局部信息
            if results[local_best_index] < local_bestVal[j]:
                local_best_positions[j] = np.copy(particles[local_best_index])
                local_bestVal[j] = results[local_best_index]

            if results[j] < particles_bestVal[j]:
                personal_best_positions[j] = np.copy(particles[j])
                particles_bestVal[j] = results[j]

            if particles_bestVal[j] < global_bestVal:
                global_best_position = np.copy(personal_best_positions[j])
                global_bestVal = particles_bestVal[j]
                file.write(f"\n***Finding the new global best at PSO iteration {i} particle {j} for value {global_bestVal}\n")
                file.write(f"hyperparameters: {global_best_position}\n")
                file.flush()
                print(f"\n\n***Finding the new global best at PSO iteration {i} particle {j} for the current best global value: {global_bestVal}\n")
                print(f"hyperparameters: {global_best_position}\n\n")

    return global_best_position

# 調用PSO函數
hyperparameters = pso(objective_function, lower_bound, upper_bound, n_particles, n_dimensions, max_iter, w, c1, c2)

# 使用最佳超參數訓練MLP模型
(num_neuron1, num_neuron2,num_neuron3, batch_size, alpha, beta_1, beta_2, epsilon) = hyperparameters

# 隨機選擇文字類型的超參數
#random_activation = random.choice(activation_options)
#random_solver = random.choice(solver_options)
#random_learning_rate = random.choice(learning_rate_options)
#random_nesterovs_momentum = random.choice([True, False])
#random_shuffle = random.choice([True, False])
#random_early_stopping = random.choice([True, False])

#hidden_layer_sizes = [int(num_neuron)] * int(num_hidden_layers)

mlp = MLPClassifier(
    hidden_layer_sizes = (int(num_neuron1), int(num_neuron2), int(num_neuron3)),
    alpha=alpha,
    batch_size= int(batch_size),
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
    max_iter=max_iter4ML,
    activation=random_activation,
    solver=random_solver,
    learning_rate=random_learning_rate,
    n_iter_no_change=20,
    nesterovs_momentum=random_nesterovs_momentum,
    shuffle=random_shuffle,
    early_stopping=random_early_stopping,
    tol=1e-4,
    validation_fraction=0.1,
    verbose = random_verbose
)

mlp.fit(X, y)

# 使用交叉驗證評估模型性能
accuracy_scores = []
kf = KFold(n_splits=5, shuffle=True, random_state=current_time)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    joblib.dump(mlp, model_filename)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

mean_accuracy = np.mean(accuracy_scores)
file.write("\n*** Final training with the most optimal hyperparameters\n")
file.write(f"Accuracy (Cross-Validation): {mean_accuracy}\n")
file.write(f"Final hyperparameters: {hyperparameters}\n")
finish = time.perf_counter()
file.write(f'\n**Finished in {round(finish-start, 2)} second(s)\n')
file.write(f'Model saved as {model_filename}\n')
file.close()
