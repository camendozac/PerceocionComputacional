from collections import Counter
import math
def euclidean_distance(fruit1, fruit2):
    color_map = {'red': 0, 'yellow': 1}
    shape_map = {'round': 0, 'oval': 1}
    
    fruit1_encoded = [color_map[fruit1[0]], shape_map[fruit1[1]]]
    fruit2_encoded = [color_map[fruit2[0]], shape_map[fruit2[1]]]
    
    return math.sqrt(sum((f1 - f2) ** 2 for f1, f2 in zip(fruit1_encoded, fruit2_encoded)))
def find_nearest_neighbors(test_fruit, training_data, k=5):
    distances = []
    
    for fruit in training_data:
        distance = euclidean_distance(test_fruit, fruit[:-1])
        distances.append((distance, fruit))
    
    distances.sort(key=lambda x: x[0])
    nearest_neighbors = [neighbor[1] for neighbor in distances[:k]]
    
    return nearest_neighbors
def predict(fruit):
    num_apples = sum([1 for f in training_data if f[-1] == 'apple'])
    num_oranges = sum([1 for f in training_data if f[-1] == 'orange'])

    nearest_neighbors = find_nearest_neighbors(fruit, training_data, k=3)
    
    num_apples_nn = sum([1 for nn in nearest_neighbors if nn[-1] == 'apple'])
    num_oranges_nn = sum([1 for nn in nearest_neighbors if nn[-1] == 'orange'])

    if num_apples_nn > num_oranges_nn:
        return 'apple'
    else:
        return 'orange'
training_data = [
    ['red', 'round', 'apple'],
    ['red', 'oval', 'apple'],
    ['yellow', 'round', 'orange'],
    ['yellow', 'oval', 'orange']
]

test_fruit = ['red', 'round']
print(predict(test_fruit))