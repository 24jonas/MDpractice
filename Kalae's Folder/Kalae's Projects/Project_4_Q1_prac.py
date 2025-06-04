import numpy as np
import matplotlib.pyplot as plt

#not my code. i was just looking for an example.

# Initial positions
r1 = np.array([2.0, 5.0])
r2 = np.array([5.0, 5.0])

# Initial velocities (r1 moving right, r2 stationary)
v1 = np.array([1.0, 0.0])
v2 = np.array([0.0, 0.0])

# Function to compute elastic collision response
def elastic_collision(r1, r2, v1, v2):
    d = r1 - r2
    dist_squared = np.dot(d, d)
    if dist_squared == 0:
        return v1, v2  # Avoid division by zero
    v_rel = v1 - v2
    proj = np.dot(v_rel, d) / dist_squared
    v1_new = v1 - proj * d
    v2_new = v2 + proj * d
    return v1_new, v2_new

# Simulate one collision
v1_after, v2_after = elastic_collision(r1, r2, v1, v2)

# Display results
print("Before collision:")
print(f"v1 = {v1}, v2 = {v2}")
print("\nAfter collision:")
print(f"v1 = {v1_after}, v2 = {v2_after}")