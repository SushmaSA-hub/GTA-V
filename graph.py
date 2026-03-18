import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Proposed', 'Deco', 'Local', 'Random', 'Greedy']
values = [229.815, 87.195, 785256.49, 75425.015, 87.115]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(methods, values, color='skyblue')
plt.yscale('log')  # Apply log scale to Y-axis
plt.ylabel('Value (log scale)')
plt.title('Comparison of Methods with Log Scaling')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
