import matplotlib.pyplot as plt
import numpy as np

original_size = 64
categories = ['Original', 'BDI', 'FPC', 'BDI 2B', 'BDI+ZR', 'ZV', 'ZR', 'BDI+ZE']
values = {
    'sparse no pattern': [original_size, 64, 24, 64, 64, 27, 27, 64],
    'repeating':         [original_size, 40, 19, 40, 33, 64, 64, 44],
    'dense no pattern':  [original_size, 64, 64, 64, 64, 64, 64, 64],
    'sparse':            [original_size, 64, 24, 64, 64, 27, 27, 64],
}

width_max = 0.8
width = width_max / len(values.keys())

x_axis = np.arange(len(categories))

for idx, (key, val) in enumerate(values.items()):
    print((idx - (len(values.keys()) / 2)))
    plt.bar(x_axis + ((idx - (len(values.keys()) / 2) + 0.5) * width), original_size / np.array(val), width=width, label=key)

plt.xticks(x_axis, categories)

plt.legend()
plt.tight_layout()
plt.show()
