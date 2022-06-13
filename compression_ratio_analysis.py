import matplotlib.pyplot as plt
import numpy as np
import os


analysis_results_dirname = os.path.join(os.curdir, "analysis_results")
analysis_result_filename = "compression_ratio_comparison(simple MLP and CNN).txt"
filepath = os.path.join(analysis_results_dirname, analysis_result_filename)

categories = []
bdi_results = []
fpc_results = []


with open(filepath, "rt") as file:
    for line in file.readlines():
        cat, bdi_re, fpc_re = line.strip().split('\t')
        categories.append(cat)
        bdi_results.append(float(bdi_re))
        fpc_results.append(float(fpc_re))
        print(f"{cat:<30s}    {float(bdi_re):<.4f}    {float(fpc_re):<.4f}")


x_axis = np.arange(len(categories))
plt.bar(x_axis -0.2, bdi_results, width=0.4, label='BDI')
plt.bar(x_axis +0.2, fpc_results, width=0.4, label='FPC')
plt.xticks(x_axis, categories, rotation=45, ha='right')

plt.legend()
plt.tight_layout()
plt.show()
