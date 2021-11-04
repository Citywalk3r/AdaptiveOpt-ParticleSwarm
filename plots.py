import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# plt.style.use(['seaborn-white', 'seaborn-paper'])
matplotlib.rc("font", family="monospace")
# import seaborn as sns

column_names = ['iterations', 'h', 776, 12, 234, 9238, 123556, 59933, 98232, 85732, 5432, 12291]

df = pd.read_excel("../ps_var_it_800s_10seeds.xlsx")
problem = "Eggholder Function"

fig = plt.figure(figsize=(10, 5))

# Effect of ps
grouped = pd.melt(df, id_vars=["iterations", "swarm_size"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[(grouped["iterations"] == 20) & (grouped["swarm_size"] == 1000)]
plt.subplot(2,2,1)
plt.title('iterations = 20')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["iterations", "swarm_size"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[(grouped["iterations"] == 50) & (grouped["swarm_size"] == 1000)]
plt.subplot(2,2,2)
plt.title('iterations = 50')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["iterations", "swarm_size"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[(grouped["iterations"] == 100) & (grouped["swarm_size"] == 1000)]
plt.subplot(2,2,3)
plt.title('iterations = 100')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["iterations", "swarm_size"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[(grouped["iterations"] == 500) & (grouped["swarm_size"] == 1000)]
plt.subplot(2,2,4)
plt.title('iterations = 500')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

# grouped = pd.melt(df, id_vars=["iterations"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
# grouped = grouped[grouped["iterations"] == 200]
# plt.subplot(2,2,4)
# plt.title('iterations = 200')
# # plt.yscale('log')
# boxplot = grouped.boxplot(column="value")

plt.suptitle("Impact of iterations on PS for the " + problem)
plt.show()


# Effect of swarm size

grouped = pd.melt(df, id_vars=["swarm_size"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["swarm_size"] == 100]
plt.subplot(2,2,1)
plt.title('swarm size = 100')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["swarm_size"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["swarm_size"] == 200]
plt.subplot(2,2,2)
plt.title('swarm size = 200')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["swarm_size"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["swarm_size"] == 400]
plt.subplot(2,2,3)
plt.title('swarm size = 400')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["swarm_size"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["swarm_size"] == 800]
plt.subplot(2,2,4)
plt.title('swarm size = 800')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

plt.suptitle("Impact of swarm_size on PS for the " + problem)
plt.show()


# Effect of iterations

grouped = pd.melt(df, id_vars=["iterations"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["iterations"] == 50]
plt.subplot(2,2,1)
plt.title('iterations = 50')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["iterations"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["iterations"] == 100]
plt.subplot(2,2,2)
plt.title('iterations = 100')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["iterations"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["iterations"] == 200]
plt.subplot(2,2,3)
plt.title('iterations = 200')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["iterations"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["iterations"] == 300]
plt.subplot(2,2,4)
plt.title('iterations = 300')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

plt.suptitle("Impact of iterations on PS for the " + problem)
plt.show()
