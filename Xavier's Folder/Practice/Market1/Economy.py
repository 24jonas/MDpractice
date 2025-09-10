# Economic activity

# Imports
from Sectors import *
import matplotlib.pyplot as plt

# Activity
for i in range(Time_Period):
    Agents = Business(Agents)
    Agents = Public(Agents)
    Agents = Consumers(Agents)
    Agents = Income(Agents)
    Agents = Taxes(Agents)
    Agents = Accounting(Agents, Return, Growth)
    Agents, nav_list, earnings_list, expenditures_list, balance_list, earnings_growth_list, dvidend_list, mcap_list = Record(Agents, i, nav_list, earnings_list, expenditures_list, balance_list, earnings_growth_list, dvidend_list, mcap_list)

# Plot a basket of NAVs
x, y1, y2, y3, y4, y5 = np.linspace(1, Time_Period, Time_Period), nav_list[:, 0], nav_list[:, 25], nav_list[:, 50], nav_list[:, 91], nav_list[:, 92]
plt.plot(x, y1, marker='o', color = "red")
plt.plot(x, y2, marker='o', color = "orange")
plt.plot(x, y3, marker='o', color = "green")
plt.plot(x, y4, marker='o', color = "blue")
plt.plot(x, y5, marker='o', color = "purple")
ymins = [min(y1), min(y2), min(y3), min(y4), min(y5)]
ymaxes = [max(y1), max(y2), max(y3), max(y4), max(y5)]
ymin = min(ymins)
ymax = max(ymaxes)
if ymin < 0:
    z = ymin
else:
    z = 0
plt.ylim(z, 1.1*ymax)
plt.xlabel("Time")
plt.ylabel("NAV")
plt.title("Basket Of NAVs")
plt.show()

# Plot a basket of MCAPs
x, y1, y2, y3, y4, y5 = np.linspace(1, Time_Period, Time_Period), mcap_list[:, 0], mcap_list[:, 20], mcap_list[:, 40], mcap_list[:,60], mcap_list[:, 80]
plt.plot(x, y1, marker='o', color = "red")
plt.plot(x, y2, marker='o', color = "orange")
plt.plot(x, y3, marker='o', color = "green")
plt.plot(x, y4, marker='o', color = "blue")
plt.plot(x, y5, marker='o', color = "purple")
ymins = [min(y1), min(y2), min(y3), min(y4), min(y5)]
ymaxes = [max(y1), max(y2), max(y3), max(y4), max(y5)]
ymin = min(ymins)
ymax = max(ymaxes)
if ymin < 0:
    z = ymin
else:
    z = 0
plt.ylim(z, 1.1*ymax)
plt.xlabel("Time")
plt.ylabel("NAV")
plt.title("Basket Of MCAPs")
plt.show()
