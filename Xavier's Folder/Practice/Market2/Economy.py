# Economic model for 'Market.py'

# Inputs
from Functions import *
import matplotlib.pyplot as plt

for i in range(time):
    z = Private(z)
    z = Public(z)
    z = Value(z)
    z = Consumer(z)
    z = Income(z)
    z = TaxationA(z)
    z = TaxationB(z)
    z = Welfare(z)
    z = Investment(z, exrt)
    z = Accounting(z)
    z, gl, dl, nl, rl, el, bl, ml = Record(z, i, gl, dl, nl, rl, el, bl, ml)

# Plot a basket of NAVs
x = np.linspace(1, time, time)
y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = nl[:, 0], nl[:, 1], nl[:, 10], nl[:, 20], nl[:, 30], nl[:, 40], nl[:, 50], nl[:, 60], nl[:, 70], nl[:, 80]
y11 = nl[:, -1]
plt.plot(x, y1, marker='.', color = "black")
plt.plot(x, y2, marker='.', color = "gray")
plt.plot(x, y3, marker='.', color = "red")
plt.plot(x, y4, marker='.', color = "orange")
plt.plot(x, y5, marker='.', color = "yellow")
plt.plot(x, y6, marker='.', color = "green")
plt.plot(x, y7, marker='.', color = "blue")
plt.plot(x, y8, marker='.', color = "purple")
plt.plot(x, y9, marker='.', color = "pink")
plt.plot(x, y10, marker='.', color = "brown")
plt.plot(x, y11, marker='.', color = "gold")      # Total NAV
ymins = [min(y1), min(y2), min(y3), min(y4), min(y5), min(y6), min(y7), min(y8), min(y9), min(y10), min(y11)]
ymaxes = [max(y1), max(y2), max(y3), max(y4), max(y5), max(y6), max(y7), max(y8), max(y9), max(y10), max(y11)]
ymin = min(ymins)
ymax = max(ymaxes)
if ymin < 0:
    w = ymin
else:
    w = 0
#plt.ylim(w, 1.1*ymax)
plt.xlabel("Time")
plt.ylabel("NAV")
plt.title("Basket Of NAVs")
plt.show()

# Plot a basket of MCAPs
x = np.linspace(1, time, time)
y1, y2, y3, y4, y5 = ml[:, 0], ml[:, 1], ml[:, 2], ml[:, 3], ml[:, 4]
y6, y7, y8, y9, y10 = ml[:, 5], ml[:, 6], ml[:, 7], ml[:, 8], ml[:, 9]
y11, y12, y13, y14, y15 = ml[:, 10], ml[:, 11], ml[:, 12], ml[:, 13], ml[:, 14]
y16, y17, y18, y19, y20 = ml[:, 15], ml[:, 16], ml[:, 17], ml[:, 18], ml[:, 19]
y21 = ml[:, -1]
plt.plot(x, y1, marker='.', color = "red")
plt.plot(x, y2, marker='.', color = "orange")
plt.plot(x, y3, marker='.', color = "yellow")
plt.plot(x, y4, marker='.', color = "green")
plt.plot(x, y5, marker='.', color = "cyan")
plt.plot(x, y6, marker='.', color = "blue")
plt.plot(x, y7, marker='.', color = "purple")
plt.plot(x, y8, marker='.', color = "magenta")
plt.plot(x, y9, marker='.', color = "pink")
plt.plot(x, y10, marker='.', color = "brown")
plt.plot(x, y11, marker='.', color = "red")
plt.plot(x, y12, marker='.', color = "orange")
plt.plot(x, y13, marker='.', color = "yellow")
plt.plot(x, y14, marker='.', color = "green")
plt.plot(x, y15, marker='.', color = "cyan")
plt.plot(x, y16, marker='.', color = "blue")
plt.plot(x, y17, marker='.', color = "purple")
plt.plot(x, y18, marker='.', color = "magenta")
plt.plot(x, y19, marker='.', color = "pink")
plt.plot(x, y20, marker='.', color = "brown")
# plt.plot(x, y21, marker='.', color = "gold")      # Total Market Capitalization
ymins = [min(y1), min(y2), min(y3), min(y4), min(y5)]
ymaxes = [max(y1), max(y2), max(y3), max(y4), max(y5)]
ymin = min(ymins)
ymax = max(ymaxes)
if ymin < 0:
    w = ymin
else:
    w = 0
#plt.ylim(w, 1.1*ymax)
plt.xlabel("Time")
plt.ylabel("MCAP")
plt.title("Basket Of MCAPs")
plt.show()
