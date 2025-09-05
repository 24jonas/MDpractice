# Agents for 'Economy.py'

# Imports
import numpy as np
import random
import matplotlib.pyplot as plt

# Variables
Initial_NAV = 1000000
Initial_GDP = 100000
Initial_Growth = 2
Expected_Return = 8
Time_Frame = 10

# Renaming
inav = Initial_NAV
igdp = Initial_GDP
igro = Initial_Growth/100
exrt = Expected_Return/100
time = Time_Frame

# Sectors
"""
[Sector, Number, Growth, Dividend, NAV, Revenue, Expenses, Balance, MCAP]
1. Government (govt) - [a, 1, 2%, 0%, 10%, 0, 15%, 0, 0]
2. Individuals (indv) - [b, 1, 2%, 0%, 15%, 0, 10%, 0, 0]
3. Technology (tech) - [c, 11, 5%, 0.5%, 8%, 0, 9%, 0, 0]
4. Finance (finc) - [d, 7, 3.5%, 2%, 7%, 0, 8%, 0, 0]
5. Healthcare (hlth) - [e, 2, 4%, 2.5%, 5%, 0, 5%, 0, 0]
6. Education (educ) - [f, 1, 3%, 1.5%, 4%, 0, 3%, 0, 0]
7. Real Estate (real) - [g, 6, 2%, 3.5%, 3%, 0, 2%, 0, 0]
8. Construction (cstr) - [h, 3, 1.5%, 3%, 3%, 0, 2%, 0, 0]
9. Consumption (cons) - [j, 12, 1%, 3.5%, 11%, 0, 10%, 0, 0]
10. Entertainment (entr) - [k, 9, 2.5%, 1%, 1%, 0, 2%, 0, 0]
11. Automotive (auto) - [l, 5, 1%, 2.5%, 2%, 0, 3%, 0, 0]
12. Transportation (tran) - [m, 10, 1%, 2%, 3%, 0, 3%, 0, 0]
13. Service (srvc) - [n, 13, 1.5%, 2%, 13%, 0, 12%, 0, 0]
14. Utilities (util) - [o, 4, 0.5%, 4%, 5%, 0, 5%, 0, 0]
15. Manufacturing (manu) - [p, 8, 1%, 3%, 10%, 0, 11%, 0, 0]
93 Total Market Agents
"""

# Initial
def Initial(sector, number, growth, dividend, nav, gdp, abrv):
    for i in range(number):
        a1 = growth*random.randint(50, 150)/100
        a2 = dividend*random.randint(50, 150)/100
        a3 = nav*random.randint(50, 150)/100/number
        a4 = gdp*random.randint(50, 150)/100/number
        a5 = round((nav*growth*dividend)/(np.absolute(exrt-growth)) + 0.1*nav, 2)*random.randint(90,110)/100
        abrv.append([sector, number, a1, a2, a3, 0, a4, 0, a5])
    return abrv

govt = Initial("a", 1, igro, 0, 0.1*inav, 0.15*igdp, [])
indv = Initial("b", 1, igro, 0, 0.15*inav, 0.15*igdp, [])
tech = Initial("c", 11, 2.5*igro, 0.005, 0.08*inav, 0.09*igdp, [])
finc = Initial("d", 7, 2*igro, 0.02, 0.07*inav, 0.08*igdp, [])
hlth = Initial("e", 2, 2.25*igro, 0.025, 0.05*inav, 0.05*igdp, [])
educ = Initial("f", 1, 1.75*igro, 0.015, 0.04*inav, 0.03*igdp, [])
real = Initial("g", 6, 0.75*igro, 0.035, 0.03*inav, 0.02*igdp, [])
cstr = Initial("h", 3, 0.75*igro, 0.03, 0.03*inav, 0.02*igdp, [])
cons = Initial("j", 12, 0.5*igro, 0.035, 0.11*inav, 0.1*igdp, [])
entr = Initial("k", 9, 1.5*igro, 0.01, 0.01*inav, 0.02*igdp, [])
auto = Initial("l", 5, 0.25*igro, 0.025, 0.02*inav, 0.03*igdp, [])
tran = Initial("m", 10, 0.25*igro, 0.02, 0.03*inav, 0.03*igdp, [])
srvc = Initial("n", 13, 1.25*igro, 0.02, 0.13*inav, 0.12*igdp, [])
util = Initial("o", 4, 0.1*igro, 0.04, 0.05*inav, 0.05*igdp, [])
manu = Initial("p", 8, 0.5*igro, 0.03, 0.1*inav, 0.11*igdp, [])
govt[0][8] = indv[0][8] = 0

# Agents
z = agents = []
z.extend(govt)
z.extend(indv)
z.extend(tech)
z.extend(finc)
z.extend(hlth)
z.extend(educ)
z.extend(real)
z.extend(cstr)
z.extend(cons)
z.extend(entr)
z.extend(auto)
z.extend(tran)
z.extend(srvc)
z.extend(util)
z.extend(manu)

# Record Lists
ln = len(z)
gl = np.zeros((time, ln))
dl = np.zeros((time, ln))
nl = np.zeros((time, ln))
rl = np.zeros((time, ln))
el = np.zeros((time, ln))
bl = np.zeros((time, ln))
ml = np.zeros((time, ln))

# Functions
def Private(z):   # Businesses pay half of their expenses on other businesses.
    for i in range(ln-2):
        expense = 0.1*z[i+2][6]
        for i in range(5):
            x = random.randint(2,ln-1)
            z[x][5] = z[x][5] + expense
    return z

def Public(z):     # Government pays half of its expenses on businesses.
    expense = 0.05*z[0][6]
    for i in range(10):
        x = random.randint(2,ln-1)
        z[x][5] = z[x][5] + expense
    return z

def Welfare(z):    # Government pays half of its expenses on individuals.
    expense = 0.5*z[0][6]
    z[1][5] = z[1][5] + expense
    return z

def Value(z):      # Value is created by businesses.
    for i in range(ln-2):
        x = random.randint(0,200)/100
        growth = x*z[i+2][2]*np.absolute(z[i+2][4])
        z[i+2][4] = z[i+2][4] + growth
    return z

def Consumer(z):   # Individuals pay 3/4 of their expenses to businesses.
    expense = 0.01*z[1][6]
    for i in range(75):
        x = random.randint(2,ln-1)
        z[x][5] = z[x][5] + expense
    return z

def Income(z):     # Businesses pay the other half of their expenses on individuals.
    for i in range(ln-2):
        expense = 0.5*z[i+2][6]
        z[1][5] = z[1][5] + expense
    return z

def TaxationA(z):   # Individuals pay 1/4 of their expenses to government.
    expense = 0.25*z[1][6]
    z[0][5] = z[0][5] + expense
    return z

def TaxationB(z):   # Government taxes the balances of business.
    for i in range(ln-2):
        z[i+2][7] = z[i+2][5] - z[i+2][6]
        if z[i+2][7] > 0:
            z[i+2][5] = z[i+2][5] - 0.1*z[i+2][7]
            z[0][5] = z[0][5] + 0.1*z[i+2][7]
    return z

def Investment(z, exrt):  # Individuals buy and sell shares.
    for i in range(ln-2):
        w = z[i+2][8]
        z[i+2][8] = round((z[i][5]*z[i][3])/(np.absolute(exrt-z[i][2])) + 0.1*z[i][4], 2)*random.randint(90,110)/100
        if z[i+2][8] > 10*z[i+2][5]:
            z[i+2][8] = 0.9*z[i+2][8]
        z[0][8] = z[1][8] = 0
        z[i+2][4] = z[i+2][4] - z[i][3]*z[i][8]
        if z[i+2][8] < 0:
            z[i+2][8] = 0
        Capital_Gain = z[i+2][8] - w
        z[i+2][4] = z[i+2][4] + Capital_Gain
        z[1][4] = z[1][4] - Capital_Gain
    return z

def Accounting(z):       # Calulate NAV, Growth, etc...
    if z[0][7] < 0:
        z[0][7] = 0.9*z[0][7]   # Money printed go brrr...
    for i in range(ln):
        z[i][7] = z[i][5] - z[i][6]
        z[i][4] = z[i][4] + z[i][7]
        if z[i][4] < -5*z[i][5] and z[i][4] < 0 and (i != 0 or 1):  # Bankruptcy
            z[1][4] = z[1][4] + z[i][4]
            z[i][4] = 0
            z[1][4] = z[1][4] + z[i][8]
            z[i][8] = 0
        if z[i][7] > 0:             # Growth Reduction
            z[i][2] = 0.9*z[i][2]
        if z[i][7] < 0:
            z[i][2] = 0.8*z[i][2]
    return z

def Record(z, i, gl, dl, nl, rl, el, bl, ml):      # Record all numerical information over time.
    for j in range(ln):
        gl[i][j] = z[j][2]
        dl[i][j] = z[j][3]
        nl[i][j] = z[j][4]
        rl[i][j] = z[j][5]
        el[i][j] = z[j][6]
        bl[i][j] = z[j][7]
        ml[i][j] = z[j][8]
        z[i][6] = z[i][4]*random.randint(90,110)/1000 + z[i][7]
        z[i][5] = 0
    z[0][6] = 0.9*z[0][4]   # Government expenses
    return z, gl, dl, nl, rl, el, bl, ml

# Simulation
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
plt.ylabel("NAV")
plt.title("Basket Of NAVs")
plt.show()

# Plot a basket of MCAPs
x = np.linspace(1, time, time)
y1, y2, y3, y4, y5 = ml[:, 0], ml[:, 1], ml[:, 2], ml[:, 3], ml[:, 4]
y6, y7, y8, y9, y10 = ml[:, 5], ml[:, 6], ml[:, 7], ml[:, 8], ml[:, 9]
y11, y12, y13, y14, y15 = ml[:, 10], ml[:, 11], ml[:, 12], ml[:, 13], ml[:, 14]
y16, y17, y18, y19, y20 = ml[:, 15], ml[:, 16], ml[:, 17], ml[:, 18], ml[:, 19]
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
