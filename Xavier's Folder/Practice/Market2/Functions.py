# Functions for 'Economy.py'

# Import
from Sectors import *

# Record Lists
ln = len(z)
gl = np.zeros((time, ln+1))
dl = np.zeros((time, ln+1))
nl = np.zeros((time, ln+1))
rl = np.zeros((time, ln+1))
el = np.zeros((time, ln+1))
bl = np.zeros((time, ln+1))
ml = np.zeros((time, ln+1))

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

def Value(z):      # Value is created by businesses.
    y = random.randint(-500, 500)/100   # Volatility
    for i in range(ln-2):
        x = random.randint(0,200)/100
        growth = y*x*z[i+2][2]*np.absolute(z[i+2][4])
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
            z[i+2][5] = z[i+2][5] - 0.25*z[i+2][7]
            z[0][5] = z[0][5] + 0.25*z[i+2][7]
    return z

def Welfare(z):    # Government pays half of its expenses on individuals.
    expense = 0.5*z[0][6]
    z[1][5] = z[1][5] + expense
    return z

def Investment(z, exrt):  # Individuals buy and sell shares.
    for i in range(ln-2):
        w = z[i+2][8]
        z[i+2][8] = round((z[i][5]*z[i][3])/(np.absolute(exrt-z[i][2])) + 0.5*z[i][4], 2)*random.randint(90,110)/100
        if z[i+2][8] > 10*z[i+2][5]:
            z[i+2][8] = 0.9*z[i+2][8]
        if z[i+2][8] > 20*z[i+2][5]:
            z[i+2][8] = 0.9*z[i+2][8]
        if z[i+2][8] > 30*z[i+2][5]:
            z[i+2][8] = 0.9*z[i+2][8]
        if z[i+2][8] > 40*z[i+2][5]:
            z[i+2][8] = 0.9*z[i+2][8]
        if z[i+2][8] > 40*z[i+2][5]:
            z[i+2][8] = 0.9*z[i+2][8]
        z[0][8] = z[1][8] = 0
        z[i+2][4] = z[i+2][4] - z[i][3]*z[i][8]
        z[1][4] = z[1][4] + z[i][3]*z[i][8]
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
        if z[i][4] > z[0][4] and z[i][4] > 0:
            extra = z[i][4] - z[0][4]
            z[i][4] = z[i][4] - extra
            z[0][4] = z[0][4] + extra
    return z

def Record(z, i, gl, dl, nl, rl, el, bl, ml):      # Record all numerical information over time.
    for j in range(ln):
        gl[i][j] = z[j][2]
        gl[i][-1] += z[j][2]
        dl[i][j] = z[j][3]
        dl[i][-1] += z[j][3]
        nl[i][j] = z[j][4]
        nl[i][-1] += z[j][4]
        rl[i][j] = z[j][5]
        rl[i][-1] += z[j][5]
        el[i][j] = z[j][6]
        el[i][-1] += z[j][6]
        bl[i][j] = z[j][7]
        bl[i][-1] += z[j][7]
        ml[i][j] = z[j][8]
        ml[i][-1] += z[j][8]
        z[i][6] = z[i][4]*random.randint(90,110)/1000 + z[i][7]
        z[i][5] = 0
    z[0][6] = 0.75*z[0][4]   # Government expenses
    return z, gl, dl, nl, rl, el, bl, ml
