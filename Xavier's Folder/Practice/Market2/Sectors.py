# Agents for 'Economy.py'

# Imports
import numpy as np
import random

# Variables
Initial_NAV = 1000000
Initial_GDP = 100000
Initial_Growth = 0
Expected_Return = 8
Time_Frame = 25

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
        a5 = round((nav*growth*dividend)/(np.absolute(exrt-growth)) + 0.5*nav, 2)*random.randint(90,110)/100
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
