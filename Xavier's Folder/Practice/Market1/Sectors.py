# Simulate a simplified miniature economy for purposes of valueing securities.

# imports
import numpy as np
from Functions import *

# Sectors [NAV, Earnings, Expendiutres, Balance, Earnings Growth, Dividend, MCAP]
Technology = np.zeros((7,))         # 11
Finance = np.zeros((7,))            # 7
Healthcare = np.zeros((7,))         # 2
Education = np.zeros((7,))          # 1
Real_Estate = np.zeros((7,))        # 6
Construction = np.zeros((7,))       # 4
Consumption = np.zeros((7,))        # 12
Entertainment = np.zeros((7,))      # 8
Automotive = np.zeros((7,))         # 5
Transportation = np.zeros((7,))     # 9
Service = np.zeros((7,))            # 13
Utilities = np.zeros((7,))          # 3
Manufacturing = np.zeros((7,))      # 10
Individuals = np.zeros((7,))        # 1
# Government                        # 1
Total = np.zeros((7,))              # 91 + 1 + 1

# Technology (11)
a = []
for i in range(11):
    company  = Initial(NAV*0.08, GDP*0.1, Growth*2, Dividend, Return)
    a.append(company)
for company in a:
    Technology = Technology + company
Total = Total + Technology

# Finance (7)
b = []
for i in range(7):
    company = Initial(NAV*0.07, GDP*0.08, Growth*1.2, Dividend, Return)
    b.append(company)
for company in b:
    Finance = Finance + company
Total = Total + Finance

# Healtcare (2)
c = []
for i in range(2):
    company = Initial(NAV*0.05, GDP*0.05, Growth*1.5, Dividend, Return)
    c.append(company)
for company in c:
    Healthcare = Healthcare + company
Total = Total + Healthcare

# Education (1)
d = []
for i in range(1):
    company = Initial(NAV*0.02, GDP*0.03, Growth*1.3, Dividend, Return)
    d.append(company)
for company in d:
    Education = Education + company
Total = Total + Education

# Real Estate (6)
e = []
for i in range(6):
    company = Initial(NAV*0.06, GDP*0.04, Growth*1, Dividend, Return)
    e.append(company)
for company in e:
    Real_Estate = Real_Estate + company
Total = Total + Real_Estate

# Construction (4)
f = []
for i in range(4):
    company = Initial(NAV*0.03, GDP*0.03, Growth*0.8, Dividend, Return)
    f.append(company)
for company in f:
    Construction = Construction + company
Total = Total + Construction

# Consumption (12)
g = []
for i in range(12):
    company = Initial(NAV*0.06, GDP*0.1, Growth*0.9, Dividend, Return)
    g.append(company)
for company in g:
    Consumption = Consumption + company
Total = Total + Consumption

# Entertainment (8)
h = []
for i in range(8):
    company = Initial(NAV*0.01, GDP*0.03, Growth*0.7, Dividend, Return)
    h.append(company)
for company in h:
    Entertainment = Entertainment + company
Total = Total + Entertainment

# Automotive (5)
ii = []
for i in range(5):
    company = Initial(NAV*0.02, GDP*0.02, Growth*0.6, Dividend, Return)
    ii.append(company)
for company in ii:
    Automotive = Automotive + company
Total = Total + Automotive

# Transportation (9)
j = []
for i in range(9):
    company = Initial(NAV*0.04, GDP*0.02, Growth*0.5, Dividend, Return)
    j.append(company)
for company in j:
    Transportation = Transportation + company
Total = Total + Transportation

# Service (13)
k = []
for i in range(13):
    company = Initial(NAV*0.07, GDP*0.12, Growth*0.9, Dividend, Return)
    k.append(company)
for company in k:
    Service = Service + company
Total = Total + Service

# Utilities (3)
l = []
for i in range(3):
    company = Initial(NAV*0.13, GDP*0.06, Growth*0.5, Dividend, Return)
    l.append(company)
for company in l:
    Utilities = Utilities + company
Total = Total + Utilities

# Manufacturing (10)
m = []
for i in range(10):
    company = Initial(NAV*0.1, GDP*0.1, Growth*0.7, Dividend, Return)
    m.append(company)
for company in m:
    Manufacturing = Manufacturing + company
Total = Total + Manufacturing

# Individuals (1) - Not Listed
n = []
Individuals = np.array([NAV*0.1, GDP/10, GDP/10, 0, 0, 0, 0])
n.append(Individuals)
Total = Total + Individuals

# Average growth and dividends.
Total[4] = Total[4]/91
Total[5] = Total[5]/91
# print(Total)    # Private Economy

# Governments (1) - Not Listed
o = []
Government = np.array([NAV, GDP, GDP, 0, Growth, 0, 0]) - Total
Government[4] = 0
Government[5] = 0
Government[6] = 0
o.append(Government)
Economy = Total + Government

# Total Economy
# print(Economy)  # Private and public economy

# List of agents
Agents = []
Agents.extend(a)
Agents.extend(b)
Agents.extend(c)
Agents.extend(d)
Agents.extend(e)
Agents.extend(f)
Agents.extend(g)
Agents.extend(h)
Agents.extend(ii)
Agents.extend(j)
Agents.extend(k)
Agents.extend(l)
Agents.extend(m)
Agents.extend(n)
Agents.extend(o)
# print(Agents)   # All initial values.

Sectors = [a, b, c, d, e, f, g, h, ii, j, k, l, m, n, o]
