# Functions for 'Exchange.py'

# Imports
import numpy as np
import random

# Variables
NAV = 1000000
GDP = 100000
Growth = 5
Dividend = 2
Return = 8
Time_Period = 5
nav_list = np.zeros((Time_Period, 93))
earnings_list = np.zeros((Time_Period, 93))
expenditures_list = np.zeros((Time_Period, 93))
balance_list = np.zeros((Time_Period, 93))
earnings_growth_list = np.zeros((Time_Period, 93))
dvidend_list = np.zeros((Time_Period, 93))
mcap_list = np.zeros((Time_Period, 93))

# Functions
def Initial(NAV, GDP, Growth, Dividend, Return):
    nav = NAV*random.randint(5, 20)/100
    earnings = GDP*random.randint(1, 10)/50
    expenditures = earnings*random.randint(20, 70)/50
    balance = earnings - expenditures
    earnings_growth = Growth*random.randint(-10, 30)/1000
    dividend = Dividend*random.randint(1, 200)/10000
    mcap = round((earnings*dividend)/(Return-earnings_growth) + 0.9*nav, 2)
    if mcap < 0:
        mcap = 0
    return np.array([nav, earnings, expenditures, balance, earnings_growth, dividend, mcap])

def Business(Agents):
    for i in range(91):
        expenditure = Agents[i][2]
        for i in range(5):
            x = random.randint(0,92)
            Agents[x][1] = Agents[x][1] + 0.2*expenditure
    return Agents

def Public(Agents):
    for i in range(20):
        x = random.randint(0,90)
        Agents[x][1] = Agents[x][1] + 0.1*Agents[92][2]
    return Agents

def Consumers(Agents):
    z = 0.009
    if Agents[91][3] < 0:
        z = 0.008
    for i in range(100):
        x = random.randint(0,92)
        Agents[91][2] = Agents[91][2] + z*Agents[91][0]
        Agents[x][1] = Agents[x][1] + z*Agents[91][0]
    return Agents

def Income(Agents):
    for i in range(91):
        Agents[i][2] = Agents[i][2] + 0.25*Agents[i][1]
        Agents[91][1] = Agents[91][1] + 0.25*Agents[i][1]
    Agents[91][1] = Agents[91][1] + 0.5*Agents[92][2]
    return Agents

def Taxes(Agents):
    for i in range(92):
        Agents[i][3] = Agents[i][1] - Agents[i][2]
        if Agents[i][3] > 0:
            Agents[i][2] = Agents[i][2] + 0.1*Agents[i][3]
            Agents[92][1] = Agents[92][1] + 0.1*Agents[i][3]
    return Agents

def Accounting(Agents, Return, Growth):
    for i in range(93):
        Agents[i][3] = Agents[i][1] - Agents[i][2]
        Agents[i][4] = Agents[i][3]/Agents[i][0]
        Agents[i][0] = Agents[i][0] + Agents[1][3] - Agents[i][5]*Agents[i][6] + 1000
        Agents[91][0] = Agents[91][0] + Agents[i][5]*Agents[i][6]
        Agents[i][6] = round((Agents[i][1]*Agents[i][5])/(np.absolute(Return-Agents[i][4])) + 0.5*Agents[i][0], 2)
        if Agents[i][6] < 0:
            Agents[i][6] = 0
        if Agents[i][0] < -5*Agents[i][1]:
            Agents[92][2] = Agents[92][2] - Agents[i][0]
            Agents[i][0] = 0
        Agents[i][2] = Agents[i][2]*random.randint(85,105)/100
        Agents[i][1] = 0
    Agents[91][6] = Agents[92][6] = 0
    Agents[92][0] = Agents[92][0] + Growth*np.absolute(Agents[92][0])/100 + 10000 # Money printer go brrr...
    Agents[92][2] = Agents[92][0]
    return Agents

def Record(Agents, i, nav_list, earnings_list, expenditures_list, balance_list, earnings_growth_list, dvidend_list, mcap_list):
    for j in range(93):
        nav_list[i][j] = Agents[j][0]
        earnings_list[i][j] = Agents[j][1]
        expenditures_list[i][j] = Agents[j][2]
        balance_list[i][j] = Agents[j][3]
        earnings_growth_list[i][j] = Agents[j][4]
        dvidend_list[i][j] = Agents[j][5]
        mcap_list[i][j] = Agents[j][6]
    return Agents, nav_list, earnings_list, expenditures_list, balance_list, earnings_growth_list, dvidend_list, mcap_list
