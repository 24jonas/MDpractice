import numpy as np

## Acceleration Function for Position Verlet ##
def acc(R, R_1, R_2, m_1, m_2):

    r_mag_1 = np.sqrt((R[0, 0] - R_1[0, 0]) ** 2 + (R[0, 1] - R_1[0, 1]) **2)
    r_mag_2 = np.sqrt((R[0, 0] - R_2[0, 0]) ** 2 + (R[0, 1] - R_2[0, 1]) **2)

    A = -m_1 * (R - R_1) / (r_mag_1 ** 3) - m_2 * (R - R_2) / (r_mag_2 ** 3)

    return A