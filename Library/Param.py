"""
Module-level documentation goes here.
"""

import numpy as np

# --8<-- [start:group-size]
GROUPS = ["A", "B", "C", "D"]

GROUP_SIZE = {
    "A": 2,
    "B": 3,
    "C": 3,
    "D": 3,
}

N_ORBITALS = sum(GROUP_SIZE.values())   # 11
# --8<-- [end:group-size]

# --8<-- [start:fitted-values]
# --8<-- [start:fitted-values-preview]
params_H0_AA = np.zeros((3, 4, 9), dtype=float)
params_H0_AA[0, 0, 1] = -4.873
params_H0_AA[1, 0, 1] = -2.498
params_H0_AA[2, 0, 0] = -0.890
# --8<-- [end:fitted-values-preview]

params_H0_BB = np.zeros((3, 4, 9), dtype=float)
params_H0_BB[0, 0, 0] = -6.720
params_H0_BB[0, 0, 1] = -7.235
params_H0_BB[1, 0, 0] =  1.623
params_H0_BB[1, 0, 1] = -1.500
params_H0_BB[2, 0, 0] = -0.094
params_H0_BB[2, 0, 1] =  0.273


params_H0_CC = np.zeros((3, 4, 9), dtype=float)
params_H0_CC[0, 0, 0] = -6.082
params_H0_CC[0, 0, 1] = -5.856
params_H0_CC[1, 0, 0] = -1.021
params_H0_CC[1, 0, 1] = -1.817
params_H0_CC[2, 0, 0] = -0.370
params_H0_CC[2, 0, 1] = -0.043


params_H0_DD = np.zeros((3, 4, 9), dtype=float)
params_H0_DD[0, 0, 0] = -8.839
params_H0_DD[0, 0, 1] = -7.850
params_H0_DD[1, 0, 0] = -0.858
params_H0_DD[1, 0, 1] = -3.317
params_H0_DD[2, 0, 0] = -1.142
params_H0_DD[2, 0, 1] =  0.720


params_H1_BA = np.zeros((3, 4, 9), dtype=float)
params_H1_BA[0, 1, 0] = -0.789
params_H1_BA[0, 1, 1] =  2.158
params_H1_BA[0, 1, 3] = -1.379
params_H1_BA[1, 1, 0] =  0.545
params_H1_BA[1, 1, 1] = -0.605
params_H1_BA[1, 1, 3] =  1.845
params_H1_BA[2, 1, 0] = -1.076
params_H1_BA[2, 1, 1] =  0.401
params_H1_BA[2, 1, 3] = -2.100
params_H1_BA[2, 1, 5] =  0.859
params_H1_BA[2, 1, 7] = -0.377
params_H1_BA[2, 1, 8] = -0.836


params_H1_DC = np.zeros((3, 4, 9), dtype=float)
params_H1_DC[0, 1, 0] =  1.411
params_H1_DC[0, 1, 1] =  0.652
params_H1_DC[0, 1, 2] = -0.940
params_H1_DC[0, 1, 3] = -0.954
params_H1_DC[0, 1, 4] = -0.883
params_H1_DC[1, 1, 0] = -0.486
params_H1_DC[1, 1, 1] =  0.843
params_H1_DC[1, 1, 2] =  2.178
params_H1_DC[1, 1, 3] =  0.446
params_H1_DC[1, 1, 4] = -0.208
params_H1_DC[2, 1, 0] =  1.724
params_H1_DC[2, 1, 1] = -0.353
params_H1_DC[2, 1, 2] = -2.204
params_H1_DC[2, 1, 3] = -0.682
params_H1_DC[2, 1, 4] = -0.850
params_H1_DC[2, 1, 5] =  0.899
params_H1_DC[2, 1, 6] = -0.542
params_H1_DC[2, 1, 7] = -2.093
params_H1_DC[2, 1, 8] =  1.101


params_H3_DC = np.zeros((3, 4, 9), dtype=float)
params_H3_DC[0, 3, 0] =  0.014
params_H3_DC[0, 3, 1] = -0.245
params_H3_DC[0, 3, 2] = -0.150
params_H3_DC[0, 3, 3] = -0.221
params_H3_DC[0, 3, 4] = -0.069
params_H3_DC[1, 3, 0] =  0.173
params_H3_DC[1, 3, 1] =  0.204
params_H3_DC[1, 3, 2] =  0.567
params_H3_DC[1, 3, 3] =  0.744
params_H3_DC[1, 3, 4] =  0.035
params_H3_DC[2, 3, 0] = -0.178
params_H3_DC[2, 3, 1] = -1.069
params_H3_DC[2, 3, 2] = -0.070
params_H3_DC[2, 3, 3] = -0.267
params_H3_DC[2, 3, 4] = -0.281
params_H3_DC[2, 3, 5] = -0.690   # Fang 2018 Table VII, beta_5 of H_DC^(3), MoS2 (was +0.690)
params_H3_DC[2, 3, 6] = -0.382
params_H3_DC[2, 3, 7] = -0.340
params_H3_DC[2, 3, 8] =  0.015


params_H2_AA = np.zeros((3, 4, 9), dtype=float)
params_H2_AA[0, 2, 0] = -0.206
params_H2_AA[0, 2, 1] =  0.031
params_H2_AA[0, 2, 3] = -0.257
params_H2_AA[1, 2, 0] = -0.258
params_H2_AA[1, 2, 1] = -0.202
params_H2_AA[1, 2, 3] =  0.705
params_H2_AA[2, 2, 0] = -0.676
params_H2_AA[2, 2, 1] = -0.192
params_H2_AA[2, 2, 3] =  0.555
params_H2_AA[2, 2, 6] = -0.095


params_H2_BB = np.zeros((3, 4, 9), dtype=float)
params_H2_BB[0, 2, 0] =  0.865
params_H2_BB[0, 2, 1] = -0.187
params_H2_BB[0, 2, 2] = -0.174
params_H2_BB[0, 2, 3] = -0.070
params_H2_BB[0, 2, 4] =  0.100
params_H2_BB[0, 2, 5] = -0.068
params_H2_BB[1, 2, 0] = -1.841
params_H2_BB[1, 2, 1] = -0.027
params_H2_BB[1, 2, 2] =  0.444
params_H2_BB[1, 2, 3] = -0.045
params_H2_BB[1, 2, 4] = -0.210
params_H2_BB[1, 2, 5] =  0.141
params_H2_BB[2, 2, 0] = -2.203
params_H2_BB[2, 2, 1] =  0.768
params_H2_BB[2, 2, 2] =  0.350
params_H2_BB[2, 2, 3] = -0.065
params_H2_BB[2, 2, 4] = -0.208
params_H2_BB[2, 2, 5] =  0.096
params_H2_BB[2, 2, 6] =  0.482
params_H2_BB[2, 2, 7] = -0.146
params_H2_BB[2, 2, 8] = -0.089


params_H2_CC = np.zeros((3, 4, 9), dtype=float)
params_H2_CC[0, 2, 0] =  0.275
params_H2_CC[0, 2, 1] = -0.558
params_H2_CC[0, 2, 2] = -0.298
params_H2_CC[0, 2, 3] = -0.249
params_H2_CC[0, 2, 4] =  0.114
params_H2_CC[0, 2, 5] =  0.410
params_H2_CC[1, 2, 0] = -1.027
params_H2_CC[1, 2, 1] =  1.544
params_H2_CC[1, 2, 2] =  1.032
params_H2_CC[1, 2, 3] =  0.206
params_H2_CC[1, 2, 4] =  0.285
params_H2_CC[1, 2, 5] = -0.738
params_H2_CC[2, 2, 0] = -0.910
params_H2_CC[2, 2, 1] =  1.337
params_H2_CC[2, 2, 2] =  0.376
params_H2_CC[2, 2, 3] = -0.003
params_H2_CC[2, 2, 4] =  0.188
params_H2_CC[2, 2, 5] = -0.779
params_H2_CC[2, 2, 6] = -0.634
params_H2_CC[2, 2, 7] =  0.288
params_H2_CC[2, 2, 8] = -0.152


params_H2_DD = np.zeros((3, 4, 9), dtype=float)
params_H2_DD[0, 2, 0] =  0.912
params_H2_DD[0, 2, 1] =  0.006
params_H2_DD[0, 2, 2] = -0.192
params_H2_DD[0, 2, 3] = -0.038
params_H2_DD[0, 2, 4] = -0.106
params_H2_DD[0, 2, 5] =  0.008
params_H2_DD[1, 2, 0] = -1.425
params_H2_DD[1, 2, 1] = -0.057
params_H2_DD[1, 2, 2] =  0.644
params_H2_DD[1, 2, 3] = -0.170
params_H2_DD[1, 2, 4] = -0.199
params_H2_DD[1, 2, 5] =  0.065
params_H2_DD[2, 2, 0] = -2.013
params_H2_DD[2, 2, 1] =  0.828
params_H2_DD[2, 2, 2] =  0.540
params_H2_DD[2, 2, 3] =  0.143
params_H2_DD[2, 2, 4] = -0.056
params_H2_DD[2, 2, 5] =  0.082
params_H2_DD[2, 2, 6] =  0.744
params_H2_DD[2, 2, 7] =  0.051
params_H2_DD[2, 2, 8] = -0.099


PARAMS = {
    (0, "A", "A"): params_H0_AA,
    (0, "B", "B"): params_H0_BB,
    (0, "C", "C"): params_H0_CC,
    (0, "D", "D"): params_H0_DD,

    (1, "B", "A"): params_H1_BA,
    (1, "D", "C"): params_H1_DC,

    (2, "A", "A"): params_H2_AA,
    (2, "B", "B"): params_H2_BB,
    (2, "C", "C"): params_H2_CC,
    (2, "D", "D"): params_H2_DD,

    (3, "D", "C"): params_H3_DC,
}
# --8<-- [end:fitted-values]

# --8<-- [start:valid-neighbors]
VALID = {
    0: {("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")},
    1: {("B", "A"), ("D", "C")},
    2: {("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")},
    3: {("D", "C")},
}
# --8<-- [end:valid-neighbors]