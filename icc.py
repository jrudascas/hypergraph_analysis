from rpy2.robjects import DataFrame, FloatVector, IntVector
from rpy2.robjects.packages import importr
from math import isclose

groups = [1, 2, 0, 1, 1, 3, 3, 2, 3, 8, 1, 4, 6, 4, 3,
          3, 6, 5, 5, 6, 7, 5, 6, 2, 8, 7, 7, 9, 9, 9, 9, 8]
values = [1, 2, 0, 1, 1, 3, 3, 2, 3, 8, 1, 4, 6, 4, 3,
          3, 6, 5, 5, 6, 7, 5, 6, 2, 8, 7, 7, 9, 9, 9, 9, 8]

r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
                "values": FloatVector(values)})
icc_res = r_icc.ICCbare(x="groups", y="values", data=df)
icc_val = icc_res[0] # icc_val now holds the icc value

# check whether icc value equals reference value
print(icc_val)
print(isclose(icc_val, 0.728, abs_tol=0.001))