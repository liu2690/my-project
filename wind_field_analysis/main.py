from read_wind import read_files
from plot_field import plot_field



root="/home/liuxy/0418"



# ======================
# 1 流向
# ======================


folder=root+"/流向wind-height5m"


Ux,files=read_files(
    folder,
    include=[
        "streamwise",
        "fux5m",
        "13-14"
    ]
)


print(
    "流向:",
    Ux.shape
)


plot_field(
    Ux,
    "x position",
    "Streamwise velocity field",
    "streamwise.png"
)



# ======================
# 2 展向 z=5m
# ======================


folder=root+"/展向"


Uy,files=read_files(
    folder,
    include=[
        "sp",
        "fux5m",
        "13-14"
    ],
    exclude=[
        "fuy",
        "fuz",
        "fte"
    ]
)


print(
    "展向:",
    Uy.shape
)


plot_field(
    Uy,
    "y position",
    "Spanwise velocity field z=5m",
    "spanwise_z5.png"
)



# ======================
# 3 垂向
# ======================


folder=root+"/垂向"


Uz,files=read_files(
    folder,
    include=[
        "fux",
        "13-14"
    ],
    exclude=[
        "fuy",
        "fuz",
        "fte"
    ]
)


print(
    "垂向:",
    Uz.shape
)


plot_field(
    Uz,
    "height z",
    "Vertical velocity field",
    "vertical.png"
)