include("get_data.jl")
include("present_data.jl")

architectures_2=[
    [2,2,1],
    [2,4,1],
    [2,5,1],
    [2,6,1],
    [2,7,1],
    [2,8,1],
    [2,9,1],
    [2,10,1],
    [2,2,2,1],
    [2,2,3,1],
    [2,2,4,1],
    [2,2,5,1],
    [2,2,6,1],
    [2,2,2,1],
    [2,3,2,1],
    [2,4,2,1],
    [2,5,2,1],
    [2,6,2,1],
    [2,7,2,1],
    [2,2,7,1],
]

architectures_3=[
    [3,2,1],
    [3,4,1],
    [3,5,1],
    [3,6,1],
    [3,7,1],
    [3,8,1],
    [3,9,1],
    [3,10,1],
    [3,2,2,1],
    [3,2,3,1],
    [3,2,4,1],
    [3,2,5,1],
    [3,2,6,1],
    [3,2,2,1],
    [3,3,2,1],
    [3,4,2,1],
    [3,5,2,1],
    [3,6,2,1],
    [3,7,2,1],
    [3,2,7,1],
]

architectures_4=[
    [4,2,1],
    [4,4,1],
    [4,5,1],
    [4,6,1],
    [4,7,1],
    [4,8,1],
    [4,9,1],
    [4,10,1],
    [4,2,2,1],
    [4,2,3,1],
    [4,2,4,1],
    [4,2,5,1],
    [4,2,6,1],
    [4,2,2,1],
    [4,3,2,1],
    [4,4,2,1],
    [4,5,2,1],
    [4,6,2,1],
    [4,7,2,1],
    [4,2,7,1],
]

architectures_8=[
    [8,2,1],
    [8,4,1],
    [8,5,1],
    [8,6,1],
    [8,7,1],
    [8,8,1],
    [8,9,1],
    [8,10,1],
    [8,2,2,1],
    [8,2,3,1],
    [8,2,4,1],
    [8,2,5,1],
    [8,2,6,1],
    [8,2,2,1],
    [8,3,2,1],
    [8,4,2,1],
    [8,5,2,1],
    [8,6,2,1],
    [8,7,2,1],
    [8,2,7,1],
]

# to obtain the data presented in the paper run these loops
for architecture in architectures_2
    get_data(architecture,10)
end

for architecture in architectures_4
    get_data(architecture,10)
end

get_time([2,4])
get_plots([2,4])