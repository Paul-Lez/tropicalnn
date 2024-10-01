include("get_data.jl")
include("summarise_data.jl")

num_samples=25
architectures=[
    [2,6,1],
    [3,5,1],
    [4,4,1],
    [5,3,1],
    [6,2,1],
    [3,2,2,1],
    [3,3,2,1]
    ]
for architecture in architectures
    get_data(architecture,num_samples)
end

summarise_data()