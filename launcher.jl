include("load_packages.jl")
include("paper_computations.jl")
#using ArgParse

#s = ArgParseSettings()

#TODO: add parser to allow custom inputs from the command line

n_variables = 2
n_terms = [20, 50, 100] #, 200, 350, 500, 800, 1000]
n_samples = 2

t1 = time()
computations(n_variables, n_terms, n_samples) 
t2 = time()

println(t2-t1)

