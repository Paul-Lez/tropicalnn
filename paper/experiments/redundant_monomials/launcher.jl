include("get_data.jl")
include("get_plots.jl")

num_samples=25
num_layers=1
network_widths=[2,3,4,5,6]
for network_width in network_widths
    get_data_untrained(network_width,num_layers,num_samples)
    get_data_trained(network_width,num_layers,num_samples)
end

layer_plot(1)