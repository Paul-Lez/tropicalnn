include("algorithms.jl")

num_samples=25
m=5
n=6

e_v=[]
l_v=[]
u_v=[]

e_t=[]
l_t=[]
u_t=[]

for i in 1:num_samples
    A=rand(Normal(0, sqrt(2/n)),m,n)

    t_start=time()
    v=exact_hoff(A)
    t_taken=time()-t_start
    push!(e_v,v)
    push!(e_t,t_taken)

    t_start=time()
    v=lower_hoff(A)
    t_taken=time()-t_start
    push!(l_v,v)
    push!(l_t,t_taken)

    t_start=time()
    v=upper_hoff(A)
    t_taken=time()-t_start
    push!(u_v,v)
    push!(u_t,t_taken)

    println("Sample $i: estimates ("*string(round(l_v[end],digits=3))*", "*string(round(e_v[end],digits=3))*", "*string(round(u_v[end],digits=3))*") times ("*string(round(l_t[end],digits=3))*", "*string(round(e_t[end],digits=3))*", "*string(round(u_t[end],digits=3))*")")
end

lower_tightness_relative=[(e_v[k]-l_v[k])/e_v[k] for k in 1:length(e_v)]
upper_tightness_relative=[(u_v[k]-e_v[k])/e_v[k] for k in 1:length(e_v)]
lower_time_delta_relative=[1-(e_t[k]-l_t[k])/e_t[k] for k in 1:length(e_t)]
upper_time_delta_relvative=[1-(e_t[k]-u_t[k])/e_t[k] for k in 1:length(e_t)]

ltr_avg=round(mean(lower_tightness_relative),digits=5)
ltr_std=round(std(lower_tightness_relative),digits=5)
utr_avg=round(mean(upper_tightness_relative),digits=5)
utr_std=round(std(upper_tightness_relative),digits=5)
ltdr_avg=round(mean(lower_time_delta_relative),digits=5)
ltdr_std=round(std(lower_time_delta_relative),digits=5)
utdr_avg=round(mean(upper_time_delta_relvative),digits=5)
utdr_std=round(std(upper_time_delta_relvative),digits=5)

println("Data summary: ($ltr_avg +/- $ltr_std, $utr_avg +/- $utr_std), times ($ltdr_avg +/- $ltdr_std, $utdr_avg +/- $utdr_std)")