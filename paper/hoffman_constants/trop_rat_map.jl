include("algorithms.jl")

num_samples=25
m_p=2
m_q=3
n=6

e_v=[]
l_v=[]
u_v=[]

e_t=[]
l_t=[]
u_t=[]

for i in 1:num_samples
    ind_e_v=[]
    ind_l_v=[]
    ind_u_v=[]

    ind_e_t=[]
    ind_l_t=[]
    ind_u_t=[]

    A_p=rand(Normal(0, sqrt(2/n)),m_p,n)
    A_q=rand(Normal(0, sqrt(2/n)),m_q,n)
    for i in 1:m_p
        tilde_A_p=A_p-ones(m_p,1)*reshape(A_p[i,:],(1,n))
        for j in 1:m_q
            tilde_A_q=A_q-ones(m_q,1)*reshape(A_q[j,:],(1,n))
            tilde_A=[tilde_A_p;tilde_A_q]

            t_start=time()
            v=exact_hoff(tilde_A)
            t_taken=time()-t_start
            push!(ind_e_v,v)
            push!(ind_e_t,t_taken)

            t_start=time()
            v=lower_hoff(tilde_A)
            t_taken=time()-t_start
            push!(ind_l_v,v)
            push!(ind_l_t,t_taken)

            t_start=time()
            v=upper_hoff(tilde_A)
            t_taken=time()-t_start
            push!(ind_u_v,v)
            push!(ind_u_t,t_taken)
        end
    end
    
    push!(e_v,maximum(ind_e_v))
    push!(e_t,sum(ind_e_t))

    push!(l_v,minimum(ind_l_v))
    push!(l_t,sum(ind_l_t))

    push!(u_v,maximum(ind_u_v))
    push!(u_t,sum(ind_u_t))

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