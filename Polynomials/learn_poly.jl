using GraphicalModelLearning, LaTeXStrings
include("QMC.jl")
function dict_2_array(A,n)
    S = Int.(zeros(length(A),n+1))
    f = 1
    for (k,v) in A

        S[f,2:n+1] .= Int.(k)
        S[f,1] = v
        f += 1
    end

    return S

end

function ZZ_plot(Corr, Corr_learned, n; name = "NN")
    #=
    matplotlib.style.use("default")
    PyPlot.rc("text", usetex = false)
    PyPlot.rc("font", family="serif", size = 5)
    PyPlot.rc("figure", figsize=(7/4, 7/5))
    PyPlot.rc("axes", linewidth=0.05, facecolor="w",edgecolor="grey")
    =#
#    xticks(1:4:n)
    plot(1:n, Corr,label="QMC", markersize=1.0, linewidth=0.5)
    scatter!(1:n, Corr_learned,  label="Learned", markersize = 2.0,linewidth=0.5)
   # legend(title =  L"$\langle Z_{10} Z_j \rangle$")
    xlabel!("Site index(j)")
    #grid()
#    tight_layout()
#    savefig("two_site_QMC_" * name * ".pdf", dpi = 300)

end



n_samples =10^5
q = 2
n_qubits =  20
n_time_steps =  10 #Has to be atleast 2
beta =  2
Jz =  -1
Jx = 1
J_spatial =  -beta*Jz/n_time_steps
J_temporal = -0.5*log(tanh(beta*Jx/n_time_steps))


n =  n_qubits*n_time_steps

println(J_spatial,", " , J_temporal,", " ,beta*Jx/n_time_steps)
s = world_line_MCMC(n_samples,n_qubits,n_time_steps,J_spatial,J_temporal,100000)
samples = [a[:,10] for a in s] ##One slice is used for samples
C =  countmap(samples)
Corr_ZZ =  zeros(n_qubits)
for l in 1:n_qubits
    Corr_ZZ[l] = sum( s[1]*s[l] for s in samples)/n_samples
end


##Learning EBM from samples

S = dict_2_array(countmap(samples), n_qubits)
E = learn( S, ISE(),EntropicDescent(1e4, 0.1, 30, 1e-8) )
S_new = GraphicalModelLearning.sample(FactorGraph(E), 4*10^6, Glauber(ones(n_qubits)))

##Comparing ZZ values
Corr_ZZ_poly =  zeros(n_qubits)
for l in 1:n_qubits
    Corr_ZZ_poly[l] = sum( v*s[1]*s[l] for (s,v) in S_new)/(4*10^6)
end


ZZ_plot(Corr_ZZ, Corr_ZZ_poly,n_qubits; name = "poly")



