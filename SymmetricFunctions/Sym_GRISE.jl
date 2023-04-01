using StatsBase
using Distributions
using JuMP
using Ipopt
using DelimitedFiles
using Combinatorics: with_replacement_combinations
import LinearAlgebra
import LinearAlgebra: diag
import Statistics

function countmap_4(sample)
    d = [0,0,0,0]
    for s in sample
        d[s] += 1
    end
    return d
end


function  optimizer_occupationnumber( raw_samples::Array{}; lambda=0.0)
    """
    Here we assume that the probability distribution is fully permuation symmetric
    Function to learn a q alphabet graphical model with interactions of any order. arxiv:1902.00600
    The learning is done in an occupation number basis
    q =  number of alphabets in the Potts model
    raw_samples = one dimesional array of samples. The raw samples not the weights.


    """


    q = 4
    n =  length(raw_samples[1])
    n_samples =  length(raw_samples)
    #sample_hist =  countmap(raw_samples)

    a =  1.0-(1.0/q)
    b = - 1.0/q
    conditional_list = [[] for i in 1:q]

    for s in raw_samples
        push!(conditional_list[s[1]],countmap_4(s[2:n]))
    end
    sample_hists = [countmap(conditional_list[i]) for i in 1:q]
    #println(sample_hists)



    #if learn_vertices == []
    #    vert =  Array[1:n]
    #    else
    #    vert = learn_vertices
    #end
    J_soln = Dict{}()

    I = 5

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_cpu_time", 300.0)
  #  set_optimizer_attribute(model, "print_level", 1)

       #setting up the variables for JuMP

        """
        Ts has two two levels. By symmetry there is only one unique interaction terms per order.
        Ts[i][k], is the albhabet at site 1. k is a choice of  combination of the array 1:q
        """
        T=  [ Dict{}() for i in 1:q]
        Z =  [ Dict{}() for i in 1:q]
        a =  collect(0:n-1)
        A =  fill(a, q)
        Q =  Iterators.product(A...)
        for i in 1:q
            for k in Q
                if sum(k) ==  n -1
                        T[i][[k...]] =  @variable(model, base_name = "T[$i][$k]", lower_bound =  -500, upper_bound = 500)
                        Z[i][[k...]] =  @variable(model, base_name = "Z[$i][$k]", lower_bound =  -500, upper_bound = 500)
                        @constraint(model, Z[i][[k...]]>=  T[i][[k...]]) #z_plus
                        @constraint(model, Z[i][[k...]] >= -T[i][[k...]] ) #z_minus

                end
            end

        end

        Terms = []

        emp_prob = []

        for i in 1:q
            for (S, N_S) in sample_hists[i]
                #println(S, N_S)
                push!(Terms,  -T[i][S] + (1.0/q)*sum( T[j][S]  for j in 1:q))
                push!(emp_prob,N_S/n_samples )

            end
        end
       #  println(Terms[10])

        n_config =  length(emp_prob)
        @variable(model, aux[i=1:n_config] )
        [@constraint(model, aux[i] == Terms[i]) for i in 1:n_config ]

        @NLexpression(model, nlexp[i=1:n_config], exp(aux[i]))
        @NLobjective(model, Min, sum( emp_prob[i]*nlexp[i] for i in 1:n_config ) +
                        lambda * sum( sum( Z[j][L] for L in keys(Z[j])  ) for j in 1:q  )
                    )
        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED



      #Constructing the non-linear objective function

return T
end


function MCMC_symmetric_sampler(n, n_samples, T; burnin=10^6)
    q =4
    state =  convert.(Int, ones(n) )
    phi(u, a) = (u==a) ? 1-1/q : -1/q
    phi_vec(s) = [phi(s,a) for a in 1:q]
    state_list = []

    F=  [ Dict{}() for i in 1:q]
    a =  collect(0:n-1)
    A =  fill(a, q)
    Q =  Iterators.product(A...)
    for i in 1:q

        for k in Q
            if sum(k) ==  n -1

                    F[i][[k...]] =  JuMP.value(T[i][[k...]])
            end
        end

    end
    for t in 1:burnin
        for u in 1:n
            indices = deleteat!( collect( 1:n ),  u )
            k= countmap_4(state[indices])
            wts =  [ exp(F[s][k]) for s in 1:q  ]
            state[u] =  wsample( 1:q , wts )
        end
        if t%(burnin/100) < 100/burnin
            a =  (t/(burnin/100))


            print(" Burnin- $a % complete \r")
        end

    end
    println(" ")
    println("Burnin complete")


    for t in 1:n_samples
        for u in 1:n
            indices = deleteat!( collect( 1:n ),  u )
            k= countmap_4(state[indices])
            #wts =  [ exp(sum(phi_vec(s)[a]*F[a][k] for a in 1:q)) for s in 1:q  ]
            wts =  [ exp(F[s][k]) for s in 1:q  ]
            state[u] =  wsample( 1:q , wts )
        end
        push!(state_list, state[:])
        if t%(n_samples/100) < 100/n_samples
            a =  (t/(n_samples/100))


            print(" $a % complete \r")
        end

    end
    return state_list

end
