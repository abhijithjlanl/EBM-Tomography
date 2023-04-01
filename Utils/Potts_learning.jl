using StatsBase
using Distributions
using JuMP
using Ipopt
using DelimitedFiles
using Combinatorics: with_replacement_combinations

import LinearAlgebra
import LinearAlgebra: diag
import Statistics: mean





mutable struct FactorGraph{}
    order_list::Array{Int64,1}
    variable_count::Int
    n_alphabets::Int
    terms::Dict{Tuple,Array} # TODO, would be nice to have a stronger tuple type
    #TODO In the sanity check make sure that an kth order hyper edge is mapped to  a k rank tensor
    #variable_names::Union{Vector{String}, Nothing}
    #FactorGraph(a,b,c,d,e) = check_model_data(a,b,c,d,e) ? new(a,b,c,d,e) : error("generic init problem")
end







function  optimizer( raw_samples::Array{}, q::Int, order_list::Array{Int64,1}, learn_vertices=[])
    """
    Function to learn a q alphabet graphical model with interactions of any order. arxiv:1902.00600

    q =  number of alphabets in the Potts model
    raw_samples = one dimesional array of samples. The raw samples not the weights.
    order_list =  list of the order of interactions to consider. Useful if you want your model to exculde magnetic fields etc.

    """
    n =  length(raw_samples[1])
    n_samples =  length(raw_samples)
    sample_hist =  countmap(raw_samples)
    n_config =  length(sample_hist)
    a =  1.0- (1.0/q)
    b = - 1.0/q
    if learn_vertices == []
        vert =  1:n
        else
        vert = learn_vertices
    end
    J_soln = Dict{}()

    for I in vert
    #for I in 1
        model = Model(with_optimizer(Ipopt.Optimizer, max_cpu_time=60.0, print_level=0))

       #setting up the variables for JuMP
        T = Dict{Int,Any}()
        """
        T is a 3 level russian doll of dicts. T[i][j][k] gives a variable for
        the i-th order interaction acting on the vertices given in j  and the
        alphabets given in k. j is a sorted tuple of size i
        """

        for i in order_list

            A =  permutations(Array(1:n),i )
            Q = permutations(Array(1:q), i , asymmetric=true)
            K = Dict{Tuple,Any}()

            for t in A
                if I in t
                L = Dict{Tuple,Any}()

                for k in Q

                      L[k] =  @variable(model, base_name = "T[$i][$t][$k]", lower_bound =  -500, upper_bound = 500)


                end
            K[t] = copy(L)
                end #For the if
            end
        T[i] = copy(K)
        end

      #Constructing the non-linear objective function
        @variable(model, aux[i=1:n_config] ) #auxilary variables are needed to make sure that JuMP doesn't complain
        lin_exponent = []
        emp_prob = []



        for (S, N_S) in sample_hist
            Terms = []

            for r  in order_list
                alphabet_keys =  permutations(Array(1:q), r, asymmetric=true)
                for K in keys(T[r])

                    if I in K
                        #println("K, I", K, I)
                        clr =  Tuple(S[collect(K)])


                        push!(Terms,  sum(  -1*a^(sum(clr.==c))*b^(r -  sum(clr.==c))*  T[r][K][c] for c in alphabet_keys   ) )



                    end




                end
            end
            push!(lin_exponent, sum(Terms))
            push!(emp_prob, N_S/n_samples)
            #println("S = ", S)

            #println("H = ", sum(Terms))


        end

        [@constraint(model, aux[i] == lin_exponent[i]) for i in 1:n_config ]
        @NLexpression(model, nlexp[i=1:n_config], exp(aux[i]))
        @NLobjective(model, Min, sum( emp_prob[i]*nlexp[i] for i in 1:n_config ))
        JuMP.optimize!(model)
        J_soln[I] =  deepcopy(T)

    end




return J_soln
end




function  optimizer_graph(raw_samples::Array{}, q::Int, order_list::Array{Int64,1}, edge_list::Array{Tuple,1}, learn_vertices=[])
    """
    Function to learn a q alphabet graphical model with interactions of any order. arxiv:1902.00600
    In this case the edges in the graph are known as prior inforamtion. edge_list is an array of tuples
    q =  number of alphabets in the Potts model
    raw_samples = one dimesional array of samples. The raw samples not the weights.
    order_list =  list of the order of interactions to consider. Useful if you want your model to exculde magnetic fields etc.
    """
    n =  length(raw_samples[1])
    n_samples =  length(raw_samples)
    sample_hist =  countmap(raw_samples)
    n_config =  length(sample_hist)
    a =  1.0- (1.0/q)
    b = - 1.0/q
    #if learn_vertices == []
    #    vert =  Array[1:n]
    #    else
    #    vert = learn_vertices
    #end
    J_soln = Dict{}()

    for I in 1:n
    #for I in 1
        model = Model(with_optimizer(Ipopt.Optimizer, max_cpu_time=60.0, print_level=3))

       #setting up the variables for JuMP
        T = Dict{Int,Any}()
        """
        T is a 3 level russian doll of dicts. T[i][j][k] gives a variable for
        the i-th order interaction acting on the vertices given in j  and the
        alphabets given in k. j is a sorted tuple of size i
        """

        for i in order_list
            A =  [e for  e in  edge_list if length(e)==i]
            Q = permutations(Array(1:q), i , asymmetric=true)
            K = Dict{Tuple,Any}()

            for t in A
                if I in t
                L = Dict{Tuple,Any}()

                for k in Q

                      L[k] =  @variable(model, base_name = "T[$i][$t][$k]", lower_bound =  -500, upper_bound = 500)


                end
            K[t] = copy(L)
                end #For the if
            end
        T[i] = copy(K)
        end

      #Constructing the non-linear objective function
        @variable(model, aux[i=1:n_config] ) #auxilary variables are needed to make sure that JuMP doesn't complain
        lin_exponent = []
        emp_prob = []



        for (S, N_S) in sample_hist
            Terms = []

            for r  in order_list
                alphabet_keys =  permutations(Array(1:q), r, asymmetric=true)
                for K in keys(T[r])

                    if I in K
                        #println("K, I", K, I)
                        clr =  Tuple(S[collect(K)])


                        push!(Terms,  sum(  -1*a^(sum(clr.==c))*b^(r -  sum(clr.==c))*  T[r][K][c] for c in alphabet_keys   ) )



                    end




                end
            end
            push!(lin_exponent, sum(Terms))
            push!(emp_prob, N_S/n_samples)
            #println("S = ", S)

            #println("H = ", sum(Terms))


        end

        [@constraint(model, aux[i] == lin_exponent[i]) for i in 1:n_config ]
        @NLexpression(model, nlexp[i=1:n_config], exp(aux[i]))
        @NLobjective(model, Min, sum( emp_prob[i]*nlexp[i] for i in 1:n_config ))
        JuMP.optimize!(model)
        J_soln[I] =  deepcopy(T)

    end




return J_soln
end










function  optimizer_symmetric( raw_samples::Array{}, q::Int, order_list::Array{Int64,1}, learn_vertices=[])
    """
    Here we assume that the probability distribution is fully permuation symmetric
    Function to learn a q alphabet graphical model with interactions of any order. arxiv:1902.00600

    q =  number of alphabets in the Potts model
    raw_samples = one dimesional array of samples. The raw samples not the weights.
    order_list =  list of the order of interactions to consider. Useful if you want your model to exculde magnetic fields etc.

    """
    n =  length(raw_samples[1])
    n_samples =  length(raw_samples)
    sample_hist =  countmap(raw_samples)
    n_config =  length(sample_hist)
    a =  1.0- (1.0/q)
    b = - 1.0/q
    #if learn_vertices == []
    #    vert =  Array[1:n]
    #    else
    #    vert = learn_vertices
    #end
    J_soln = Dict{}()

    I = 1

        model = Model(with_optimizer(Ipopt.Optimizer, max_cpu_time=60.0, print_level=0))

       #setting up the variables for JuMP
        Ts = Dict{Int,Any}()
        """
        Ts has two two levels. By symmetry there is only one unique interaction terms per order.
        Ts[i][j], i is the order of the intreaction. k is a i-th order combination of the array 1:q
        """

        for i in order_list
        Qs = with_replacement_combinations(1:q, i)
        Ls = Dict{Tuple,Any}()
            for k in Qs
                Ls[Tuple(k)] =  @variable(model, base_name = "T[$i][$k]", lower_bound =  -500, upper_bound = 500)
            end
        Ts[i] = copy(Ls)
        end
########### Now copying the symmetric variables to the old setup ################################

        T = Dict{Int,Any}()
        """
        T is a 3 level russian doll of dicts. T[i][j][k] gives a variable for
        the i-th order interaction acting on the vertices given in j  and the
        alphabets given in k. j is a sorted tuple of size i
        """

        for i in order_list

            A =  permutations(Array(1:n),i )
            Q = permutations(Array(1:q), i , asymmetric=true)
            K = Dict{Tuple,Any}()

            for t in A
                if I in t
                L = Dict{Tuple,Any}()

                for k in Q

                      L[k] =  Ts[i][Tuple(sort([k...]))]


                end
            K[t] = copy(L)
                end #For the if
            end
        T[i] = copy(K)
        end



      #Constructing the non-linear objective function
        @variable(model, aux[i=1:n_config] ) #auxilary variables are needed to make sure that JuMP doesn't complain
        lin_exponent = []
        emp_prob = []



        for (S, N_S) in sample_hist
            Terms = []

            for r  in order_list
                alphabet_keys =  permutations(Array(1:q), r, asymmetric=true)
                for K in keys(T[r])

                    if I in K
                        #println("K, I", K, I)
                        clr =  Tuple(S[collect(K)])


                        push!(Terms,  sum(  -1*a^(sum(clr.==c))*b^(r -  sum(clr.==c))*  T[r][K][c] for c in alphabet_keys   ) )



                    end




                end
            end
            push!(lin_exponent, sum(Terms))
            push!(emp_prob, N_S/n_samples)
            #println("S = ", S)

            #println("H = ", sum(Terms))


        end

        [@constraint(model, aux[i] == lin_exponent[i]) for i in 1:n_config ]
        @NLexpression(model, nlexp[i=1:n_config], exp(aux[i]))
        @NLobjective(model, Min, sum( emp_prob[i]*nlexp[i] for i in 1:n_config ))
        JuMP.optimize!(model)
        J_soln[I] =  deepcopy(T)



for I in 2:n

    T = Dict{Int,Any}()
    """
    T is a 3 level russian doll of dicts. T[i][j][k] gives a variable for
    the i-th order interaction acting on the vertices given in j  and the
    alphabets given in k. j is a sorted tuple of size i
    """

    for i in order_list

        A =  permutations(Array(1:n),i )
        Q = permutations(Array(1:q), i , asymmetric=true)
        K = Dict{Tuple,Any}()

        for t in A
            if I in t
            L = Dict{Tuple,Any}()

            for k in Q

                  L[k] =  Ts[i][Tuple(sort([k...]))]


            end
        K[t] = copy(L)
            end #For the if
        end
    T[i] = copy(K)
    end

    J_soln[I] =  deepcopy(T)

end






return J_soln
end



permutations(items, order::Int; asymmetric::Bool = false) = sort(permutations([], items, order, asymmetric))

function permutations(partial_perm::Array{Any,1}, items, order::Int, asymmetric::Bool)
    """
    All possible permutations of a given size.
    If asymmetric is false, then it returns combinations of items of the given order
    If asymmetric is true it returns all possible tuples of the size given by order from items
    """
    if order == 0
        return [tuple(partial_perm...)]
    else
        perms = []
        for item in items
            if !asymmetric && length(partial_perm) > 0
                if partial_perm[end] >= item
                    continue
                end
            end
            perm = permutations(vcat(partial_perm, item), items, order-1, asymmetric)
            append!(perms, perm)
        end
        return perms
    end
end



function extract_couplings_graph(J, n::Int, q::Int, order_list::Array{Int64,1}, edge_list)
   """
    Takes the output from the optimizer and turns it into a factor graph
    Given prior information about the graph


    """

    Terms = Dict{}()
    for r in order_list
        edges =  [e for  e in  edge_list if length(e)==r]
        colours = permutations(1:q, r, asymmetric=true)
        dims =  q*ones(Int64, r)
        for e in edges
            Terms[e] = zeros(dims...)

            for c in colours
                counter = 0
                val = 0.0
                for I in e
                 val += JuMP.value(J[I][r][e][c])
                 #println(c, JuMP.value(J[I][r][e][c]))

                 counter += 1   #Always equal to r.
                end
                Terms[e][c...] = val/counter
            end

        end

    end

    return FactorGraph(order_list, n, q, Terms)

end







function extract_couplings(J, n::Int, q::Int, order_list::Array{Int64,1})
   """
    Takes the output from the optimizer and turns it into a factor graph


    """

    Terms = Dict{}()
    for r in order_list
        edges =  permutations(1:n, r)
        colours = permutations(1:q, r, asymmetric=true)
        dims =  q*ones(Int64, r)
        for e in edges
            Terms[e] = zeros(dims...)

            for c in colours
                counter = 0
                val = 0.0
                for I in e
                 val += JuMP.value(J[I][r][e][c])
                 #println(c, JuMP.value(J[I][r][e][c]))

                 counter += 1   #Always equal to r.
                end
                Terms[e][c...] = val/counter
            end

        end

    end

    return FactorGraph(order_list, n, q, Terms)

end




function  gauge_fixing(H::FactorGraph)
    """
    Refer 1902.00600 Eq(57) and  Eq(40)

    """

    n =  H.variable_count
    q =  H.n_alphabets
    b =  -1.0/q
    a = 1.0 -(1.0/q)
    new_terms= Dict()

        for (edge, theta) in H.terms

            r = length(edge)
            new_terms[edge] =  zeros(  size(theta))
            indices = permutations( Array(1:q), r, asymmetric=true )
            for s in indices

                ct =  Tuple(s)

                [new_terms[edge][s...] += a^(sum( ct.==c )) * b^(r -  sum(ct.==c)) *theta[c...] for c in indices]



            end

        end





    return FactorGraph(H.order_list, n, q, new_terms)

    end






function raw_sampler_potts(H::FactorGraph, n_samples::Int, centered::Bool)
    """
    Given the FactorGraph, return samples according to its Gibbs distribution

    """

    n = H.variable_count
    q = H.n_alphabets
    n_config = q^n
    configs = [ digits(i,base=q, pad=n) .+ 1 for i = 0:n_config-1]
    weights = [ exp(Energy_Potts(K, H, centered)) for K in configs ]
    #print(configs, weights/sum(weights))
    raw_samples =  wsample(configs, weights, n_samples)
    return raw_samples

   end

function Energy_Potts(state::Array{Int64,  1},H::FactorGraph, cent::Bool)
    """
    Given a state and a FactorGraph, return its energy
    """
    q =  H.n_alphabets
    b =  -1.0/q
    a = 1.0 -(1.0/q)
    if !(cent)
     E = 0.0

     for (e, theta) in H.terms
        edge=Any[]

        [push!(edge, state[j] ) for j in e]

        E += theta[edge...]
     end
    return E
    end

    if cent
        E = 0.0

     for (e, theta) in H.terms
        clrs=Any[]
        r = length(e) #order of interaction
        alphabet_keys =  permutations(Array(1:q), r, asymmetric=true) #No need to generate this everytime

        [push!(clrs, state[j] ) for j in e]
        ct =  Tuple(clrs)

        [E += a^(sum( ct.==c )) * b^(r -  sum(ct.==c)) *theta[c...] for c in alphabet_keys]
     end
    return E
    end

end


function TVD(truth::Dict{}, est::Dict{}, n_samples::Int)
    """
    Total variation distance between two distributions.
    """
    s = 0.0
    for (k ,v) in est
        if haskey(truth, k)
            s+= abs( v -  truth[k])
        else
            s+= v
        end
    end

    for (k,v) in truth
        if !haskey(est, k)
            s+=v
        end
    end





    return s/(2*n_samples)

end



function conditional_energy(u::Int, state::Array{Int64,  1},H::FactorGraph, cent::Bool)
"""
Given a state and a FactorGraph, return its  local energy at u
"""
q =  H.n_alphabets
b =  -1.0/q
a = 1.0 -(1.0/q)
if !(cent)
    E = 0.0

    for (e, theta) in H.terms
        if u in e
            edge=Any[]

            [push!(edge, state[j] ) for j in e]

            E += theta[edge...]
        end
    end
    return E
end

if cent
    E = 0.0

    for (e, theta) in H.terms
        if u in e
            clrs=Any[]
            r = length(e) #order of interaction
            alphabet_keys =  permutations(Array(1:q), r, asymmetric=true) #No need to generate this everytime

            [push!(clrs, state[j] ) for j in e]
            ct =  Tuple(clrs)

            [E += a^(sum( ct.==c )) * b^(r -  sum(ct.==c)) *theta[c...] for c in alphabet_keys]
        end
    end
    return E
    end

end


function Gibbs_sampler_Factor_graph(n_variables::Int,q::Int ,n_samples::Int, burn_in::Int, H::FactorGraph)

    #state =  convert.(Int, ones(n_variables) )
    state =  rand(1:q, n_variables)
    phi(u, a) = (u==a) ? 1-1/q : -1/q
    phi_vec(s) = [phi(s,a) for a in 1:q]
    prob_lookup = Dict{Any,Any}()

    for t in 1:burn_in
            for u in 1:n_variables
                indices = deleteat!( collect( 1:n_variables ),  u )
                #spins_u =  onehotbatch( state[indices] , 1:q)
                spins_u = state[indices]
                #if (spins_u,u) in keys(prob_lookup)
            #        wts =  prob_lookup[(spins_u,u)]
        #        else

                    wts =    [exp( conditional_energy(u, insert!(spins_u[1:n-1],u,v), H, false )   ) for v in 1:q]

        #            prob_lookup[(spins_u[:],u)] =  wts[:]
    #            end
                if t%(burn_in/100) < 100/burn_in
                    a =  (t/(burn_in/100))


                    #print(" Burnin- $a % complete \r")
                end

                state[u] =  wsample( 1:q , wts )
            end
    end
    state_list = []
    for t in 1:n_samples
            for u in 1:n_variables
                indices = deleteat!( collect( 1:n_variables ),  u )
                #spins_u =  onehotbatch( state[indices] , 1:q)
                spins_u = state[indices]
                #if (spins_u,u) in keys(prob_lookup)
                #    wts =  prob_lookup[(spins_u,u)]
                #else
                    wts =    exp.([ conditional_energy(u,  insert!(spins_u[1:n-1],u,v), H, false )    for v in 1:q])
                #    prob_lookup[(spins_u[:],u)] =  wts[:]
                #end

                if t%(2*n_samples/100) < 100/n_samples
                    a =  (t/(n_samples/100))


                    #print(" $a % complete \r")

                end

                state[u] =  wsample( 1:q , wts )
            end
            push!(state_list, copy(state))
    end
    return state_list
end
