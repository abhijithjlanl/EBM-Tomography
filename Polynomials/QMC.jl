###Simple quantum montecarlo using Path integrals
using IterTools:product
using ProgressBars
using GraphicalModelLearning
using LinearAlgebra, StatsBase, Plots
### Use glauber branch for Entropic GD  pkg> add GraphicalModelLearning#glauber



function pth_order_tensor(r)
    a =  Tuple([2 for i in 1:r])
    M =  zeros(a)
    for i in CartesianIndices(M)
        s = prod(2*[Tuple(i)...] .- 3)
        M[i] = s
    end
    return M
end


cord2num(i,j) =  (i -1)*n_qubits + j

num2cord(k) = ( div(k-1,n_qubits) + 1, (k-1)%n_qubits + 1  )

function int_to_spin(int_representation::Int, spin_number::Int)
    spin = 2*digits(int_representation, base=2, pad=spin_number) .- 1
    return spin
end

function spin_to_int(spins, spin_number::Int )
    return Int(sum( ((1.0 + spins[k] )/2)* 2^(k-1) for k in 1:spin_number ))
end

function Ising_energy_1D(state,J_list, h_list,L)
    E = 0.0
    for i in 1:L
        E += J_list[i]*state[i]*state[i%L + 1]
        #E += h_list[i]*state[i]
    end
    return E + dot(h_list, state)
end

function Ising_energy_1D_test(state,J_list, h_list,L)
    return dot(J_list, state.*circshift(state,-1)) + dot(h_list, state)
end

function Exact_1D_Ising_sampler(L,n_samples, J_list, h_list, population)
    wt_list =  [exp(Ising_energy_1D(s, J_list, h_list, L)) for s in population]
    return wsample(population, wt_list,n_samples)
end




function world_line_MCMC(n_samples, n_qubits, n_time_steps, J_spatial, J_temporal, burnin)
    state =  rand([-1,1], n_qubits, n_time_steps)
    population = [int_to_spin(a,n_time_steps) for a in 0:(2^n_time_steps) -1]
    #println(state)
    state_list = []
    J_list =  J_temporal*ones(n_time_steps)
    for m in ProgressBar(1:burnin)
        for i in 1:n_qubits

	          if 1<i<n_qubits
                h_list = J_spatial*(state[i+1,:] + state[i-1,:])
	          elseif i == 1
                h_list = J_spatial*(state[i+1,:] )
	          elseif i == n_qubits
                h_list = J_spatial*(state[i-1,:] )
            end
            state[i,:] =  Exact_1D_Ising_sampler(n_time_steps, 1, J_list, h_list , population)[1]
        end

    end

     for m in ProgressBar(1:n_samples)
        for i in 1:n_qubits

	          if 1<i<n_qubits
                h_list = J_spatial*(state[i+1,:] + state[i-1,:])
	          elseif i == 1
                h_list = J_spatial*(state[i+1,:] )
	          elseif i == n_qubits
                h_list = J_spatial*(state[i-1,:] )
            end
            state[i,:] =  Exact_1D_Ising_sampler(n_time_steps, 1, J_list, h_list, population )[1]
        end
        push!(state_list, copy(state))
    end
    return state_list
end


