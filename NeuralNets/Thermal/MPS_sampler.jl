using ITensors
using StatsBase
using DelimitedFiles
using ProgressBars

prob(x) =  abs(x)^2


function conditionals_from_amplitudes(AMPLITUDE_MPS, q)
    conditional_MPS = [deepcopy(AMPLITUDE_MPS)]
    n =  length(PROB_MPS)
    for i in 1:n-1
        @show "Computing maginal $i"
        A =  deepcopy(PROB_MPS)
        for j in 1:i
            A[j] =  A[j]*delta(siteinds(A)[j])
        end
        B = reduce(*,A[1:i+1])
        C = [A[x] for x in i+2:n]
        #return [B , C...]
        A = MPS([B , C...])
        push!(conditional_MPS, deepcopy(A))
    end
    return conditional_MPS
end
    
function exact_sampler(conditional_MPS, SITEINDS,q, n, n_samples)
    samples =  Int.(zeros(n_samples,  n))
    for t in ProgressBar(1:n_samples)
        samples[t,n] = wsample(1:q, real(Array(conditional_MPS[end][1], siteinds(conditional_MPS[end])[1])))
        for i in n-1:-1:1
            V =  conditional_MPS[i][1]
            for j in 2: n-i+1
                V *=  conditional_MPS[i][j]*state(SITEINDS[j+i-1], samples[t, j+i-1])
            end
#            @show V
            samples[t, i] = wsample(1:q, real(Array(V, inds(V)... ))   )
        end
    end

    return samples

end



