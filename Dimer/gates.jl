#Oscillators gate

function locHamOsc(frequency::Float64,sysIdx::Index{Int64}) 
   return -1.0im * frequency * (op("N⋅",sysIdx) - op("⋅N",sysIdx)) 
end

function locDissipatorOsc(frequency::Float64,temperature::Float64,sysIdx::Index{Int64})
   if(temperature == 0.)
      return op("A⋅ * ⋅Adag",sysIdx)-0.5*op("N⋅",sysIdx) - 0.5*op("⋅N",sysIdx) 
   else
      avg = 1/expm1(frequency/temperature) 
   return (1+avg) * (op("A⋅ * ⋅Adag",sysIdx)-0.5*op("N⋅",sysIdx) - 0.5*op("⋅N",sysIdx) ) +
   avg * (op("Adag⋅ * ⋅A",sysIdx)-0.5*op( "A⋅ * Adag⋅",sysIdx) - 0.5*op("⋅Adag * ⋅A",sysIdx) ) 
   end
end

function getLocOscGates(dt::Float64, freqs::Vector{Float64}, 
   temps::Vector{Float64},damps::Vector{Float64}, sys::Vector{Index{Int64}})
   ll = length(sys)
   gates = [
      exp(dt * (
      locHamOsc(freqs[j],sys[j]) + 
      damps[j] * locDissipatorOsc(freqs[j],temps[j],sys[j])
      )) 
      for j in 1:ll]
  
   return gates
   
end

#Interactions gates

function getEVIntGates(dt::Float64, coups::Vector{Matrix{Float64}}, sys::Vector{Index{Int64}},ns::Int64)
   gateMat = Matrix{Vector{ITensor}}(undef,ns,ns)
   for m in 1:ns
      for n in 1:ns
         appo = Array{ITensor}(undef,length(sys))
         for i in 1:length(sys) 
            pippo = (Int(coups[i][1]) == m ? coups[i][2] : 0.) * op("Asum⋅",sys[i])-  
            (Int(coups[i][1]) == n ? coups[i][2] : 0.) * op("⋅Asum",sys[i])
            appo[i] = exp(dt * (-1.0im * pippo))
         end
         gateMat[m,n] = appo
      end
   end
   return gateMat
end

function getEVIntGates2(dt::Float64, coups::Vector{Float64}, sys::Vector{Index{Int64}},ns::Int64)
   
    gateMat = Matrix{Vector{ITensor}}(undef,ns,ns)
 
    appo1 = Vector{ITensor}(undef,length(sys))
    [appo1[i] = exp(-1.0im * 0. * op("Asum⋅",sys[i])) for i in 1:12]
    for i in 1:6
       appo1[i] = exp(dt * (-1.0im * coups[i] * (op("Asum⋅",sys[i]) - op("⋅Asum",sys[i])))) 
    end
 
    gateMat[1,1] = appo1
 
    appo2 = Vector{ITensor}(undef,length(sys))
    [appo2[i] = exp(-1.0im * 0. * op("Asum⋅",sys[i])) for i in 1:12]
    for i in 7:12
       appo2[i] = exp(dt * (-1.0im * coups[i] * (op("Asum⋅",sys[i]) - op("⋅Asum",sys[i])))) 
    end
 
    gateMat[2,2] = appo2
 
    appo12 = Vector{ITensor}(undef,length(sys))
    for i in 1:6
       appo12[i] = exp(dt * (-1.0im * coups[i] * (op("Asum⋅",sys[i])))) 
    end
    for i in 7:12
       appo12[i] = exp(dt * (1.0im * coups[i] * (op("⋅Asum",sys[i])))) 
    end
 
    gateMat[1,2] = appo12
 
    appo21 = Vector{ITensor}(undef,length(sys))
    for i in 7:12
       appo21[i] = exp(dt * (-1.0im * coups[i] * (op("Asum⋅",sys[i])))) 
    end
    for i in 1:6
       appo21[i] = exp(dt * (1.0im * coups[i] * (op("⋅Asum",sys[i])))) 
    end
 
    gateMat[2,1] = appo21
    return gateMat
end

#Electronic gates

function uEl(ham::Matrix{Float64},dt::Float64)
    return exp(-1.0im * dt* ham)
end