using ITensors
using PseudomodesTTEDOPA
using LinearAlgebra
using Plots
using DataFrames
using CSV
using QuadGK
using ProgressMeter
using LaTeXStrings
include("gates.jl")

#Here we define all the global parameters

global nsites::Int64 = 2; #Number of sites
global nosc::Int64 = 2; #Number of oscilators of the bath
global localDim::Int64 = 3; #Local dimensions of oscillators
global maxBondDim = 8; #Bond dimension 
global step::Int64 = 1000; #Number of step 
global Cutoff::Float64 = 10e-10 #Limit values beyond which to neglect in the weighted sum of MPS

#Useful conversions

global eVtocm=8065.6; 
global unitKcm = 8.617330350*10^(-5) * eVtocm;
global lightspeed = 299792458.;
global time_s::Float64 = 300; 
global unit_cm_fs = 0.00018836515661233488;
global time_cm = time_s * unit_cm_fs; 
global timestep::Float64 = time_cm/step;

#parameters of system and bath

energies = [12400.,12400.]; 
exchange = [0. 200.; 200. 0.];  
freqss = [763.,763.];
coups = [[1 1.], [2 1.]]  
damps = [76.,76.];
temps = [1.,1.]; 
temp = 77. * unitKcm;
temps *= temp; 
hr_factors = [0.133,0.133];
[coups[i][2] *= freqss[i] * sqrt(hr_factors[i]) for i in 1:nsites]


#creating environment MPS
bath = siteinds("HvOsc",nosc,dim=localDim,);
bathMPS=chain(
   [parse_init_state_osc(bath[j],"thermal",frequency=freqss[j],temperature=temps[j]) for j in 1:nosc]... 
)
orthogonalize!(bathMPS,1);
rho_E = bathMPS;

#creating reduced system density matrix
rho_S1 = ComplexF64.(zeros(nsites,nsites));
rho_S1[1,1] = 1.0; #sistem starting in the pure state e1

#total density matrix
rho1 = rho_S1 .* fill(rho_E,nsites,nsites);

#Creating evolution gates
locOscGates = getLocOscGates(timestep,freqss,temps,damps,bath);
intGates = getEVIntGates(timestep,coups,bath,nsites);

elHam = exchange + Diagonal(energies) #Electronic Hamiltonian
evoEl = uEl(elHam, timestep)
evoEldaga = evoEl';

#Coefficients for the weighted sum
C=[
   [
      [ 
         [
            evoEl[m,i]*evoEldaga[j,l]
         for l in 1:nsites]
      for j in 1: nsites]  
   for  i in 1:nsites]   
for m in 1:nsites]

#Time evolution

pop1_1 = ComplexF64[]
pop2_1 = ComplexF64[]
coh12 = ComplexF64[]
coh21 = ComplexF64[]
T = Float64[]

@showprogress 1 "Computing time evolution" for t in 0.:timestep:time_cm
   
   #duplicate current rho
   newRho1 = rho1
   
   #Measure elements of rho
   V = ITensor(1.)
   F = ITensor(1.)
   L = ITensor(1.)
   P = ITensor(1.)

    for j=1:nosc
        V *= (rho1[1,1][j]*state(bath[j],"vId"))
        F *= (rho1[2,2][j]*state(bath[j],"vId"))
        L *= (rho1[1,2][j]*state(bath[j],"vId"))
        P *= (rho1[2,1][j]*state(bath[j],"vId"))
    end

   v = scalar(V)
   f = scalar(F)
   l = scalar(L)
   p = scalar(P)

   push!(pop1_1,v)
   push!(pop2_1,f)
   push!(coh12,l)
   push!(coh21,p)
   push!(T,t)

   #############
   
   #apply H_e
   for m  in 1:nsites
      for l in 1:nsites
        weightOlist = vcat([C[m][i][j][l]* rho1[i,j] for i in 1:nsites, j in 1:nsites]...)
        newRho1[m,l]= add(weightOlist...,cutoff = 10e-18, maxdim=maxBondDim) 
      end
   end 
    
   #apply H_ev
   for m in 1:nsites
     for l in 1:nsites
        newRho1[m,l] = apply(intGates[m,l],newRho1[m,l])
     end
   end
   
   #apply H_v
   for m in 1:nsites
      for l in 1:nsites
        newRho1[m,l] = apply(locOscGates,newRho1[m,l])
      end
   end

   global rho1 = newRho1

end 


#Plot
default(size=(800,500), fontfamily="Computer Modern");
x=range(0,300,length=length(pop1_1));
plt = plot(x,real(pop1_1),label = L"$\rho_{11}$",linewidth = 1.5,lc = "red");
plot!(x,real(pop2_1),label = L"$\rho_{22}$",linewidth = 1.5, lc = "blue");
#plot!(x,broadcast(abs,coh12),label = L"$\rho_{12}$",linewidth = 1.5, alpha = 0.9,lc = "lightgreen"); plot for off-diagonal elements
xlabel!(plt, "Time (fs) ");
savefig("plot.pdf")

#saving data
dati = DataFrame("t" => T, "rho_11" => pop1_1, "rho_22" => pop2_1, "rho_12" => broadcast(abs,coh12), "rho_21" => broadcast(abs,coh21))
CSV.write("data.csv",dati)