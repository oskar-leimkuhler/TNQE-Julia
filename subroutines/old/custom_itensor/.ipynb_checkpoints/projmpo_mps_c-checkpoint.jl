using ITensors




function absproduct(P::ITensors.ProjMPS, v::ITensor)::ITensor
  if ITensors.nsite(P) != 2
    error("Only two-site ProjMPS currently supported")
  end

  Lpm = dag(prime(P.M[P.lpos + 1], "Link"))
  !isnothing(ITensors.lproj(P)) && (Lpm *= ITensors.lproj(P))

  Rpm = dag(prime(P.M[P.rpos - 1], "Link"))
  !isnothing(rproj(P)) && (Rpm *= ITensors.rproj(P))

  pm = Lpm * Rpm

  pv = scalar(pm * v)

  Mv = abs(pv) * dag(pm)

  return noprime(Mv)
end




mutable struct ProjMPO_MPS_c
  PH::ITensors.ProjMPO
  pm::Vector{ITensors.ProjMPS}
  weight::Float64
end

copy(P::ProjMPO_MPS_c) = ProjMPO_MPS_c(copy(P.PH), copy.(P.pm), P.weight)

function ProjMPO_MPS_c(H::MPO, mpsv::Vector{MPS}; weight=1.0)
  return ProjMPO_MPS_c(ITensors.ProjMPO(H), [ITensors.ProjMPS(m) for m in mpsv], weight)
end

ProjMPO_MPS_c(H::MPO, Ms::MPS...; weight=1.0) = ProjMPO_MPS_c(H, [Ms...], weight)

nsite(P::ProjMPO_MPS_c) = ITensors.nsite(P.PH)

function set_nsite!(Ps::ProjMPO_MPS_c, nsite)
  ITensors.set_nsite!(Ps.PH, nsite)
  for P in Ps.pm
    ITensors.set_nsite!(P, nsite)
  end
  return Ps
end

Base.length(P::ProjMPO_MPS_c) = length(P.PH)

function product(P::ProjMPO_MPS_c, v::ITensor)::ITensor
  Pv = ITensors.product(P.PH, v)
  for p in P.pm
    Pv += P.weight * ITensors.product(p, v)
  end
  return Pv
end

function Base.eltype(P::ProjMPO_MPS_c)
  elT = eltype(P.PH)
  for p in P.pm
    elT = promote_type(elT, eltype(p))
  end
  return elT
end

(P::ProjMPO_MPS_c)(v::ITensor) = product(P, v)

Base.size(P::ProjMPO_MPS_c) = size(P.H)

function position!(P::ProjMPO_MPS_c, psi::MPS, pos::Int)
  ITensors.position!(P.PH, psi, pos)
  for p in P.pm
    ITensors.position!(p, psi, pos)
  end
end

noiseterm(P::ProjMPO_MPS_c, phi::ITensor, dir::String) = ITensors.noiseterm(P.PH, phi, dir)