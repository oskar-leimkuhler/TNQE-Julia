using ITensors

mutable struct ProjCostFunc
  PH::ITensors.ProjMPO
  pm::Vector{ITensors.ProjMPS}
  coeffs::Vector{Float64}
  po::Vector{ITensors.ProjMPS}
  weight::Float64
end

copy(P::ProjCostFunc) = ProjCostFunc(copy(P.PH), copy.(P.pm), P.coeffs, copy.(P.po), P.weight)

function ProjCostFunc(H::MPO, mpsv::Vector{MPS}, coeffs, mpso::Vector{MPS}, weight)
  return ProjCostFunc(ITensors.ProjMPO(H), [ITensors.ProjMPS(m) for m in mpsv], coeffs, [ITensors.ProjMPS(m) for m in mpso], weight)
end

#ProjCostFunc(H::MPO, Ms::MPS..., weights) = ProjCostFunc(H, [Ms...], weights)

nsite(P::ProjCostFunc) = ITensors.nsite(P.PH)

function set_nsite!(Ps::ProjCostFunc, nsite)
  ITensors.set_nsite!(Ps.PH, nsite)
  for P in Ps.pm
    ITensors.set_nsite!(P, nsite)
  end
  for P in Ps.po
    ITensors.set_nsite!(P, nsite)
  end
  return Ps
end

Base.length(P::ProjCostFunc) = length(P.PH)

function product(P::ProjCostFunc, v::ITensor)::ITensor
  Pv = ITensors.product(P.PH, v)
  for (i,p) in enumerate(P.pm)
    Pv += P.coeffs[i] * ITensors.product(p, v)
  end
  for p in P.po
    Pv += P.weight * absproduct(p, v)
  end
  return Pv
end

function Base.eltype(P::ProjCostFunc)
  elT = eltype(P.PH)
  for p in P.pm
    elT = promote_type(elT, eltype(p))
  end
  return elT
end

(P::ProjCostFunc)(v::ITensor) = product(P, v)

Base.size(P::ProjCostFunc) = size(P.H)

function position!(P::ProjCostFunc, psi::MPS, pos::Int)
  ITensors.position!(P.PH, psi, pos)
  for p in P.pm
    ITensors.position!(p, psi, pos)
  end
  for p in P.po
    ITensors.position!(p, psi, pos)
  end
end

noiseterm(P::ProjCostFunc, phi::ITensor, dir::String) = ITensors.noiseterm(P.PH, phi, dir)