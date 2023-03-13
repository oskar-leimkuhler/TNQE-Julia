import KrylovKit
using ITensors


# Corrected DMRG for overlap minimization:

function dmrg_c(H::MPO, Ms::Vector{MPS}, psi0::MPS, sweeps::Sweeps; kwargs...)
  ITensors.check_hascommoninds(siteinds, H, psi0)
  ITensors.check_hascommoninds(siteinds, H, psi0')
  for M in Ms
    ITensors.check_hascommoninds(siteinds, M, psi0)
  end
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  Ms .= ITensors.permute.(Ms, Ref((linkind, siteinds, linkind)))
  weight = get(kwargs, :weight, 1.0)
  if weight <= 0.0
    error(
      "weight parameter should be > 0.0 in call to excited-state dmrg (value passed was weight=$weight)",
    )
  end
  PMM = ProjMPO_MPS_c(H, Ms; weight=weight)
  return dmrg_c(PMM, psi0, sweeps; kwargs...)
end


function dmrg_c(x1, x2, psi0::MPS; kwargs...)
  return dmrg_c(x1, x2, psi0, ITensors._dmrg_sweeps(; kwargs...); kwargs...)
end


# DMRG with custom cost functions:

function dmrg_custom(H::MPO, Ms::Vector{MPS}, coeffs::Vector{Float64}, Mo::Vector{MPS}, weight, psi0::MPS, sweeps::Sweeps; kwargs...)
  ITensors.check_hascommoninds(siteinds, H, psi0)
  ITensors.check_hascommoninds(siteinds, H, psi0')
  for M in Ms
    ITensors.check_hascommoninds(siteinds, M, psi0)
  end
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  Ms .= ITensors.permute.(Ms, Ref((linkind, siteinds, linkind)))
  Mo .= ITensors.permute.(Mo, Ref((linkind, siteinds, linkind)))
  PMM = ProjCostFunc(H, Ms, coeffs, Mo, weight)
  return dmrg_c(PMM, psi0, sweeps; kwargs...)
end

function dmrg_custom(x1, x2, psi0::MPS; kwargs...)
  return dmrg_custom(x1, x2, psi0, ITensors._dmrg_sweeps(; kwargs...); kwargs...)
end





#### Just a copy of the standard DMRG script with the correct scope for the new functions: ####


function dmrg_c(PH, psi0::MPS, sweeps::Sweeps; kwargs...)
  if length(psi0) == 1
    error(
      "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
  end

  ITensors.ITensors.@debug_check begin
    # Debug level checks
    # Enable with ITensors.enable_debug_checks()
    checkflux(psi0)
    checkflux(PH)
  end

  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
  svd_alg::String = get(kwargs, :svd_alg, "divide_and_conquer")
  obs = get(kwargs, :observer, NoObserver())
  outputlevel::Int = get(kwargs, :outputlevel, 1)

  write_when_maxdim_exceeds::Union{Int,Nothing} = get(
    kwargs, :write_when_maxdim_exceeds, nothing
  )
  write_path = get(kwargs, :write_path, tempdir())

  # eigsolve kwargs
  eigsolve_tol::Number = get(kwargs, :eigsolve_tol, 1e-14)
  eigsolve_krylovdim::Int = get(kwargs, :eigsolve_krylovdim, 3)
  eigsolve_maxiter::Int = get(kwargs, :eigsolve_maxiter, 1)
  eigsolve_verbosity::Int = get(kwargs, :eigsolve_verbosity, 0)

  ishermitian::Bool = get(kwargs, :ishermitian, true)

  # TODO: add support for targeting other states with DMRG
  # (such as the state with the largest eigenvalue)
  # get(kwargs, :eigsolve_which_eigenvalue, :SR)
  eigsolve_which_eigenvalue::Symbol = :SR

  # TODO: use this as preferred syntax for passing arguments
  # to eigsolve
  #default_eigsolve_args = (tol = 1e-14, krylovdim = 3, maxiter = 1,
  #                         verbosity = 0, ishermitian = true,
  #                         which_eigenvalue = :SR)
  #eigsolve = get(kwargs, :eigsolve, default_eigsolve_args)

  # Keyword argument deprecations
  if haskey(kwargs, :maxiter)
    error("""maxiter keyword has been replaced by eigsolve_krylovdim.
             Note: compared to the C++ version of ITensor,
             setting eigsolve_krylovdim 3 is the same as setting
             a maxiter of 2.""")
  end

  if haskey(kwargs, :errgoal)
    error("errgoal keyword has been replaced by eigsolve_tol.")
  end

  if haskey(kwargs, :quiet)
    error("quiet keyword has been replaced by outputlevel")
  end

  psi = ITensors.copy(psi0)
  N = length(psi)

  if !isortho(psi) || ITensors.orthocenter(psi) != 1
    orthogonalize!(psi, 1)
  end
  @assert isortho(psi) && ITensors.orthocenter(psi) == 1

  position!(PH, psi, 1)
  energy = 0.0

  for sw in 1:nsweep(sweeps)
    sw_time = @elapsed begin
      maxtruncerr = 0.0

      if !isnothing(write_when_maxdim_exceeds) &&
        maxdim(sweeps, sw) > write_when_maxdim_exceeds
        if outputlevel >= 2
          println(
            "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw)), writing environment tensors to disk",
          )
        end
        PH = disk(PH; path=write_path)
      end

      for (b, ha) in sweepnext(N)
        ITensors.@debug_check begin
          checkflux(psi)
          checkflux(PH)
        end

        ITensors.@timeit_debug timer "dmrg: position!" begin
          position!(PH, psi, b)
        end

        ITensors.@debug_check begin
          checkflux(psi)
          checkflux(PH)
        end

        ITensors.@timeit_debug timer "dmrg: psi[b]*psi[b+1]" begin
          phi = psi[b] * psi[b + 1]
        end

        ITensors.@timeit_debug timer "dmrg: eigsolve" begin
          vals, vecs = KrylovKit.eigsolve(
            PH,
            phi,
            1,
            eigsolve_which_eigenvalue;
            ishermitian=ishermitian,
            tol=eigsolve_tol,
            krylovdim=eigsolve_krylovdim,
            maxiter=eigsolve_maxiter,
          )
        end

        energy = vals[1]
        phi::ITensor = vecs[1]

        ortho = ha == 1 ? "left" : "right"

        drho = nothing
        if noise(sweeps, sw) > 0.0
          ITensors.@timeit_debug timer "dmrg: noiseterm" begin
            # Use noise term when determining new MPS basis
            drho = noise(sweeps, sw) * noiseterm(PH, phi, ortho)
          end
        end

        ITensors.@debug_check begin
          checkflux(phi)
        end

        ITensors.@timeit_debug timer "dmrg: replacebond!" begin
          spec = replacebond!(
            psi,
            b,
            phi;
            maxdim=maxdim(sweeps, sw),
            mindim=mindim(sweeps, sw),
            cutoff=cutoff(sweeps, sw),
            eigen_perturbation=drho,
            ortho=ortho,
            normalize=true,
            which_decomp=which_decomp,
            svd_alg=svd_alg,
          )
        end
        maxtruncerr = max(maxtruncerr, spec.truncerr)

        ITensors.@debug_check begin
          checkflux(psi)
          checkflux(PH)
        end

        if outputlevel >= 2
          ITensors.@printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
          ITensors.@printf(
            "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
            cutoff(sweeps, sw),
            maxdim(sweeps, sw),
            mindim(sweeps, sw)
          )
          ITensors.@printf(
            "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
          )
          flush(stdout)
        end

        sweep_is_done = (b == 1 && ha == 2)
        measure!(
          obs;
          energy=energy,
          psi=psi,
          bond=b,
          sweep=sw,
          half_sweep=ha,
          spec=spec,
          outputlevel=outputlevel,
          sweep_is_done=sweep_is_done,
        )
      end
    end
    if outputlevel >= 1
      ITensors.@printf(
        "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
        sw,
        energy,
        maxlinkdim(psi),
        maxtruncerr,
        sw_time
      )
      flush(stdout)
    end
    isdone = checkdone!(obs; energy=energy, psi=psi, sweep=sw, outputlevel=outputlevel)
    isdone && break
  end
  return (energy, psi)
end