using Printf
# using PyPlot
using Colors, Plots
using LinearAlgebra
using Statistics
using Makie
using GLMakie
using JLD
using DelimitedFiles
using SparseArrays
using Debugger

function prepare_filter(nelx, nely, nelz, rmin)
    nele = nelx * nely * nelz
    iH = ones(Int64(nele * (2 * (ceil(rmin) - 1) + 1)^2))
    jH = ones(size(iH))
    sH = zeros(size(iH))
    k = 0
    for k1 = 1:nelz
        for i1 = 1:nelx
            for j1 = 1:nely
                e1 = (k1 - 1) * nelx * nely + (i1 - 1) * nely + j1
                for k2 = max(k1 - (ceil(rmin) - 1), 1):min(k1 + (ceil(rmin) - 1), nelz)
                    for i2 = max(i1 - (ceil(rmin) - 1), 1):min(i1 + (ceil(rmin) - 1), nelx)
                        for j2 =
                            max(j1 - (ceil(rmin) - 1), 1):min(j1 + (ceil(rmin) - 1), nely)
                            e2 = (k2 - 1) * nelx * nely + (i2 - 1) * nely + j2
                            k = k + 1
                            if k <= size(iH, 1)
                                iH[k] = e1
                                jH[k] = e2
                                sH[k] = max(
                                    0,
                                    rmin - sqrt((i1 - i2)^2 + (j1 - j2)^2 + (k1 - k2)^2),
                                )
                            else
                                append!(iH, e1)
                                append!(jH, e2)
                                append!(
                                    sH,
                                    max(
                                        0,
                                        rmin -
                                        sqrt((i1 - i2)^2 + (j1 - j2)^2 + (k1 - k2)^2),
                                    ),
                                )
                            end
                        end
                    end
                end
            end
        end
    end
    H = sparse(iH, jH, sH)
    Hs = sum(H, dims = 2)
    # println("H in prepare_filter=$(size(H))");
    H, Hs
end

function brick_stiffnessMatrix()
    # elastic matrix formulation
    nu = 0.3
    D =
        1 ./ ((1 + nu) * (1 - 2 * nu)) * [
            1-nu nu nu 0 0 0
            nu 1-nu nu 0 0 0
            nu nu 1-nu 0 0 0
            0 0 0 (1-2*nu)/2 0 0
            0 0 0 0 (1-2*nu)/2 0
            0 0 0 0 0 (1-2*nu)/2
        ]
    # stiffness matrix formulation
    A = [
        32 6 -8 6 -6 4 3 -6 -10 3 -3 -3 -4 -8
        -48 0 0 -24 24 0 0 0 12 -12 0 12 12 12
    ]
    A = convert(Matrix{BigFloat}, A)
    k = 1 / 144 * A' * [1; nu]

    K1 = [
        k[1] k[2] k[2] k[3] k[5] k[5]
        k[2] k[1] k[2] k[4] k[6] k[7]
        k[2] k[2] k[1] k[4] k[7] k[6]
        k[3] k[4] k[4] k[1] k[8] k[8]
        k[5] k[6] k[7] k[8] k[1] k[2]
        k[5] k[7] k[6] k[8] k[2] k[1]
    ]

    K2 = [
        k[9] k[8] k[12] k[6] k[4] k[7]
        k[8] k[9] k[12] k[5] k[3] k[5]
        k[10] k[10] k[13] k[7] k[4] k[6]
        k[6] k[5] k[11] k[9] k[2] k[10]
        k[4] k[3] k[5] k[2] k[9] k[12]
        k[11] k[4] k[6] k[12] k[10] k[13]
    ]

    K3 = [
        k[6] k[7] k[4] k[9] k[12] k[8]
        k[7] k[6] k[4] k[10] k[13] k[10]
        k[5] k[5] k[3] k[8] k[12] k[9]
        k[9] k[10] k[2] k[6] k[11] k[5]
        k[12] k[13] k[10] k[11] k[6] k[4]
        k[2] k[12] k[9] k[4] k[5] k[3]
    ]

    K4 = [
        k[14] k[11] k[11] k[13] k[10] k[10]
        k[11] k[14] k[11] k[12] k[9] k[8]
        k[11] k[11] k[14] k[12] k[8] k[9]
        k[13] k[12] k[12] k[14] k[7] k[7]
        k[10] k[9] k[8] k[7] k[14] k[11]
        k[10] k[8] k[9] k[7] k[11] k[14]
    ]

    K5 = [
        k[1] k[2] k[8] k[3] k[5] k[4]
        k[2] k[1] k[8] k[4] k[6] k[11]
        k[8] k[8] k[1] k[5] k[11] k[6]
        k[3] k[4] k[5] k[1] k[8] k[2]
        k[5] k[6] k[11] k[8] k[1] k[8]
        k[4] k[11] k[6] k[2] k[8] k[1]
    ]

    K6 = [
        k[14] k[11] k[7] k[13] k[10] k[12]
        k[11] k[14] k[7] k[12] k[9] k[2]
        k[7] k[7] k[14] k[10] k[2] k[9]
        k[13] k[12] k[10] k[14] k[7] k[11]
        k[10] k[9] k[2] k[7] k[14] k[7]
        k[12] k[2] k[9] k[11] k[7] k[14]
    ]

    KE =
        1 / ((nu + 1) * (1 - 2 * nu)) * [
            K1 K2 K3 K4
            K2' K5 K6 K3'
            K3' K6 K5' K2'
            K4 K3 K2 K1'
        ]

    KE = Float64.(round.(KE, sigdigits = 15))
    # strain matrix formulation
    B_1 = [
        -0.044658 0 0 0.044658 0 0 0.16667 0
        0 -0.044658 0 0 -0.16667 0 0 0.16667
        0 0 -0.044658 0 0 -0.16667 0 0
        -0.044658 -0.044658 0 -0.16667 0.044658 0 0.16667 0.16667
        0 -0.044658 -0.044658 0 -0.16667 -0.16667 0 -0.62201
        -0.044658 0 -0.044658 -0.16667 0 0.044658 -0.62201 0
    ]
    B_2 = [
        0 -0.16667 0 0 -0.16667 0 0 0.16667
        0 0 0.044658 0 0 -0.16667 0 0
        -0.62201 0 0 -0.16667 0 0 0.044658 0
        0 0.044658 -0.16667 0 -0.16667 -0.16667 0 -0.62201
        0.16667 0 -0.16667 0.044658 0 0.044658 -0.16667 0
        0.16667 -0.16667 0 -0.16667 0.044658 0 -0.16667 0.16667
    ]
    B_3 = [
        0 0 0.62201 0 0 -0.62201 0 0
        -0.62201 0 0 0.62201 0 0 0.16667 0
        0 0.16667 0 0 0.62201 0 0 0.16667
        0.16667 0 0.62201 0.62201 0 0.16667 -0.62201 0
        0.16667 -0.62201 0 0.62201 0.62201 0 0.16667 0.16667
        0 0.16667 0.62201 0 0.62201 0.16667 0 -0.62201
    ]
    B = [B_1 B_2 B_3]
    KE, B, D
end

function Stress_3D_Sensitivity_Comp(x, nelx, nely, nelz, pl, q, p)
    KE, B, D = brick_stiffnessMatrix()
    #  MATERIAL PROPERTIES
    E0 = 1           # Young's modulus of solid material
    Emin = 1e-9      # Young's modulus of void-like material
    # USER-DEFINED LOAD DOFs
    il, jl, kl = meshgrid(nelx, 0, 0:nelz)                 # Coordinates
    loadnid = kl .* (nelx + 1) .* (nely + 1) + il .* (nely + 1) + (nely + 1 .- jl) # Node IDs
    loaddof = 3 * vec(loadnid) .- 1                             # DOFs

    # USER-DEFINED SUPPORT FIXED DOFs
    iif, jf, kf = meshgrid(0, 0:nely, 0:nelz)                  # Coordinates
    fixednid = kf .* (nelx + 1) .* (nely + 1) .+ iif .* (nely + 1) + (nely + 1 .- jf) # Node IDs (try using broadcast assignment)
    fixeddof = [3 * vec(fixednid); 3 * vec(fixednid) .- 1; 3 * vec(fixednid) .- 2] # DOFs (try using broadcast assignment)
    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    F = sparse(
        loaddof,
        repeat([1], size(loaddof, 1)),
        repeat([-1], size(loaddof, 1)),
        ndof,
        1,
    )
    U = zeros(ndof, 1)
    freedofs = Int64.(setdiff(1:ndof, fixeddof))
    # KE = lk_H8(nu);
    nodegrd = reshape(1:(nely+1)*(nelx+1), nely + 1, nelx + 1)
    nodeids = reshape(nodegrd[1:end-1, 1:end-1], nely * nelx, 1)
    nodeidz = 0:(nely+1)*(nelx+1):(nelz-1)*(nely+1)*(nelx+1)
    nodeids =
        repeat(nodeids, outer = (size(nodeidz, 2), size(nodeidz, 1))) +
        repeat(reshape(nodeidz, 1, :), outer = size(nodeids))
    edofVec = 3 * vec(nodeids) .+ 1
    edofMat =
        repeat(edofVec, outer = (1, 24)) + repeat(
            [0 1 2 3 * nely .+ [3 4 5 0 1 2] -3 -2 -1 3 * (nely + 1) * (nelx + 1) .+
                                                      [0 1 2 3 * nely .+ [3 4 5 0 1 2] -3 -2 -1]],
            outer = (nele, 1),
        )
    iK = Int64.(reshape(kron(edofMat, ones(24, 1))', 24 * 24 * nele))
    jK = Int64.(reshape(kron(edofMat, ones(1, 24))', 24 * 24 * nele))
    sK = reshape(vec(KE) * (Emin .+ vec(x)' .^ pl * (E0 - Emin)), 24 * 24 * nele)
    K = sparse(iK, jK, sK)
    K = (K + K') / 2
    KK = cholesky(K[freedofs, freedofs])
    U[freedofs, :] = KK \ F[freedofs, :]
    # println("typeof(U)=$(typeof(U))");
    MISES = zeros(nele, 1) # von Mises stress vector
    S = zeros(nele, 6)
    # println("size(D)=$(size(D))");
    # println("size(B)=$(size(B))");
    # println("size(U)=$(size(U))");
    for i = 1:nele
        temp = x[i]^q * (D * B * U[edofMat[i, :]])'
        S[i, :] = temp
        MISES[i] = sqrt(
            0.5 * (
                (temp[1] - temp[2])^2 +
                (temp[1] - temp[3])^2 +
                (temp[2] - temp[3])^2 +
                6 * sum(temp[4:6] .^ 2)
            ),
        )
    end
    nele = size(x[:], 1) # total element number
    ndof, _ = size(U)
    DvmDs = zeros(nele, 6)
    dpn_dvms = (sum(MISES .^ p))^(1 / p - 1)
    index_matrix = edofMat'
    pnorm = (sum(MISES .^ p))^(1 / p)
    for i = 1:nele
        DvmDs[i, 1] = 1 / 2 / MISES[i] * (2 * S[i, 1] - S[i, 2] - S[i, 3])
        DvmDs[i, 2] = 1 / 2 / MISES[i] * (2 * S[i, 2] - S[i, 1] - S[i, 3])
        DvmDs[i, 3] = 1 / 2 / MISES[i] * (2 * S[i, 3] - S[i, 1] - S[i, 2])
        DvmDs[i, 4] = 3 / MISES[i] * S[i, 4]
        DvmDs[i, 5] = 3 / MISES[i] * S[i, 5]
        DvmDs[i, 6] = 3 / MISES[i] * S[i, 6]
    end
    # println("size(q)=$(size(q))");
    # println("size(DvmDs[i,:]')=$(size(DvmDs[1,:]'))");
    beta = zeros(nele, 1)
    for i = 1:nele
        u = reshape(U[edofMat[i, :], :]', :, 1)
        a1 = q * (x[i])^(q - 1)
        a1 = a1 * MISES[i]^(p - 1)
        a1 = a1 * DvmDs[i, :]'
        a1 = a1 * D
        a1 = a1 * B
        # println("size(a1)=$(size(a1))");
        # println("size(u)=$(size(u))");
        a1 = a1 * u
        # println("size(a1)=$(size(a1))");
        beta[i] = a1[1]
        # beta[i]=q*(x[i])^(q-1)*MISES[i]^(p-1)*DvmDs[i,:]'*D*B*u;
    end
    T1 = dpn_dvms * beta
    gama = zeros(ndof, 1)
    for i = 1:nele
        index = index_matrix[:, i]
        # gama[index]=gama[index]+x[i]^q*dpn_dvms*B'*D'*DvmDs[i,:]'*MISES[i].^(p-1);
        QQ = x[i]^q * dpn_dvms * B' * D'
        # println("size(QQ)=$(size(QQ))");
        # println("size(DvmDs[i,:]')=$(size(DvmDs[i,:]'))");
        # X=QQ*DvmDs[i,:]';
        X = QQ * DvmDs[i, :]
        gama[index] = gama[index] + X * MISES[i] .^ (p - 1)
    end
    lamda = zeros(ndof, 1)
    lamda[freedofs, :] = K[freedofs, freedofs] \ gama[freedofs, :]
    T2 = zeros(nele, 1)
    for i = 1:nele
        index = index_matrix[:, i]
        T2[i] = -lamda[index]' * pl * x[i]^(pl - 1) * KE * U[index]
    end
    pnorm_sen = T1 + T2


    pnorm, pnorm_sen, MISES
end

function stress_minimize(x, Hs, H, outit)
    nelx = 200
    nely = 60
    nelz = 1
    pl = 3
    q = 0.5
    p = 10
    # println(size(x[:]));
    # println(size(H));
    x[:] = (H * x[:]) ./ Hs
    open("julia/x_in_sm.txt", "w") do io
        writedlm(io, x)
    end
    pnorm, pnorm_sen, MISES = Stress_3D_Sensitivity_Comp(x, nelx, nely, nelz, pl, q, p)
    # figure(1)
    x_plot = reshape(x, nely, nelx, nelz)
    # save("x_plot.jld", "$outit", x_plot);
    # r = dropdims(x_plot[:,:,1], dims=3)
    # figure(1); 
    # f=Figure();
    # ax1 = Axis(f[1, 1]);
    # title = "Current design",);
    # image!(ax1, reverse(x_plot[:,:,1]));
    # display(f);
    Plots.plot(Gray.(x_plot[:, :, 1]), show = true)
    # println("Plotting!!!");
    # contourf(reverse(x_plot(:,:,1), dims = 1),[0.5,0.5]);
    # colormap([0,0,0]); set(gcf,'color','w'); axis equal; axis off;title('material layout');drawnow;
    # figure(2)
    # pcolormesh(reshape(MISES.*(0.5*sign.(x.-0.5).+0.5),nely,nelx));
    # colorbar();
    # imagesc(reshape(MISES.*(0.5*sign(x-0.5)+0.5),nely,nelx));axis equal;axis off; colormap('jet');title('Von-Mises Stress');colorbar;drawnow;
    dv = ones(nely, nelx, nelz) / (nelx * nely * nelz)
    # println("typeof(pnorm_sen[:]./Hs)=$(typeof(pnorm_sen[:]./Hs))");
    # println("typeof(H)=$(typeof(H))");
    sen = zeros(reverse(size(Hs)))
    # println("size(H)=$(size(H))");
    # println("size(pnorm_sen)=$(size(pnorm_sen))");
    # println("size(Hs)=$(size(Hs))");
    sen[:] = H * (pnorm_sen[:] ./ Hs)
    dv[:] = H * (dv[:] ./ Hs)
    fval = [mean(x[:]) - 0.3]
    dfdx = dv[:]'
    # dfdx=[dv[:]'];
    df0dx = sen'
    f0val = pnorm
    f0val, df0dx, fval, dfdx
end

function meshgrid(x, y, z)
    X = zeros(size(y, 1), size(x, 1), size(z, 1))
    Y = zeros(size(y, 1), size(x, 1), size(z, 1))
    Z = zeros(size(y, 1), size(x, 1), size(z, 1))
    X .= x' .* ones(size(y, 1)) .* ones(1, 1, size(z, 1))
    Y .= ones(size(x, 1))' .* y .* ones(1, 1, size(z, 1))
    Z .= reshape(z, 1, 1, :) .* ones(size(y, 1), size(x, 1))
    X, Y, Z
end

function asymp(
    outeriter,
    n,
    xval,
    xold1,
    xold2,
    xmin,
    xmax,
    low,
    upp,
    raa0,
    raa,
    raa0eps,
    raaeps,
    df0dx,
    dfdx,
)
    #
    eeen = ones(n)
    xmami = xmax - xmin
    xmamieps = 0.00001 * eeen
    xmami = max.(xmami, xmamieps)
    # println("size(df0dx)=$(size(df0dx))");
    # println("size(xmami)=$(size(xmami))");
    raa0 = abs.(df0dx)' * xmami
    raa0 = max.(raa0eps, (0.1 / n) * raa0)
    raa = abs.(dfdx) * xmami
    raa = max.(raaeps, (0.1 / n) * raa)
    if outeriter < 2.5
        low = xval - 0.5 * xmami
        upp = xval + 0.5 * xmami
    else
        xxx = (xval - xold1) .* (xold1 - xold2)
        factor = copy(eeen)
        factor[findall(x -> x > 0, xxx)] .= 1.2
        factor[findall(x -> x < 0, xxx)] .= 0.7
        low = xval - factor .* (xold1 - low)
        upp = xval + factor .* (upp - xold1)
        lowmin = xval - 10 * xmami
        lowmax = xval - 0.01 * xmami
        uppmin = xval + 0.01 * xmami
        uppmax = xval + 10 * xmami
        low = max.(low, lowmin)
        low = min.(low, lowmax)
        upp = min.(upp, uppmax)
        upp = max.(upp, uppmin)
    end
    return low, upp, raa0, raa
end

# function gcmmasub(m,n,iter,epsimin,xval,xmin,xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx,a0,a,c,d);
#     #
#     eeen = ones(n)
#     # eeen = ones(n)
#     zeron = zeros(n)
#     # zeron = zeros(n)
#     #
#     # Calculations of the bounds alfa and beta.
#     albefa = 0.1
#     move = 0.5
#     #
#     zzz1 = low + albefa*(xval-low)
#     zzz2 = xval - move*(xmax-xmin)
#     zzz  = max.(zzz1,zzz2)
#     alfa = max.(zzz,xmin)
#     zzz1 = upp - albefa*(upp-xval)
#     zzz2 = xval + move*(xmax-xmin)
#     zzz  = min.(zzz1,zzz2)
#     beta = min.(zzz,xmax)
#     #
#     # Calculations of p0, q0, r0, P, Q, r and b.
#     xmami = xmax-xmin
#     xmamieps = 0.00001*eeen
#     xmami = max.(xmami,xmamieps)
#     xmamiinv = eeen./xmami
#     ux1 = upp-xval
#     ux2 = ux1.*ux1
#     xl1 = xval-low
#     xl2 = xl1.*xl1
#     uxinv = eeen./ux1
#     xlinv = eeen./xl1
#     #
#     p0 = copy(zeron)
#     q0 = copy(zeron)
#     p0 = max.(df0dx,0)
#     q0 = max.(-df0dx,0)
#     pq0 = p0 + q0
#     p0 = p0 + 0.001*pq0
#     q0 = q0 + 0.001*pq0
#     # println("size(raa0)=$(size(raa0))");
#     # println("size(xmamiinv)=$(size(xmamiinv))");
#     p0 = p0 + raa0[1]*xmamiinv
#     q0 = q0 + raa0[1]*xmamiinv
#     p0 = p0.*ux2;
#     q0 = q0.*xl2;
#     r0 = f0val .- p0'*uxinv - q0'*xlinv
#     #
#     P = spzeros(m,n)
#     Q = spzeros(m,n)
#     P = max.(dfdx,0)
#     Q = max.(-dfdx,0)
#     PQ = P + Q
#     P = P + 0.001*PQ
#     Q = Q + 0.001*PQ
#     P = P + raa*xmamiinv'
#     Q = Q + raa*xmamiinv'
#     P = P * spdiagm(n,n,vec(ux2))
#     Q = Q * spdiagm(n,n,vec(xl2))
#     r = fval - P*uxinv - Q*xlinv
#     b = -r
#     #
#     # Solving the subproblem by a primal-dual Newton method
#     xmma,ymma,zmma,lam,xsi,eta,mu,zet,s = subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d);
#     #
#     # Calculations of f0app and fapp.
#     ux1 = upp-xmma
#     xl1 = xmma-low
#     uxinv = eeen./ux1
#     xlinv = eeen./xl1
#     f0app = r0 + p0'*uxinv + q0'*xlinv
#     fapp  =  r +   P*uxinv +   Q*xlinv
#     #
#     #---------------------------------------------------------------------
#     return xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp
# end
# using SparseArrays
function gcmmasub(
    m,
    n,
    iter,
    epsimin,
    xval,
    xmin,
    xmax,
    low,
    upp,
    raa0,
    raa,
    f0val,
    df0dx,
    fval,
    dfdx,
    a0,
    a,
    c,
    d,
)
    #
    eeen = ones(n)
    zeron = zeros(n)
    #
    # Calculations of the bounds alfa and beta.
    albefa = 0.01
    move = 0.2
    #
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = max.(zzz1, zzz2)
    alfa = max.(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = min.(zzz1, zzz2)
    beta = min.(zzz, xmax)
    #
    # Calculations of p0, q0, r0, P, Q, r and b.
    xmami = xmax - xmin
    xmamieps = 0.00001 * eeen
    xmami = max(xmami, xmamieps)
    xmamiinv = eeen ./ xmami
    ux1 = upp - xval
    ux2 = ux1 .* ux1
    xl1 = xval - low
    xl2 = xl1 .* xl1
    uxinv = eeen ./ ux1
    xlinv = eeen ./ xl1
    #
    p0 = copy(zeron)
    q0 = copy(zeron)
    p0 = max.(df0dx, 0)
    q0 = max.(-df0dx, 0)
    pq0 = p0 + q0
    p0 = p0 + 0.001 * pq0
    q0 = q0 + 0.001 * pq0
    # @show size(raa0)
    # @show size(xmamiinv)
    # p0 = p0 + dot(raa0, xmamiinv)
    p0 = p0 + raa0 .* xmamiinv
    # p0 = p0 + raa0*xmamiinv
    q0 = q0 + raa0 .* xmamiinv
    # q0 = q0 + raa0*xmamiinv
    p0 = p0 .* ux2
    q0 = q0 .* xl2
    r0 = f0val .- p0' * uxinv - q0' * xlinv
    # r0 = f0val - p0'*uxinv - q0'*xlinv
    #
    P = spzeros(m, n)
    Q = spzeros(m, n)
    P = max.(dfdx, 0)
    Q = max.(-dfdx, 0)
    PQ = P + Q
    P = P + 0.001 * PQ
    Q = Q + 0.001 * PQ
    P = P + raa * xmamiinv'
    Q = Q + raa * xmamiinv'
    P = P * spdiagm(n, n, vec(ux2))
    # P = P * spdiagm(n,n,ux2)
    Q = Q * spdiagm(n, n, vec(xl2))
    # Q = Q * spdiagm(n,n,xl2)
    r = fval - P * uxinv - Q * xlinv
    b = -r
    #
    # Solving the subproblem by a primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s =
        subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)
    #
    # Calculations of f0app and fapp.
    ux1 = upp - xmma
    xl1 = xmma - low
    uxinv = eeen ./ ux1
    xlinv = eeen ./ xl1
    f0app = r0 + p0' * uxinv + q0' * xlinv
    fapp = r + P * uxinv + Q * xlinv
    #
    #---------------------------------------------------------------------
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp
end

function subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)
    #
    # This function subsolv solves the MMA subproblem:
    #         
    # minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
    #          + SUM[ ci*yi + 0.5*di*(yi)^2 ],
    #
    # subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
    #            alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
    #        
    # Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
    # Output: xmma,ymma,zmma, slack variables and Lagrange multiplers.
    #
    een = ones(n)
    eem = ones(m)
    epsi = 1
    epsvecn = epsi * een
    epsvecm = epsi * eem
    x = 0.5 * (alfa + beta)
    y = copy(eem)
    z = 1
    lam = copy(eem)
    xsi = een ./ (x - alfa)
    xsi = max.(xsi, een)
    eta = een ./ (beta - x)
    eta = max.(eta, een)
    mu = max.(eem, 0.5 * c)
    zet = 1
    s = copy(eem)
    itera = 0
    while epsi > epsimin
        epsvecn = epsi * een
        epsvecm = epsi * eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 .* ux1
        xl2 = xl1 .* xl1
        uxinv1 = een ./ ux1
        xlinv1 = een ./ xl1
        plam = p0 + P' * lam
        qlam = q0 + Q' * lam
        gvec = P * uxinv1 + Q * xlinv1
        dpsidx = plam ./ ux2 - qlam ./ xl2
        rex = dpsidx - xsi + eta
        rey = c + d .* y - mu - lam
        rez = a0 - zet .- Float64.(a)' * lam
        # rez = a0 - zet - a'*lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi .* (x - alfa) - epsvecn
        reeta = eta .* (beta - x) - epsvecn
        remu = mu .* y - epsvecm
        rezet = zet * z - epsi
        res = lam .* s - epsvecm
        residu1 = [rex' rey' rez]'
        residu2 = [relam' rexsi' reeta' remu' rezet res']'
        residu = [residu1' residu2']'
        residunorm = sqrt(residu' * residu)
        residumax = maximum(abs.(residu))
        ittt = 0
        while (residumax > 0.9 * epsi) & (ittt < 200)
            ittt = ittt + 1
            itera = itera + 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 .* ux1
            xl2 = xl1 .* xl1
            ux3 = ux1 .* ux2
            xl3 = xl1 .* xl2
            uxinv1 = een ./ ux1
            xlinv1 = een ./ xl1
            uxinv2 = een ./ ux2
            xlinv2 = een ./ xl2
            plam = p0 + P' * lam
            qlam = q0 + Q' * lam
            gvec = P * uxinv1 + Q * xlinv1
            GG = P * spdiagm(n, n, vec(uxinv2)) - Q * spdiagm(n, n, vec(xlinv2))
            # GG = P*spdiagm(n,n,uxinv2) - Q*spdiagm(n,n,xlinv2)
            dpsidx = plam ./ ux2 - qlam ./ xl2
            delx = dpsidx - epsvecn ./ (x - alfa) + epsvecn ./ (beta - x)
            dely = c + d .* y - lam - epsvecm ./ y
            delz = a0 .- Float64.(a)' * lam .- epsi / z
            # delz = a0 - a'*lam - epsi/z
            dellam = gvec - a * z - y - b + epsvecm ./ lam
            diagx = plam ./ ux3 + qlam ./ xl3
            diagx = 2 * diagx + xsi ./ (x - alfa) + eta ./ (beta - x)
            diagxinv = een ./ diagx
            diagy = d + mu ./ y
            diagyinv = eem ./ diagy
            diaglam = s ./ lam
            diaglamyi = diaglam + diagyinv
            if m < n
                blam = dellam + dely ./ diagy - GG * (delx ./ diagx)
                bb = [blam' delz]'
                Alam =
                    spdiagm(m, m, vec(diaglamyi)) + GG * spdiagm(n, n, vec(diagxinv)) * GG'
                # Alam = spdiagm(m,m,diaglamyi) + GG*spdiagm(n,n,diagxinv)*GG'
                AA = [
                    Alam a
                    a' -zet/z
                ]
                solut = AA \ bb
                dlam = solut[1:m]
                dz = solut[m+1]
                dx = -delx ./ diagx - (GG' * dlam) ./ diagx
            else
                diaglamyiinv = eem ./ diaglamyi
                dellamyi = dellam + dely ./ diagy
                Axx = spdiagm(n, n, diagx) + GG' * spdiagm(m, m, diaglamyiinv) * GG
                azz = zet / z + a' * (a ./ diaglamyi)
                axz = -GG' * (a ./ diaglamyi)
                bx = delx + GG' * (dellamyi ./ diaglamyi)
                bz = delz - a' * (dellamyi ./ diaglamyi)
                AA = [
                    Axx axz
                    axz' azz
                ]
                bb = [-bx' -bz]'
                solut = AA \ bb
                dx = solut[1:n]
                dz = solut[n+1]
                dlam =
                    (GG * dx) ./ diaglamyi - dz * (a ./ diaglamyi) + dellamyi ./ diaglamyi
            end
            #
            dy = -dely ./ diagy + dlam ./ diagy
            dxsi = -xsi + epsvecn ./ (x - alfa) - (xsi .* dx) ./ (x - alfa)
            deta = -eta + epsvecn ./ (beta - x) + (eta .* dx) ./ (beta - x)
            dmu = -mu + epsvecm ./ y - (mu .* dy) ./ y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm ./ lam - (s .* dlam) ./ lam
            xx = [y' z lam' xsi' eta' mu' zet s']'
            dxx = [dy' dz dlam' dxsi' deta' dmu' dzet ds']'
            #    
            stepxx = -1.01 * dxx ./ xx
            stmxx = maximum(stepxx)
            stepalfa = -1.01 * dx ./ (x - alfa)
            stmalfa = maximum(stepalfa)
            stepbeta = 1.01 * dx ./ (beta - x)
            stmbeta = maximum(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1)
            steg = 1 / stminv
            #
            xold = copy(x)
            yold = copy(y)
            zold = copy(z)
            lamold = copy(lam)
            xsiold = copy(xsi)
            etaold = copy(eta)
            muold = copy(mu)
            zetold = copy(zet)
            sold = copy(s)
            #
            itto = 0
            resinew = 2 * residunorm
            while (vec(resinew) > vec(residunorm)) & (itto < 50)
                # while (resinew > residunorm) & (itto < 50)
                itto = itto + 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 .* ux1
                xl2 = xl1 .* xl1
                uxinv1 = een ./ ux1
                xlinv1 = een ./ xl1
                plam = p0 + P' * lam
                qlam = q0 + Q' * lam
                gvec = P * uxinv1 + Q * xlinv1
                dpsidx = plam ./ ux2 - qlam ./ xl2
                rex = dpsidx - xsi + eta
                rey = c + d .* y - mu - lam
                rez = a0 - zet .- Float64.(a)' * lam
                # rez = a0 - zet - a'*lam
                # rez = a0 - zet - a'*lam
                relam = gvec - a * z - y + s - b
                rexsi = xsi .* (x - alfa) - epsvecn
                reeta = eta .* (beta - x) - epsvecn
                remu = mu .* y - epsvecm
                rezet = zet * z - epsi
                res = lam .* s - epsvecm
                residu1 = [rex' rey' rez]'
                residu2 = [relam' rexsi' reeta' remu' rezet res']'
                residu = [residu1' residu2']'
                resinew = sqrt(residu' * residu)
                steg = steg / 2
            end
            residunorm = copy(resinew)
            residumax = maximum(abs.(residu))
            steg = 2 * steg
        end
        if ittt > 198
            println("epsi: ", epsi)
            println("ittt: ", ittt)
        end
        epsi = 0.1 * epsi
    end
    xmma = copy(x)
    ymma = copy(y)
    zmma = copy(z)
    lamma = copy(lam)
    xsimma = copy(xsi)
    etamma = copy(eta)
    mumma = copy(mu)
    zetmma = copy(zet)
    smma = copy(s)
    #-------------------------------------------------------------
    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma
end

# function subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d)
#     #
#     # This function subsolv solves the MMA subproblem:
#     #         
#     # minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
#     #          + SUM[ ci*yi + 0.5*di*(yi)^2 ],
#     #
#     # subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
#     #            alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
#     #        
#     # Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
#     # Output: xmma,ymma,zmma, slack variables and Lagrange multiplers.
#     #
#     een = ones(n)
#     eem = ones(m)
#     epsi = 1
#     epsvecn = epsi*een
#     epsvecm = epsi*eem
#     x = 0.5*(alfa+beta)
#     y = copy(eem)
#     z = 1
#     lam = copy(eem)
#     xsi = een./(x-alfa)
#     xsi = max.(xsi,een)
#     eta = een./(beta-x)
#     eta = max.(eta,een)
#     mu  = max.(eem,0.5*c)
#     zet = 1
#     s = copy(eem)
#     itera = 0
#     while epsi > epsimin
#         epsvecn = epsi*een
#         epsvecm = epsi*eem
#         ux1 = upp-x
#         xl1 = x-low
#         ux2 = ux1.*ux1
#         xl2 = xl1.*xl1
#         uxinv1 = een./ux1
#         xlinv1 = een./xl1
#         plam = p0 + P'*lam 
#         qlam = q0 + Q'*lam 
#         gvec = P*uxinv1 + Q*xlinv1
#         dpsidx = plam./ux2 - qlam./xl2 
#         rex = dpsidx - xsi + eta
#         rey = c + d.*y - mu - lam
#         # println("size(a)=$(size(a))");
#         # println("size(lam)=$(size(lam))");
#         # println("typeof(a)=$(typeof(a))");
#         # println("typeof(lam)=$(typeof(lam))");
#         rez = a0 - zet .- Float64.(a)'*lam
#         relam = gvec - a*z - y + s - b
#         rexsi = xsi.*(x-alfa) - epsvecn
#         reeta = eta.*(beta-x) - epsvecn
#         remu = mu.*y - epsvecm
#         rezet = zet*z - epsi
#         res = lam.*s - epsvecm
#         residu1 = [rex' rey' rez]'
#         residu2 = [relam' rexsi' reeta' remu' rezet res']'
#         residu = [residu1' residu2']'
#         residunorm = sqrt(residu'*residu)
#         residumax = maximum(abs.(residu))
#         ittt = 0;
#         while (residumax > 0.9*epsi) & (ittt < 200)
#             ittt=ittt + 1
#             itera=itera + 1
#             ux1 = upp-x
#             xl1 = x-low
#             ux2 = ux1.*ux1
#             xl2 = xl1.*xl1
#             ux3 = ux1.*ux2
#             xl3 = xl1.*xl2
#             uxinv1 = een./ux1
#             xlinv1 = een./xl1
#             uxinv2 = een./ux2
#             xlinv2 = een./xl2
#             plam = p0 + P'*lam
#             qlam = q0 + Q'*lam
#             gvec = P*uxinv1 + Q*xlinv1
#             GG = P*spdiagm(n,n,vec(uxinv2)) - Q*spdiagm(n,n,vec(xlinv2))
#             dpsidx = plam./ux2 - qlam./xl2 
#             delx = dpsidx - epsvecn./(x-alfa) + epsvecn./(beta-x)
#             dely = c + d.*y - lam - epsvecm./y
#             delz = a0 .- Float64.(a)'*lam .- epsi/z
#             dellam = gvec - a*z - y - b + epsvecm./lam
#             diagx = plam./ux3 + qlam./xl3
#             diagx = 2*diagx + xsi./(x-alfa) + eta./(beta-x)
#             diagxinv = een./diagx
#             diagy = d + mu./y
#             diagyinv = eem./diagy
#             diaglam = s./lam
#             diaglamyi = diaglam+diagyinv
#             if m < n
#                 blam = dellam + dely./diagy - GG*(delx./diagx)
#                 bb = [blam' delz]'
#                 Alam = spdiagm(m,m,vec(diaglamyi)) + GG*spdiagm(n,n,vec(diagxinv))*GG'
#                 AA = [Alam     a;
#                         a'    -zet/z ]
#                 solut = AA\bb
#                 dlam = solut[1:m]
#                 dz = solut[m+1]
#                 dx = -delx./diagx - (GG'*dlam)./diagx
#             else
#                 diaglamyiinv = eem./diaglamyi
#                 dellamyi = dellam + dely./diagy
#                 Axx = spdiagm(n,n,diagx) + GG'*spdiagm(m,m,diaglamyiinv)*GG
#                 azz = zet/z + a'*(a./diaglamyi)
#                 axz = -GG'*(a./diaglamyi)
#                 bx = delx + GG'*(dellamyi./diaglamyi)
#                 bz  = delz - a'*(dellamyi./diaglamyi)
#                 AA = [Axx   axz;
#                         axz'  azz ]
#                 bb = [-bx' -bz]'
#                 solut = AA\bb
#                 dx  = solut[1:n]
#                 dz = solut[n+1]
#                 dlam = (GG*dx)./diaglamyi - dz*(a./diaglamyi) + dellamyi./diaglamyi
#             end
#         #
#             dy = -dely./diagy + dlam./diagy
#             dxsi = -xsi + epsvecn./(x-alfa) - (xsi.*dx)./(x-alfa)
#             deta = -eta + epsvecn./(beta-x) + (eta.*dx)./(beta-x)
#             dmu  = -mu + epsvecm./y - (mu.*dy)./y
#             dzet = -zet + epsi/z - zet*dz/z
#             ds   = -s + epsvecm./lam - (s.*dlam)./lam
#             xx  = [ y'  z  lam'  xsi'  eta'  mu'  zet  s']'
#             dxx = [dy' dz dlam' dxsi' deta' dmu' dzet ds']'
#         #    
#             stepxx = -1.01*dxx./xx
#             stmxx  = maximum(stepxx)
#             stepalfa = -1.01*dx./(x-alfa)
#             stmalfa = maximum(stepalfa)
#             stepbeta = 1.01*dx./(beta-x)
#             stmbeta = maximum(stepbeta)
#             stmalbe  = max(stmalfa,stmbeta)
#             stmalbexx = max(stmalbe,stmxx)
#             stminv = max(stmalbexx,1)
#             steg = 1/stminv
#         #
#             xold   =  copy(x)
#             yold   =  copy(y)
#             zold   =  copy(z)
#             lamold =  copy(lam)
#             xsiold =  copy(xsi)
#             etaold =  copy(eta)
#             muold  =  copy(mu)
#             zetold =  copy(zet)
#             sold   =  copy(s)
#         #
#             itto = 0
#             resinew = 2*residunorm
#             while (vec(resinew) > vec(residunorm)) & (itto < 50)
#                 itto = itto+1
#                 x   =   xold + steg*dx
#                 y   =   yold + steg*dy
#                 z   =   zold + steg*dz
#                 lam = lamold + steg*dlam
#                 xsi = xsiold + steg*dxsi
#                 eta = etaold + steg*deta
#                 mu  = muold  + steg*dmu
#                 zet = zetold + steg*dzet
#                 s   = sold + steg*ds
#                 ux1 = upp-x
#                 xl1 = x-low
#                 ux2 = ux1.*ux1
#                 xl2 = xl1.*xl1
#                 uxinv1 = een./ux1
#                 xlinv1 = een./xl1
#                 plam = p0 + P'*lam
#                 qlam = q0 + Q'*lam
#                 gvec = P*uxinv1 + Q*xlinv1
#                 dpsidx = plam./ux2 - qlam./xl2
#                 rex = dpsidx - xsi + eta
#                 rey = c + d.*y - mu - lam
#                 rez = a0 - zet .- Float64.(a)'*lam
#                 relam = gvec - a*z - y + s - b
#                 rexsi = xsi.*(x-alfa) - epsvecn
#                 reeta = eta.*(beta-x) - epsvecn
#                 remu = mu.*y - epsvecm
#                 rezet = zet*z - epsi
#                 res = lam.*s - epsvecm
#                 residu1 = [rex' rey' rez]'
#                 residu2 = [relam' rexsi' reeta' remu' rezet res']'
#                 residu = [residu1' residu2']'
#                 resinew = sqrt(residu'*residu)
#                 steg = steg/2
#             end
#         residunorm = copy(resinew)
#         residumax = maximum(abs.(residu))
#         steg = 2*steg
#         end
#         if ittt > 198
#             println("epsi: ", epsi)
#             println("ittt: ", ittt)
#         end
#         epsi = 0.1*epsi
#     end
#     xmma   = copy(x)
#     ymma   = copy(y)
#     zmma   = copy(z)
#     lamma  = copy(lam)
#     xsimma = copy(xsi)
#     etamma = copy(eta)
#     mumma  = copy(mu)
#     zetmma = copy(zet)
#     smma   = copy(s)
#     #-------------------------------------------------------------
#     return xmma,ymma,zmma,lamma,xsimma,etamma,mumma,zetmma,smma
# end

function kktcheck(
    m,
    n,
    x,
    y,
    z,
    lam,
    xsi,
    eta,
    mu,
    zet,
    s,
    xmin,
    xmax,
    df0dx,
    fval,
    dfdx,
    a0,
    a,
    c,
    d,
)
    #
    #  The left hand sides of the KKT conditions for the following
    #  nonlinear programming problem are calculated.
    #         
    #      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    #    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
    #                xmax_j <= x_j <= xmin_j,    j = 1,...,n
    #                z >= 0,   y_i >= 0,         i = 1,...,m
    #*** INPUT:
    #
    #   m    = The number of general constraints.
    #   n    = The number of variables x_j.
    #   x    = Current values of the n variables x_j.
    #   y    = Current values of the m variables y_i.
    #   z    = Current value of the single variable z.
    #  lam   = Lagrange multipliers for the m general constraints.
    #  xsi   = Lagrange multipliers for the n constraints xmin_j - x_j <= 0.
    #  eta   = Lagrange multipliers for the n constraints x_j - xmax_j <= 0.
    #   mu   = Lagrange multipliers for the m constraints -y_i <= 0.
    #  zet   = Lagrange multiplier for the single constraint -z <= 0.
    #   s    = Slack variables for the m general constraints.
    #  xmin  = Lower bounds for the variables x_j.
    #  xmax  = Upper bounds for the variables x_j.
    #  df0dx = Vector with the derivatives of the objective function f_0
    #          with respect to the variables x_j, calculated at x.
    #  fval  = Vector with the values of the constraint functions f_i,
    #          calculated at x.
    #  dfdx  = (m x n)-matrix with the derivatives of the constraint functions
    #          f_i with respect to the variables x_j, calculated at x.
    #          dfdx(i,j) = the derivative of f_i with respect to x_j.
    #   a0   = The constants a_0 in the term a_0*z.
    #   a    = Vector with the constants a_i in the terms a_i*z.
    #   c    = Vector with the constants c_i in the terms c_i*y_i.
    #   d    = Vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
    #     
    #*** OUTPUT:
    #
    # residu     = the residual vector for the KKT conditions.
    # residunorm = sqrt(residu'*residu).
    # residumax  = max(abs(residu)).
    #
    # println("size(dfdx)=$(size(dfdx))");
    # println("size(lam)=$(size(lam))");
    rex = df0dx + dfdx' * lam[1] - xsi + eta
    rey = c + d .* y - mu - lam
    rez = a0 - zet .- Float64.(a)' * lam
    relam = fval - a * z - y + s
    rexsi = xsi .* (x - xmin)
    reeta = eta .* (xmax - x)
    remu = mu .* y
    rezet = zet * z
    res = lam .* s
    #
    residu1 = [rex' rey' rez]'
    residu2 = [relam' rexsi' reeta' remu' rezet res']'
    residu = [residu1' residu2']'
    residunorm = sqrt(residu' * residu)
    residumax = maximum(abs.(residu))
    #---------------------------------------------------------------------
    return residu, residunorm, residumax
end

write("matching.txt", "");
write("not_matching.txt", "");
iom = open("matching.txt", "a");
ionm = open("not_matching.txt", "a");
nelx = 200;
open("julia/nelx.txt", "w") do io
    writedlm(io, nelx)
end
m_nelx = readdlm("matlab/nelx.txt")[1, 1];
@show size(m_nelx);
@show size(nelx);
if m_nelx != nelx
    write(ionm, "nelx\n")
else
    write(iom, "nelx\n")
end
nely = 60;
open("julia/nely.txt", "w") do io
    writedlm(io, nely)
end
m_nely = readdlm("matlab/nely.txt")[1, 1];
if m_nely != nely
    write(ionm, "nely\n")
else
    write(iom, "nely\n")
end
nelz = 1;
open("julia/nelz.txt", "w") do io
    writedlm(io, nelz)
end
m_nelz = readdlm("matlab/nelz.txt")[1, 1];
if m_nelz != nelz
    write(ionm, "nelz\n")
else
    write(iom, "nelz\n")
end
x = 0.3 * ones(nely, nelx, nelz);
open("julia/x.txt", "w") do io
    writedlm(io, x)
end
m_x = readdlm("matlab/x.txt");
if m_x != x
    write(ionm, "x\n")
else
    write(iom, "x\n")
end
H, Hs = prepare_filter(nelx, nely, nelz, 2.5);
(x_H, y_H, v_H) = findnz(H);
# @show x_H;
# @show size(x_H);
open("julia/H.txt", "w") do io
    writedlm(io, ([reshape(x_H, 1, :); reshape(y_H, 1, :); reshape(v_H, 1, :)])')
end
m_HH = readdlm("matlab/H.txt");
if m_HH[1, 1] != size(H, 1) ||
   m_HH[1, 2] != size(H, 2) ||
   vec(m_HH[2:size(m_HH, 1), 1]) != x_H ||
   vec(m_HH[2:size(m_HH, 1), 2]) != y_H ||
   vec(m_HH[2:size(m_HH, 1), 3]) != v_H
    write(ionm, "H\n")
else
    write(iom, "H\n")
end
open("julia/Hs.txt", "w") do io
    writedlm(io, Hs)
end
m_Hs = readdlm("matlab/x.txt");
if m_Hs != Hs
    write(ionm, "Hs\n")
else
    write(iom, "Hs\n")
end
# (x_Hs, y_Hs, v_Hs) = findnz(Hs);
# open("julia/Hs.txt", "w") do io
#     writedlm(io, [x_Hs, y_Hs, v_Hs]);
# end
# m_HHs = readdlm("matlab/Hs.txt");
# if m_HHs[1, 1] != size(Hs, 1) || m_HHs[1, 2] != size(Hs, 2) || vec(m_HHs[2:size(m_HHs, 1), 1]) != x_Hs || vec(m_HHs[2:size(m_HHs, 1), 2]) != y_Hs || vec(m_HHs[2:size(m_HHs, 1), 3]) != v_Hs
#     write(ionm, "Hs\n");
# else
#     write(iom, "Hs\n");
# end
# println("H in main=$(size(H))");
m = 1;
open("julia/m.txt", "w") do io
    writedlm(io, m)
end
m_m = readdlm("matlab/m.txt");
if m_m != m
    write(ionm, "m\n")
else
    write(iom, "m\n")
end
epsimin = 0.0000001;
open("julia/epsimin.txt", "w") do io
    writedlm(io, epsimin)
end
m_epsimin = readdlm("matlab/epsimin.txt");
if m_epsimin != epsimin
    write(ionm, "epsimin\n")
else
    write(iom, "epsimin\n")
end
n = length(x[:]);
open("julia/n.txt", "w") do io
    writedlm(io, n)
end
m_n = readdlm("matlab/n.txt");
if m_n != n
    write(ionm, "n\n")
else
    write(iom, "n\n")
end
# println("n = length(x[:])=$n");
let xval = x[:]
    open("julia/xval.txt", "w") do io
        writedlm(io, xval)
    end
    m_xval = readdlm("matlab/xval.txt")
    if m_xval != xval
        write(ionm, "xval\n")
    else
        write(iom, "xval\n")
    end
    xold1 = xval
    open("julia/xold1.txt", "w") do io
        writedlm(io, xold1)
    end
    m_xold1 = readdlm("matlab/xold1.txt")
    if m_xold1 != xold1
        write(ionm, "xold1\n")
    else
        write(iom, "xold1\n")
    end
    xold2 = xval
    open("julia/xold2.txt", "w") do io
        writedlm(io, xold2)
    end
    m_xold2 = readdlm("matlab/xold2.txt")
    if m_xold2 != xold2
        write(ionm, "xold2\n")
    else
        write(iom, "xold2\n")
    end
    xlb = 1e-3 * ones(n)
    open("julia/xlb.txt", "w") do io
        writedlm(io, xlb)
    end
    m_xlb = readdlm("matlab/xlb.txt")
    if m_xlb != xlb
        write(ionm, "xlb\n")
    else
        write(iom, "xlb\n")
    end
    xub = 1 * ones(n)
    open("julia/xub.txt", "w") do io
        writedlm(io, xub)
    end
    m_xub = readdlm("matlab/xub.txt")
    if m_xub != xub
        write(ionm, "xub\n")
    else
        write(iom, "xub\n")
    end
    xmin = xlb
    open("julia/xmin.txt", "w") do io
        writedlm(io, xmin)
    end
    m_xmin = readdlm("matlab/xmin.txt")
    if m_xmin != xmin
        write(ionm, "xmin\n")
    else
        write(iom, "xmin\n")
    end
    xmax = xub
    open("julia/xmax.txt", "w") do io
        writedlm(io, xmax)
    end
    m_xmax = readdlm("matlab/xmax.txt")
    if m_xmax != xmax
        write(ionm, "xmax\n")
    else
        write(iom, "xmax\n")
    end
    low = xlb
    open("julia/low.txt", "w") do io
        writedlm(io, low)
    end
    m_low = readdlm("matlab/low.txt")
    if m_low != low
        write(ionm, "low\n")
    else
        write(iom, "low\n")
    end
    upp = xub
    open("julia/upp.txt", "w") do io
        writedlm(io, upp)
    end
    m_upp = readdlm("matlab/upp.txt")
    if m_upp != upp
        write(ionm, "upp\n")
    else
        write(iom, "upp\n")
    end
    c = [1e4]'
    open("julia/c.txt", "w") do io
        writedlm(io, c)
    end
    m_c = readdlm("matlab/c.txt")
    if m_c != c
        write(ionm, "c\n")
    else
        write(iom, "c\n")
    end
    d = [0]'
    open("julia/d.txt", "w") do io
        writedlm(io, d)
    end
    m_d = readdlm("matlab/d.txt")
    if m_d != d
        write(ionm, "d\n")
    else
        write(iom, "d\n")
    end
    a0 = 0
    open("julia/a0.txt", "w") do io
        writedlm(io, a0)
    end
    m_a0 = readdlm("matlab/a0.txt")
    if m_a0 != a0
        write(ionm, "a0\n")
    else
        write(iom, "a0\n")
    end
    a = [0]'
    open("julia/a.txt", "w") do io
        writedlm(io, a)
    end
    m_a = readdlm("matlab/a.txt")
    if m_a != a
        write(ionm, "a\n")
    else
        write(iom, "a\n")
    end
    raa0 = 0.0001
    open("julia/raa0.txt", "w") do io
        writedlm(io, raa0)
    end
    m_raa0 = readdlm("matlab/raa0.txt")
    if m_raa0 != raa0
        write(ionm, "raa0\n")
    else
        write(iom, "raa0\n")
    end
    raa = 0.0001
    open("julia/raa.txt", "w") do io
        writedlm(io, raa)
    end
    m_raa = readdlm("matlab/raa.txt")
    if m_raa != raa
        write(ionm, "raa\n")
    else
        write(iom, "raa\n")
    end
    raa0eps = 0.0000001
    open("julia/raa0eps.txt", "w") do io
        writedlm(io, raa0eps)
    end
    m_raa0eps = readdlm("matlab/raa0eps.txt")
    if m_raa0eps != raa0eps
        write(ionm, "raa0eps\n")
    else
        write(iom, "raa0eps\n")
    end
    raaeps = 0.0000001
    open("julia/raaeps.txt", "w") do io
        writedlm(io, raaeps)
    end
    m_raaeps = readdlm("matlab/raaeps.txt")
    if m_raaeps != raaeps
        write(ionm, "raaeps\n")
    else
        write(iom, "raaeps\n")
    end
    let outeriter = 0
        open("julia/outeriter.txt", "w") do io
            writedlm(io, outeriter)
        end
        m_outeriter = readdlm("matlab/outeriter.txt")
        if m_outeriter != outeriter
            write(ionm, "outeriter\n")
        else
            write(iom, "outeriter\n")
        end
        maxoutit = 120
        open("julia/maxoutit.txt", "w") do io
            writedlm(io, maxoutit)
        end
        m_maxoutit = readdlm("matlab/maxoutit.txt")
        if m_maxoutit != maxoutit
            write(ionm, "maxoutit\n")
        else
            write(iom, "maxoutit\n")
        end
        kkttol = 0
        open("julia/kkttol.txt", "w") do io
            writedlm(io, kkttol)
        end
        m_kkttol = readdlm("matlab/kkttol.txt")
        if m_kkttol != kkttol
            write(ionm, "kkttol\n")
        else
            write(iom, "kkttol\n")
        end
        x_his = zeros(nelx * nely * nelz, maxoutit)
        open("julia/x_his.txt", "w") do io
            writedlm(io, x_his)
        end
        m_x_his = readdlm("matlab/x_his.txt")
        if m_x_his != x_his
            write(ionm, "x_his\n")
        else
            write(iom, "x_his\n")
        end
        if outeriter < 0.5
            f0val, df0dx, fval, dfdx = stress_minimize(xval, Hs, H, outeriter)
            open("julia/f0val.txt", "w") do io
                writedlm(io, f0val)
            end
            m_f0val = readdlm("matlab/f0val.txt")
            if m_f0val != f0val
                write(ionm, "f0val\n")
            else
                write(iom, "f0val\n")
            end
            open("julia/df0dx.txt", "w") do io
                writedlm(io, df0dx)
            end
            m_df0dx = readdlm("matlab/df0dx.txt")
            if m_df0dx != df0dx
                write(ionm, "df0dx\n")
            else
                write(iom, "df0dx\n")
            end
            open("julia/fval.txt", "w") do io
                writedlm(io, fval)
            end
            m_fval = readdlm("matlab/fval.txt")
            if m_fval != fval
                write(ionm, "fval\n")
            else
                write(iom, "fval\n")
            end
            open("julia/dfdx.txt", "w") do io
                writedlm(io, dfdx)
            end
            m_dfdx = readdlm("matlab/dfdx.txt")
            if m_dfdx != dfdx
                write(ionm, "dfdx\n")
            else
                write(iom, "dfdx\n")
            end
            innerit = 0
            open("julia/innerit.txt", "w") do io
                writedlm(io, innerit)
            end
            m_innerit = readdlm("matlab/innerit.txt")
            if m_innerit != innerit
                write(ionm, "innerit\n")
            else
                write(iom, "innerit\n")
            end
            outvector1 = [outeriter innerit xval']
            open("julia/outvector1.txt", "w") do io
                writedlm(io, outvector1)
            end
            @show size(outvector1)
            @show size(outeriter)
            @show size(innerit)
            @show size(xval)
            m_outvector1 = readdlm("matlab/outvector1.txt")
            if m_outvector1 != outvector1
                write(ionm, "outvector1\n")
            else
                write(iom, "outvector1\n")
            end
            outvector2 = [f0val fval']
            open("julia/outvector2.txt", "w") do io
                writedlm(io, outvector2)
            end
            m_outvector2 = readdlm("matlab/outvector2.txt")
            if m_outvector2 != outvector2
                write(ionm, "outvector2\n")
            else
                write(iom, "outvector2\n")
            end
        end
        kktnorm = kkttol + 1
        open("julia/kktnorm.txt", "w") do io
            writedlm(io, kktnorm)
        end
        m_kktnorm = readdlm("matlab/kktnorm.txt")
        if m_kktnorm != kktnorm
            write(ionm, "kktnorm\n")
        else
            write(iom, "kktnorm\n")
        end
        let outit = 0
            open("julia/outit.txt", "w") do io
                writedlm(io, outit)
            end
            m_outit1 = readdlm("matlab/outit.txt")
            if m_outit1 != outit
                write(ionm, "outit\n")
            else
                write(iom, "outit\n")
            end
            while outit < maxoutit
                outit = outit + 1
                open("julia/outit/outit$outit.txt", "w") do io
                    writedlm(io, outit)
                end
                m_outit1 = readdlm("matlab/outit/outit$outit.txt")
                if m_outit1 != outit
                    write(ionm, "outit$outit\n")
                else
                    write(iom, "outit$outit\n")
                end
                outeriter = outeriter + 1
                open("julia/outeriter/outeriter$outit.txt", "w") do io
                    writedlm(io, outeriter)
                end
                m_outeriter = readdlm("matlab/outeriter/outeriter$outit.txt")
                if m_outeriter != outeriter
                    write(ionm, "outeriter$outit\n")
                else
                    write(iom, "outeriter$outit\n")
                end
                #### The parameters low, upp, raa0 and raa are calculated:
                low, upp, raa0, raa = asymp(
                    outeriter,
                    n,
                    xval,
                    xold1,
                    xold2,
                    xmin,
                    xmax,
                    low,
                    upp,
                    raa0,
                    raa,
                    raa0eps,
                    raaeps,
                    df0dx,
                    dfdx,
                )
                open("julia/low/low$outit.txt", "w") do io
                    writedlm(io, low)
                end
                m_low = readdlm("matlab/low/low$outit.txt")
                if m_low != low
                    write(ionm, "low$outit\n")
                else
                    write(iom, "low$outit\n")
                end
                open("julia/upp/upp$outit.txt", "w") do io
                    writedlm(io, upp)
                end
                m_upp = readdlm("matlab/upp/upp$outit.txt")
                if m_upp != upp
                    write(ionm, "upp$outit\n")
                else
                    write(iom, "upp$outit\n")
                end
                open("julia/raa0/raa0$outit.txt", "w") do io
                    writedlm(io, raa0)
                end
                m_raa0 = readdlm("matlab/raa0/raa0$outit.txt")
                if m_raa0 != raa0
                    write(ionm, "raa0$outit\n")
                else
                    write(iom, "raa0$outit\n")
                end
                open("julia/raa/raa$outit.txt", "w") do io
                    writedlm(io, raa)
                end
                m_raa = readdlm("matlab/raa/raa$outit.txt")
                if m_raa != raa
                    write(ionm, "raa$outit\n")
                else
                    write(iom, "raa$outit\n")
                end
                # println("size(raa0)=$(size(raa0))");
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(
                    m,
                    n,
                    outeriter,
                    epsimin,
                    xval,
                    xmin,
                    xmax,
                    low,
                    upp,
                    raa0,
                    raa,
                    f0val,
                    df0dx,
                    fval,
                    dfdx,
                    a0,
                    a,
                    c,
                    d,
                )
                open("julia/xmma/xmma$outit.txt", "w") do io
                    writedlm(io, xmma)
                end
                m_xmma = readdlm("matlab/xmma/xmma$outit.txt")
                if m_xmma != xmma
                    write(ionm, "xmma$outit\n")
                else
                    write(iom, "xmma$outit\n")
                end
                open("julia/ymma/ymma$outit.txt", "w") do io
                    writedlm(io, ymma)
                end
                m_ymma = readdlm("matlab/ymma/ymma$outit.txt")
                if m_ymma != ymma
                    write(ionm, "ymma$outit\n")
                else
                    write(iom, "ymma$outit\n")
                end
                open("julia/zmma/zmma$outit.txt", "w") do io
                    writedlm(io, zmma)
                end
                m_zmma = readdlm("matlab/zmma/zmma$outit.txt")
                if m_zmma != zmma
                    write(ionm, "zmma$outit\n")
                else
                    write(iom, "zmma$outit\n")
                end
                open("julia/lam/lam$outit.txt", "w") do io
                    writedlm(io, lam)
                end
                m_lam = readdlm("matlab/lam/lam$outit.txt")
                if m_lam != lam
                    write(ionm, "lam$outit\n")
                else
                    write(iom, "lam$outit\n")
                end
                open("julia/xsi/xsi$outit.txt", "w") do io
                    writedlm(io, xsi)
                end
                m_xsi = readdlm("matlab/xsi/xsi$outit.txt")
                if m_xsi != xsi
                    write(ionm, "xsi$outit\n")
                else
                    write(iom, "xsi$outit\n")
                end
                open("julia/eta/eta$outit.txt", "w") do io
                    writedlm(io, eta)
                end
                m_eta = readdlm("matlab/eta/eta$outit.txt")
                if m_eta != eta
                    write(ionm, "eta$outit\n")
                else
                    write(iom, "eta$outit\n")
                end
                open("julia/mu/mu$outit.txt", "w") do io
                    writedlm(io, mu)
                end
                m_mu = readdlm("matlab/mu/mu$outit.txt")
                if m_mu != mu
                    write(ionm, "mu$outit\n")
                else
                    write(iom, "mu$outit\n")
                end
                open("julia/zet/zet$outit.txt", "w") do io
                    writedlm(io, zet)
                end
                m_zet = readdlm("matlab/zet/zet$outit.txt")
                if m_zet != zet
                    write(ionm, "zet$outit\n")
                else
                    write(iom, "zet$outit\n")
                end
                open("julia/s/s$outit.txt", "w") do io
                    writedlm(io, s)
                end
                m_s = readdlm("matlab/s/s$outit.txt")
                if m_s != s
                    write(ionm, "s$outit\n")
                else
                    write(iom, "s$outit\n")
                end
                open("julia/f0app/f0app$outit.txt", "w") do io
                    writedlm(io, f0app)
                end
                m_f0app = readdlm("matlab/f0app/f0app$outit.txt")
                if m_f0app != f0app
                    write(ionm, "f0app$outit\n")
                else
                    write(iom, "f0app$outit\n")
                end
                open("julia/fapp/fapp$outit.txt", "w") do io
                    writedlm(io, fapp)
                end
                m_fapp = readdlm("matlab/fapp/fapp$outit.txt")
                if m_fapp != fapp
                    write(ionm, "fapp$outit\n")
                else
                    write(iom, "fapp$outit\n")
                end
                xold2 = xold1
                open("julia/xold2/xold2$outit.txt", "w") do io
                    writedlm(io, xold2)
                end
                m_xold2 = readdlm("matlab/xold2/xold2$outit.txt")
                if m_xold2 != xold2
                    write(ionm, "xold2$outit\n")
                else
                    write(iom, "xold2$outit\n")
                end
                xold1 = xval
                open("julia/xold1/xold1$outit.txt", "w") do io
                    writedlm(io, xold1)
                end
                m_xold1 = readdlm("matlab/xold1/xold1$outit.txt")
                if m_xold1 != xold1
                    write(ionm, "xold1$outit\n")
                else
                    write(iom, "xold1$outit\n")
                end
                xval = xmma
                open("julia/xval/xval$outit.txt", "w") do io
                    writedlm(io, xval)
                end
                m_xval = readdlm("matlab/xval/xval$outit.txt")
                if m_xval != xval
                    write(ionm, "xval$outit\n")
                else
                    write(iom, "xval$outit\n")
                end
                f0val, df0dx, fval, dfdx = stress_minimize(xval, Hs, H, outit)
                open("julia/f0val/f0val$outit.txt", "w") do io
                    writedlm(io, f0val)
                end
                m_f0val = readdlm("matlab/f0val/f0val$outit.txt")
                if m_f0val != f0val
                    write(ionm, "f0val$outit\n")
                else
                    write(iom, "f0val$outit\n")
                end
                open("julia/df0dx/df0dx$outit.txt", "w") do io
                    writedlm(io, df0dx)
                end
                m_df0dx = readdlm("matlab/df0dx/df0dx$outit.txt")
                if m_df0dx != df0dx
                    write(ionm, "df0dx$outit\n")
                else
                    write(iom, "df0dx$outit\n")
                end
                open("julia/fval/fval$outit.txt", "w") do io
                    writedlm(io, fval)
                end
                m_fval = readdlm("matlab/fval/fval$outit.txt")
                if m_fval != fval
                    write(ionm, "fval$outit\n")
                else
                    write(iom, "fval$outit\n")
                end
                open("julia/dfdx/dfdx$outit.txt", "w") do io
                    writedlm(io, dfdx)
                end
                m_dfdx = readdlm("matlab/dfdx/dfdx$outit.txt")
                if m_dfdx != dfdx
                    write(ionm, "dfdx$outit\n")
                else
                    write(iom, "dfdx$outit\n")
                end
                # PRINT RESULTS
                @printf(
                    " It.:%5i      P-norm Stress.:%11.4f   Vol.:%7.3f \n",
                    outit,
                    f0val,
                    mean(xval[:])
                )
                #### The residual vector of the KKT conditions is calculated:
                residu, kktnorm, residumax = kktcheck(
                    m,
                    n,
                    xmma,
                    ymma,
                    zmma,
                    lam,
                    xsi,
                    eta,
                    mu,
                    zet,
                    s,
                    xmin,
                    xmax,
                    df0dx,
                    fval,
                    dfdx,
                    a0,
                    a,
                    c,
                    d,
                )
                open("julia/residu/residu$outit.txt", "w") do io
                    writedlm(io, residu)
                end
                m_residu = readdlm("matlab/residu/residu$outit.txt")
                if m_residu != residu
                    write(ionm, "residu$outit\n")
                else
                    write(iom, "epsimin$outit\n")
                end
                open("julia/kktnorm/kktnorm$outit.txt", "w") do io
                    writedlm(io, kktnorm)
                end
                m_kktnorm = readdlm("matlab/kktnorm/kktnorm$outit.txt")
                if m_kktnorm != kktnorm
                    write(ionm, "kktnorm$outit\n")
                else
                    write(iom, "kktnorm$outit\n")
                end
                open("julia/residumax/residumax$outit.txt", "w") do io
                    writedlm(io, residumax)
                end
                m_residumax = readdlm("matlab/residumax/residumax$outit.txt")
                if m_residumax != residumax
                    write(ionm, "residumax$outit\n")
                else
                    write(iom, "residumax$outit\n")
                end
                outvector1 = [outeriter innerit xval']
                open("julia/outvector1/outvector1$outit.txt", "w") do io
                    writedlm(io, outvector1)
                end
                m_outvector1 = readdlm("matlab/outvector1/outvector1$outit.txt")
                if m_outvector1 != outvector1
                    write(ionm, "outvector1$outit\n")
                else
                    write(iom, "outvector1$outit\n")
                end
                outvector2 = [f0val fval']
                open("julia/outvector2/outvector2$outit.txt", "w") do io
                    writedlm(io, outvector2)
                end
                m_outvector2 = readdlm("matlab/outvector2/outvector2$outit.txt")
                if m_outvector2 != outvector2
                    write(ionm, "outvector2$outit\n")
                else
                    write(iom, "outvector2$outit\n")
                end
                x_his[:, outit] = xmma
                open("julia/x_his/x_his$outit.txt", "w") do io
                    writedlm(io, x_his)
                end
                m_x_his = readdlm("matlab/x_his/x_his$outit.txt")
                if m_x_his != x_his
                    write(ionm, "x_his$outit\n")
                else
                    write(iom, "x_his$outit\n")
                end
            end
            # x_plot=reshape(x,nely,nelx,nelz);
            # f=Figure();
            # ax1 = Axis(f[1, 1]);
            # # title = "Current design",);
            # image!(ax1, reverse(x_plot[:,:,1]));
            # wait(display(f));
        end
    end
end
