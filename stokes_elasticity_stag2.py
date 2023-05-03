"""
In this file a staggered solution approach is taken:
1. First solve Stokes equations
2. Solve nonlinear elasticity
3. Solve ALE problem for fluid domain
"""


from dolfin import *
from multiphenics import *
from helpers import *
import numpy as np

meshname = 'channel_sphere'
# dir = '/media/eg21388/data/fenics/stokes_elasticity/'
dir = 'output/'

"""
    parameters
"""

N = 200
# e_all = np.logspace(-2, 0, N)
e_all = np.linspace(1e-2, 0.5, N)

# e_all = [0.1]
# t_all = np.linspace(0, 1, N)

# N = 1
# e_all = [1e-1]

eps = Constant(e_all[0])


snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 8,
                                          "report": True,
                                          "absolute_tolerance": 1e-8,
                                          "error_on_nonconvergence": False}}

parameters["ghost_mode"] = "shared_facet"


output_f = XDMFFile(dir + "fluid.xdmf")
output_f.parameters["rewrite_function_mesh"] = False
output_f.parameters["functions_share_mesh"] = True
output_f.parameters["flush_output"] = True


output_s = XDMFFile(dir + "solid.xdmf")
output_s.parameters["rewrite_function_mesh"] = False
output_s.parameters["functions_share_mesh"] = True
output_s.parameters["flush_output"] = True

#---------------------------------------------------------------------
# problem geometry: mesh and boundaries
#---------------------------------------------------------------------
mesh = Mesh('mesh/'+meshname+'.xml')
subdomains = MeshFunction("size_t", mesh, 'mesh/' + meshname + '_physical_region.xml')
bdry = MeshFunction("size_t", mesh, 'mesh/' + meshname + "_facet_region.xml")

# define the boundaries (values from the gmsh file)
circle = 1
fluid_axis = 2
inlet = 3
outlet = 4
wall = 5
solid_axis = 6

# define the domains
fluid = 10
solid = 11

Of = generate_subdomain_restriction(mesh, subdomains, fluid)
Os = generate_subdomain_restriction(mesh, subdomains, solid)
Sig = generate_interface_restriction(mesh, subdomains, {fluid, solid})

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=bdry)
dS = Measure("dS", domain=mesh, subdomain_data=bdry)
dS = dS(circle)

# normal and tangent vectors
nn = FacetNormal(mesh); tt = as_vector((-nn[1], nn[0]))

#---------------------------------------------------------------------
# elements, function spaces, and test/trial functions
#---------------------------------------------------------------------
P2 = VectorElement("CG", mesh.ufl_cell(), 2)
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
DGT = VectorElement("DGT", mesh.ufl_cell(), 1)
P0 = FiniteElement("R", mesh.ufl_cell(), 0)

mixed_element = BlockElement(P2, P1, DGT, P0, P0)
Vf = BlockFunctionSpace(mesh, mixed_element, restrict = [Of, Of, Sig, Sig, Of])
Vs = BlockFunctionSpace(mesh, BlockElement(P2, P1, P0), restrict = [Os, Os, Os])


# unknowns and test functions
Yf = BlockTestFunction(Vf)
(v_f, q_f, eta, V_0, theta) = block_split(Yf)

Xtf = BlockTrialFunction(Vf)

Xf = BlockFunction(Vf)
(u_f, p_f, lam, U_0, zeta) = block_split(Xf)

Ys = BlockTestFunction(Vs)
(v_s, q_s, g_s) = block_split(Ys)

Xts = BlockTrialFunction(Vs)

Xs = BlockFunction(Vs)
(u_s, p_s, f_s) = block_split(Xs)


# ALE
V_ALE = BlockFunctionSpace(mesh, [P2], restrict = [Of])

X_ALE = BlockFunction(V_ALE)
(u_a, ) = block_split(X_ALE)
Xt_ALE = BlockTrialFunction(V_ALE)
Y_ALE = BlockTestFunction(V_ALE)
(v_a, ) = block_split(Y_ALE)


#---------------------------------------------------------------------
# boundary conditions
#---------------------------------------------------------------------

far_field = Expression(('(1 - x[1] * x[1] / 0.25) * t', '0'), degree=0, t = 1)

bc_inlet = DirichletBC(Vf.sub(0), far_field, bdry, inlet)
bc_outlet = DirichletBC(Vf.sub(0), far_field, bdry, outlet)
# bc_outlet = DirichletBC(Vf.sub(1), Constant(0), bdry, outlet)
bc_fluid_axis = DirichletBC(Vf.sub(0).sub(1), Constant(0), bdry, fluid_axis)
bc_wall = DirichletBC(Vf.sub(0), Constant((0, 0)), bdry, wall)

bc_solid_axis = DirichletBC(Vs.sub(0).sub(1), Constant(0), bdry, solid_axis)


bc_f = BlockDirichletBC([bc_inlet, bc_outlet, bc_fluid_axis, bc_wall])
bc_s = BlockDirichletBC([bc_solid_axis])


#---------------------------------------------------------------------
# Define kinematics
#---------------------------------------------------------------------

I = Identity(2)

# ale
F_a = I + grad(u_a)
H_a = inv(F_a.T)
J_a = det(F_a)
E_a = 1/2 * (F_a.T * F_a - I)
sigma_a = F_a * (Constant(100) * tr(E_a) * I + 2 * E_a / eps)

# e = 1/2 * (grad(u_a) + grad(u_a).T)
# sigma_a = 100 * tr(e) * I + 2 / eps * e

# fluids
sigma_f = J_a * (-p_f * I + grad(u_f) * H_a.T + H_a * grad(u_f).T) * H_a
ic_f = div(J_a * inv(F_a) * u_f)


# solids
F = I + grad(u_s)
H = inv(F.T)
sigma_s = 1 / eps * (F - H) - p_s * H
ic_s = det(F) - 1

# sigma_s = -p_s * I + (grad(u_s) + grad(u_s).T) / eps
# ic_s = div(u_s)

#---------------------------------------------------------------------
# build equations
#---------------------------------------------------------------------

ez = as_vector([1,0])

FUN1 = -inner(sigma_f, grad(v_f)) * dx(fluid) + inner(lam("+"), v_f("+")) * dS
FUN2 = ic_f * q_f * dx(fluid) + zeta * q_f * dx(fluid)

FUN3 = -inner(sigma_s, grad(v_s)) * dx(solid) + inner(as_vector([f_s, 0]), v_s) * dx - inner(lam("-"), v_s("-")) * dS
FUN4 = ic_s * q_s * dx(solid)

FUN5 = inner(avg(eta), u_f("+") - as_vector([U_0("+"), 0])) * dS
FUN6 = dot(ez, lam("+")) * V_0("+") * dS

FUN7 = dot(ez, u_s) * g_s * dx(solid)
FUN8 = p_f * theta * dx(fluid)

F_ALE = [-inner(sigma_a, grad(v_a)) * dx(fluid)]
J_ALE = block_derivative(F_ALE, X_ALE, Xt_ALE)


FUN_f = [FUN1, FUN2, FUN5, FUN6, FUN8]
FUN_s = [FUN3, FUN4, FUN7]

JAC_f = block_derivative(FUN_f, Xf, Xtf)
JAC_s = block_derivative(FUN_s, Xs, Xts)


#---------------------------------------------------------------------
# set up the solver
#---------------------------------------------------------------------

# Initialize solver
problem_f = BlockNonlinearProblem(FUN_f, Xf, bc_f, JAC_f)
solver_f = BlockPETScSNESSolver(problem_f)
solver_f.parameters.update(snes_solver_parameters["snes_solver"])

problem_s = BlockNonlinearProblem(FUN_s, Xs, bc_s, JAC_s)
solver_s = BlockPETScSNESSolver(problem_s)
solver_s.parameters.update(snes_solver_parameters["snes_solver"])


# extract solution components
(u_f, p_f, lam, U_0, zeta) = Xf.block_split()
(u_s, p_s, f_s) = Xs.block_split()

"""
    quantities for separate meshes
"""
mesh_f = SubMesh(mesh, subdomains, fluid)
mesh_s = SubMesh(mesh, subdomains, solid)
Vf = VectorFunctionSpace(mesh_f, "CG", 1)
Vs = VectorFunctionSpace(mesh_s, "CG", 1)
Vf0 = FunctionSpace(mesh_f, "CG", 1)
Vs0 = FunctionSpace(mesh_s, "CG", 1)

u_f_only = Function(Vf)
u_a_only = Function(Vf)
u_s_only = Function(Vs)
p_s_only = Function(Vs0)


def save_f(n):
    u_f_only = project(u_f, Vf)
    u_a_only = project(u_a, Vf)
    p_f_only = project(p_f, Vf0)

    u_f_only.rename("u_f", "u_f")
    u_a_only.rename("u_a", "u_a")
    p_f_only.rename("p_f", "p_f")

    output_f.write(u_f_only, n)
    output_f.write(u_a_only, n)
    output_f.write(p_f_only, n)

def save_s(n):
    u_s_only = project(u_s, Vs)
    u_s_only.rename("u_s", "u_s")
    output_s.write(u_s_only, n)

    p_s_only = project(p_s, Vs0)
    p_s_only.rename("p_s", "p_s")
    output_s.write(p_s_only, n)


"""
    BCs for the ALE problem
"""


ac_inlet = DirichletBC(V_ALE.sub(0), Constant((0,0)), bdry, inlet)
ac_outlet = DirichletBC(V_ALE.sub(0), Constant((0,0)), bdry, outlet)
ac_fluid_axis = DirichletBC(V_ALE.sub(0).sub(1), Constant((0)), bdry, fluid_axis)
ac_wall = DirichletBC(V_ALE.sub(0).sub(1), Constant((0)), bdry, wall)
ac_int = DirichletBC(V_ALE.sub(0), u_s, bdry, circle)

bc_ALE = BlockDirichletBC([ac_inlet, ac_outlet, ac_fluid_axis, ac_wall, ac_int])

problem_ALE = BlockNonlinearProblem(F_ALE, X_ALE, bc_ALE, J_ALE)
solver_ALE = BlockPETScSNESSolver(problem_ALE)
solver_ALE.parameters.update(snes_solver_parameters["snes_solver"])

(u_a,) = X_ALE.block_split()


"""
    solve
"""

print('solving fluid problem...')
(its, conv_f) = solver_f.solve()
save_f(0)
save_s(0)
print('f_0 =', f_s.vector()[:])
print('zeta =', zeta.vector()[:])

for n in range(1,N):
    eps.assign(e_all[n])

    print('-------------------------------------------------')
    print(f'solving problem with eps[{n:d}] = {e_all[n]:.2e}')

    print('solving solid problem...')
    (its, conv_s) = solver_s.solve()

    save_s(n)

    if conv_s == False:
        break


    print('updating fluid geometry')
    (its, conv) = solver_ALE.solve()

    if conv == False:
        break

    print('solving fluid problem...')
    (its, conv_f) = solver_f.solve()
    save_f(n)
    print('f_0 =', f_s.vector()[:])
    print('zeta =', zeta.vector()[:])


print(U_0.vector()[:])
