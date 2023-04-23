from dolfin import *
from multiphenics import *
from helpers import *
import numpy as np

meshname = 'channel_sphere'

"""
    parameters
"""

N = 50
e_all = np.logspace(-3, -1, N)
eps = Constant(e_all[0])


snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 8,
                                          "report": True,
                                          "error_on_nonconvergence": True}}

parameters["ghost_mode"] = "shared_facet"


output_f = XDMFFile("output/" + "stokes_elasticity_f.xdmf")
output_f.parameters["rewrite_function_mesh"] = True
output_f.parameters["functions_share_mesh"] = True
output_f.parameters["flush_output"] = True


output_s = XDMFFile("output/" + "stokes_elasticity_s.xdmf")
output_s.parameters["rewrite_function_mesh"] = True
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
solid_subaxis = 7

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

mixed_element = BlockElement(P2, P1, P2, P1, DGT, P0)
V = BlockFunctionSpace(mesh, mixed_element, restrict = [Of, Of, Os, Os, Sig, Sig])

# unknowns and test functions
Y = BlockTestFunction(V)
(v_f, q_f, v_s, q_s, eta, V_0) = block_split(Y)

Xt = BlockTrialFunction(V)

X = BlockFunction(V)
(u_f, p_f, u_s, p_s, lam, U_0) = block_split(X)


#---------------------------------------------------------------------
# boundary conditions
#---------------------------------------------------------------------

far_field = Expression(('(1 - x[1] * x[1] / 0.25) * t', '0'), degree=0, t = 1)

bc_inlet = DirichletBC(V.sub(0), far_field, bdry, inlet)
bc_outlet = DirichletBC(V.sub(0), far_field, bdry, outlet)
bc_fluid_axis = DirichletBC(V.sub(0).sub(1), Constant(0), bdry, fluid_axis)
bc_wall = DirichletBC(V.sub(0), Constant((0, 0)), bdry, wall)

bc_solid_axis = DirichletBC(V.sub(2).sub(1), Constant(0), bdry, solid_axis)
bc_pin = DirichletBC(V.sub(2), Constant((0, 0)), bdry, solid_subaxis)

bcs = BlockDirichletBC([bc_inlet, bc_outlet, bc_fluid_axis, bc_wall, bc_solid_axis, bc_pin])


#---------------------------------------------------------------------
# Define kinematics
#---------------------------------------------------------------------

I = Identity(2)

# fluids
sigma_f = -p_f * I + grad(u_f) + grad(u_f).T

# solids
F = inv(I - grad(u_s))
B = F * F.T
sigma_s = -p_s * I + 1 / eps * (B - I)

# sigma_s = -p_s *  I + grad(u_s) + grad(u_s).T


#---------------------------------------------------------------------
# build equations
#---------------------------------------------------------------------

# Incompressibility conditions
ic_f = div(u_f)
ic_s = det(F) - 1

FUN1 = -inner(sigma_f, grad(v_f)) * dx(fluid) + inner(lam("+"), v_f("+")) * dS
FUN2 = ic_f * q_f * dx(fluid)

FUN3 = -inner(sigma_s, grad(v_s)) * dx(solid) - inner(lam("-"), v_s("-")) * dS
FUN4 = ic_s * q_s * dx(solid)

FUN5 = inner(avg(eta), u_f("+") - as_vector([U_0("+"), 0])) * dS
FUN6 = dot(as_vector([1, 0]), lam("+")) * V_0("+") * dS

FUN = [FUN1, FUN2, FUN3, FUN4, FUN5, FUN6]

JAC = block_derivative(FUN, X, Xt)


#---------------------------------------------------------------------
# set up the solver
#---------------------------------------------------------------------

# Initialize solver
problem = BlockNonlinearProblem(FUN, X, bcs, JAC)
solver = BlockPETScSNESSolver(problem)
solver.parameters.update(snes_solver_parameters["snes_solver"])

# extract solution components
(u_f, p_f, u_s, p_s, lam, U_0) = X.block_split()

"""
    quantities for separate meshes
"""
mesh_f = SubMesh(mesh, subdomains, fluid)
mesh_s = SubMesh(mesh, subdomains, solid)
Vf = VectorFunctionSpace(mesh_f, "CG", 1)
Vs = VectorFunctionSpace(mesh_s, "CG", 1)
u_f_only = Function(Vf)
u_s_only = Function(Vs)

def save(n):
    u_f_only = project(u_f, Vf)
    u_s_only = project(u_s, Vs)

    output_f.write(u_f_only, n)
    output_s.write(u_s_only, n)


"""
    setup the ALE problem
"""

V_ALE = BlockFunctionSpace(mesh, [P2, P1], restrict = [Of, Of])

X_ALE = BlockFunction(V_ALE)
(u_a, p_a) = block_split(X_ALE)
Xt_ALE = BlockTrialFunction(V_ALE)
Y_ALE = BlockTestFunction(V_ALE)
(v_a, q_a) = block_split(Y_ALE)

ac_inlet = DirichletBC(V_ALE.sub(0), Constant((0,0)), bdry, inlet)
ac_outlet = DirichletBC(V_ALE.sub(0), Constant((0,0)), bdry, outlet)
ac_fluid_axis = DirichletBC(V_ALE.sub(0), Constant((0,0)), bdry, fluid_axis)
ac_wall = DirichletBC(V_ALE.sub(0), Constant((0, 0)), bdry, wall)
ac_int = DirichletBC(V_ALE.sub(0), u_s, bdry, circle)
bc_ALE = BlockDirichletBC([ac_inlet, ac_outlet, ac_fluid_axis, ac_wall, ac_int])

# F_a = I + grad(u_a)
# H_a = inv(F_a.T)
# sigma_a = -p_a * H_a + F_a - H_a

sigma_a = -p_a * I + grad(u_a) + grad(u_a).T

F_ALE = [-inner(sigma_a, grad(v_a)) * dx(fluid), (p_a - 1e1 * div(u_a)) * q_a * dx(fluid)]
J_ALE = block_derivative(F_ALE, X_ALE, Xt_ALE)

problem_ALE = BlockNonlinearProblem(F_ALE, X_ALE, bc_ALE, J_ALE)
solver_ALE = BlockPETScSNESSolver(problem_ALE)
solver_ALE.parameters.update(snes_solver_parameters["snes_solver"])


"""
    solve
"""

for n in range(N):

    solver.solve()
    solver_ALE.solve()
    ALE.move(mesh, u_s)
    ALE.move(mesh, u_a)
    eps.assign(e_all[n])

    # save(n)

    output_f.write(u_f, n)
    output_s.write(u_s, n)


print(U_0.vector()[:])
