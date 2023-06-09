from dolfin import *
from multiphenics import *
from helpers import *
# import numpy as np
# import json
# from pathlib import Path

meshname = 'channel_sphere'

# fname = 'dynamic/' + meshname + '/Pe_huge/E_one/'

"""
    parameters
"""

# computational parameters
Nt = 100
dt = 1e-3


snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 8,
                                          "report": True,
                                          "error_on_nonconvergence": True}}

parameters["ghost_mode"] = "shared_facet"


output = XDMFFile("output/" + "stokes.xdmf")
output.parameters["rewrite_function_mesh"] = False
output.parameters["functions_share_mesh"] = True
output.parameters["flush_output"] = True



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

mixed_element = BlockElement(P2, P1, DGT, P0)
V = BlockFunctionSpace(mesh, mixed_element, restrict = [Of, Of, Sig, Sig])

# unknowns and test functions
Y = BlockTestFunction(V)
(v_f, q_f, eta, V_0) = block_split(Y)

Xt = BlockTrialFunction(V)

X = BlockFunction(V)
(u_f, p_f, lam, U_0) = block_split(X)

# X_old = BlockFunction(V)
# (u_d_old, p_old, C_old, u_b_old, lam_old) = block_split(X_old)

#---------------------------------------------------------------------
# boundary conditions
#---------------------------------------------------------------------


far_field = Expression(('1 - x[1] * x[1] / 0.25 - V_a', '0'), degree=0, V_a = 0)

bc_inlet = DirichletBC(V.sub(0), far_field, bdry, inlet)
bc_outlet = DirichletBC(V.sub(0), far_field, bdry, outlet)
bc_fluid_axis = DirichletBC(V.sub(0).sub(1), Constant(0), bdry, fluid_axis)
bc_wall = DirichletBC(V.sub(0), Constant((0, 0)), bdry, wall)


bcs = BlockDirichletBC([bc_inlet, bc_outlet, bc_fluid_axis, bc_wall])


#---------------------------------------------------------------------
# Define kinematics
#---------------------------------------------------------------------

I = Identity(2)

# fluids
sigma_f = -p_f * I + grad(u_f) + grad(u_f).T


#---------------------------------------------------------------------
# build equations
#---------------------------------------------------------------------

# Incompressibility conditions
ic_f = div(u_f)


FUN1 = -inner(sigma_f, grad(v_f)) * dx(fluid) + inner(lam("+"), v_f("+")) * dS
FUN2 = ic_f * q_f * dx(fluid)

FUN5 = inner(avg(eta), u_f("+") - as_vector([U_0("+"), 0])) * dS

FUN6 = dot(as_vector([1, 0]), lam("+")) * V_0("+") * dS

FUN = [FUN1, FUN2, FUN5, FUN6]

JAC = block_derivative(FUN, X, Xt)


#---------------------------------------------------------------------
# set up the solver
#---------------------------------------------------------------------

# Initialize solver
problem = BlockNonlinearProblem(FUN, X, bcs, JAC)
solver = BlockPETScSNESSolver(problem)
solver.parameters.update(snes_solver_parameters["snes_solver"])

# extract solution components
(u_f, p_f, lam, U_0) = X.block_split()

solver.solve()

output.write(u_f, 0)

print(U_0.vector()[:])
