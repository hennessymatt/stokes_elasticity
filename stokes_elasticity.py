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

U_0 = Constant(0.0)


# computational parameters
Nt = 100
dt = 1e-3


snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 8,
                                          "report": True,
                                          "error_on_nonconvergence": True}}

parameters["ghost_mode"] = "shared_facet"


# output = XDMFFile("output/" + fname + "drop_beam.xdmf")
# output.parameters["rewrite_function_mesh"] = False
# output.parameters["functions_share_mesh"] = True
# output.parameters["flush_output"] = True



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
Sig_w = generate_subdomain_restriction(mesh, subdomains, wall)

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
V = BlockFunctionSpace(mesh, mixed_element, restrict = [Of, Of, Os, Os, Sig, Sig_w])

# unknowns and test functions
Y = BlockTestFunction(V)
(v_f, q_f, v_s, q_s, eta, Txz) = block_split(Y)

Xt = BlockTrialFunction(V)

X = BlockFunction(V)
(u_f, p_f, u_s, p_s, lam, Sxz) = block_split(X)

# X_old = BlockFunction(V)
# (u_d_old, p_old, C_old, u_b_old, lam_old) = block_split(X_old)

#---------------------------------------------------------------------
# boundary conditions
#---------------------------------------------------------------------


far_field = Expression(('1 - x[1] * x[1] / 0.25 - V_a', '0'), degree=0, V_a = U_0)
bc_inlet = DirichletBC(V.sub(0), far_field, bdry, inlet)
bc_outlet = DirichletBC(V.sub(0), far_field, bdry, outlet)


bc_fluid_axis = DirichletBC(V.sub(0).sub(1), Constant(0), bdry, fluid_axis)

bcs = BlockDirichletBC([bc_inlet, bc_outlet, bc_fluid_axis])


#---------------------------------------------------------------------
# Define kinematics
#---------------------------------------------------------------------

I = Identity(2)

# fluids
sigma_f = -p_f * I + grad(u_f) + grad(u_f).T

# solids
F = inv(I - grad(u_s))
J = det(F)

sigma_s = Constant(1000) * (F * F.T - p_s * I)





#---------------------------------------------------------------------
# build equations
#---------------------------------------------------------------------

# Incompressibility conditions
ic_s = J - 1
ic_f = div(u_f)


# FUN1 = -inner(sigma_f, grad(v_f)) * dx(fluid) + inner(lam("+"), v_f("+")) * dS + Sxz * v_f[0] * ds(wall)
FUN1 = -inner(sigma_f, grad(v_f)) * dx(fluid) # + Sxz * v_f[0] * ds(wall)
FUN2 = ic_f * q_f * dx(fluid)

# FUN3 = -inner(sigma_s, grad(v_s)) * dx(solid) - inner(lam("-"), v_s("-")) * dS
FUN3 = inner(u_s, v_s) * dx(solid)
FUN4 = p_s * q_s * dx(solid)

FUN5 = inner(avg(eta), u_f("+") - as_vector([U_0, 0])) * dS

FUN6 = (u_f[0] - U_0) * Txz * ds(wall)

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
(u_f, p_f, u_s, p_s, lam, Sxz) = X.block_split()

solver.solve()
