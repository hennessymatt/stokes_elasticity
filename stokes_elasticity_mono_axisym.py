"""
This is a monolithic solver for pressure-driven flow around a
neo-Hookean particle.  The particle is **free** to move with
the flow.  However, equations are formulated in a moving
frame that travels with the particle

The problem is formulated using the ALE
method, which maps the deformed geometry to the initial
geometry.  The problem is solved using the
initial geometry; the deformed geometry can by using the
WarpByVector filter in Paraview using the displacement
computed in the fluid domain.

The problem uses Lagrange multipliers to ensure the centre of
mass of the particle remains fixed as well as to impose
continuity of stress

The code works by initially solving the problem with a small
value of epsilon (ratio of fluid stress to elastic stiffness)
and then gradually ramping up epsilon.  If convergence is
not obtained then the code tries again using a smaller value
of epsilon.

This code implements to axisymmetric verion, with changes to
the deformation gradient tensor, determinant, and divergence.

"""

from dolfin import *
from multiphenics import *
from helpers import *

# Python function for axisymmetric divergence
def diva(vec):
    return div(vec) + vec[1]/r

# ---------------------------------------------------------------------
# Setting up file names and paramerers
# ---------------------------------------------------------------------

"""
Define file names
"""

# name of the mesh found in mesh/
meshname = 'channel_sphere'

# directory for file output
# dir = '/media/eg21388/data/fenics/stokes_elasticity/'
dir = '/home/simon/data/fenics/stokes_elasticity_axisym/'

# files for the fluid (f) and solid (s)
output_f = XDMFFile(dir + "fluid.xdmf")
output_f.parameters["rewrite_function_mesh"] = False
output_f.parameters["functions_share_mesh"] = True
output_f.parameters["flush_output"] = True

output_s = XDMFFile(dir + "solid.xdmf")
output_s.parameters["rewrite_function_mesh"] = False
output_s.parameters["functions_share_mesh"] = True
output_s.parameters["flush_output"] = True

"""
    Physical parameters
"""

# the initial value of epsilon to try solving the problem with
eps_try = 0.1
eps = Constant(eps_try)

# the max value of epsilon
eps_max = 1

# the incremental increase in epsilon
de = 0.05

# the min and max values of the increments to make.
de_min = 1e-3
de_max = 1e-1

"""
Solver parameters
"""
snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "absolute_tolerance": 1e-8,
                                          "error_on_nonconvergence": False}}

parameters["ghost_mode"] = "shared_facet"

# ---------------------------------------------------------------------
# problem geometry: mesh and boundaries
# ---------------------------------------------------------------------
mesh = Mesh('mesh/' + meshname + '.xml')
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
nn = FacetNormal(mesh);
tt = as_vector((-nn[1], nn[0]))

ez = as_vector([1, 0])
ex = as_vector([0, 1])

# define the surface differential on the circle
x = SpatialCoordinate(mesh)
r = x[1]

# ---------------------------------------------------------------------
# elements, function spaces, and test/trial functions
# ---------------------------------------------------------------------
P2 = VectorElement("CG", mesh.ufl_cell(), 2)
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
# DGT = VectorElement("DGT", mesh.ufl_cell(), 1)
DGT = VectorElement("CG", mesh.ufl_cell(), 1)
P0 = FiniteElement("R", mesh.ufl_cell(), 0)

"""
Setting up the elements and solution: here is the notation:

u_f: fluid velocity from Stokes equations
p_f: fluid pressure from Stokes equations

u_s: solid displacement from nonlinear elasticity
p_s: solid pressure from nonlinear elasticity

f_0: Lagrange multiplier corresponding to the force needed to pin the
solid in place (should end up being zero since particle is free)

U_0: the translational velocity of the solid

lam: Lagrange multiplier corresponding to the fluid traction 
acting on the solid

lam_p: Lagrange multiplier to ensure the mean fluid pressure is zero

u_a: fluid "displacement" from the ALE method (e.g. how to deform
the fluid geometry)

lam_a: Lagrange multiplier to ensure continuity of fluid and solid
displacement (ensures compatibility between fluid/solid domains)

"""

mixed_element = BlockElement(P2, P1, P2, P1, P0, P0, DGT, P0, P2, DGT)
V = BlockFunctionSpace(mesh, mixed_element, restrict=[Of, Of, Os, Os, Os, Of, Sig, Of, Of, Sig])

X = BlockFunction(V)
(u_f, p_f, u_s, p_s, f_0, U_0, lam, lam_p, u_a, lam_a) = block_split(X)

# unknowns and test functions
Y = BlockTestFunction(V)
(v_f, q_f, v_s, q_s, g_0, V_0, eta, eta_p, v_a, eta_a) = block_split(Y)

Xt = BlockTrialFunction(V)

# Placeholder for the last converged solution
X_old = BlockFunction(V)

# ---------------------------------------------------------------------
# boundary conditions
# ---------------------------------------------------------------------

"""
Physical boundary conditions
"""

# Far-field fluid velocity
far_field = Expression(('(1 - x[1] * x[1] / 0.25) * t', '0'), degree=0, t=1)

# impose the far-field fluid velocity upstream and downstream
bc_inlet = DirichletBC(V.sub(0), far_field, bdry, inlet)
bc_outlet = DirichletBC(V.sub(0), far_field, bdry, outlet)

# impose the fluid pressure the outlet
# bc_outlet = DirichletBC(V.sub(1), Constant(0), bdry, outlet)

# impose no vertical fluid flow at the centreline axis
bc_fluid_axis = DirichletBC(V.sub(0).sub(1), Constant(0), bdry, fluid_axis)

# impose no-slip and no-penetration at the channel wall
bc_wall = DirichletBC(V.sub(0), Constant((0, 0)), bdry, wall)

# impose zero vertical solid displacement at the centreline axis
bc_solid_axis = DirichletBC(V.sub(2).sub(1), Constant(0), bdry, solid_axis)

"""
Boundary conditions for the ALE problem for fluid 
displacement.  These are no normal displacements
"""

ac_inlet = DirichletBC(V.sub(8).sub(0), Constant((0)), bdry, inlet)
ac_outlet = DirichletBC(V.sub(8).sub(0), Constant((0)), bdry, outlet)
ac_fluid_axis = DirichletBC(V.sub(8).sub(1), Constant((0)), bdry, fluid_axis)
ac_wall = DirichletBC(V.sub(8).sub(1), Constant((0)), bdry, wall)

# Combine all BCs together
bcs = BlockDirichletBC(
    [bc_inlet, bc_outlet, bc_fluid_axis, bc_wall, bc_solid_axis, ac_inlet, ac_outlet, ac_fluid_axis, ac_wall])

# ---------------------------------------------------------------------
# Define the model
# ---------------------------------------------------------------------

I = Identity(2)

"""
Solids problem
"""
# deformation gradient tensor
F = I + grad(u_s)
H = inv(F.T)

# (non-dim) PK1 stress tensor and incompressibility condition
Sigma_s = 1 / eps * (F - H) - p_s * H
ic_s = det(F) * (1 + u_s[1] / r) - 1

"""
Fluids problem: mapping the current configuration to the 
initial configuration leads a different form of the
incompressible Stokes equations
"""

# Deformation gradient for the fluid
F_a = I + grad(u_a)
H_a = inv(F_a.T)

# Jacobian for the fluid
J_a = det(F_a) * (1 + u_a[1] / r)

# PK1 stress tensor and incompressibility condition for the fluid
Sigma_f = J_a * (-p_f * I + grad(u_f) * H_a.T + H_a * grad(u_f).T) * H_a
ic_f = diva(J_a * inv(F_a) * u_f)

"""
ALE problem: there are three different versions below
"""

# Laplace
# sigma_a = grad(u_a)

# linear elasticity
nu_a = Constant(0.1)
E_a = 0.5 * (grad(u_a) + grad(u_a).T)
sigma_a = nu_a / (1 + nu_a) / (1 - 2 * nu_a) * diva(u_a) * I + 1 / (1 + nu_a) * E_a

# nonlinear elasticity
# nu_a = Constant(0.48)
# E_a = 0.5 * (F_a.T * F_a - I)
# sigma_a = F_a * (nu_a / (1 + nu_a) / (1 - 2 * nu_a) * tr(E_a) * I + 1 / (1 + nu_a) * E_a)


# ---------------------------------------------------------------------
# build equations
# ---------------------------------------------------------------------

# Stokes equations for the fluid
FUN1 = (-inner(Sigma_f, grad(v_f)) * r * dx(fluid)
        - J_a * (2 * u_f[1] * v_f[1]/(r + u_a[1]) ** 2 - p_f * v_f[1] / r * (r / (r + u_a[1]))) * r * dx(fluid)
        + inner(lam("+"), v_f("+")) * r("+") * dS)

# Incompressibility for the fluid
FUN2 = ic_f * q_f * r * dx(fluid) + lam_p * q_f * r * dx(fluid)

# Nonlinear elasticity for the solid
FUN3 = (-inner(Sigma_s, grad(v_s)) * r * dx(solid) - 1 / eps * (1 + u_s[1] / r) * v_s[1] * dx(solid)
        + (1 / eps + p_s) * v_s[1] * (r / (r + u_s[1])) * dx(solid)
        + inner(as_vector([f_0, 0]), v_s) * r * dx(solid) - inner(lam("-"), v_s("-")) * r("+") * dS)

# Incompressibility for the solid
FUN4 = ic_s * q_s * r * dx(solid)

# Continuity of fluid velocity at the solid
FUN5 = inner(avg(eta), u_f("+") - as_vector([U_0("+"), 0])) * r("+") * dS

# No total axial traction on the solid (ez . sigma_s . n = 0)
FUN6 = dot(ez, lam("+")) * V_0("+") * r("+") * dS

# ALE bulk equation
FUN7 = (-inner(sigma_a, grad(v_a)) * r * dx(fluid)
        - 1 / (1 + nu_a) * u_a[1] * v_a[1] / r * dx(fluid)
        + inner(lam_a("+"), v_a("+")) * r("+") * dS)

# Continuity of fluid and solid displacement
FUN8 = inner(avg(eta_a), u_a("+") - u_s("-")) * r("+") * dS

# mean axial solid displacement is zero
FUN9 = dot(ez, u_s) * g_0 * r * dx(solid)

# mean fluid pressure is zero
FUN10 = p_f * eta_p * r * dx(fluid)

# Combine equations and compute Jacobian
FUN = [FUN1, FUN2, FUN3, FUN4, FUN5, FUN6, FUN7, FUN8, FUN9, FUN10]
JAC = block_derivative(FUN, X, Xt)

# ---------------------------------------------------------------------
# set up the solver
# ---------------------------------------------------------------------

# Initialize solver
problem = BlockNonlinearProblem(FUN, X, bcs, JAC)
solver = BlockPETScSNESSolver(problem)
solver.parameters.update(snes_solver_parameters["snes_solver"])

# extract solution components
(u_f, p_f, u_s, p_s, f_0, U_0, lam, lam_p, u_a, lam_a) = X.block_split()

# ---------------------------------------------------------------------
# Set up code to save solid quanntities only on the solid domain and
# fluid quantities only on the fluid domain
# ---------------------------------------------------------------------

"""
    Separate the meshes
"""
mesh_f = SubMesh(mesh, subdomains, fluid)
mesh_s = SubMesh(mesh, subdomains, solid)

# Create function spaces for the velocity and displacement
Vf = VectorFunctionSpace(mesh_f, "CG", 1)
Vs = VectorFunctionSpace(mesh_s, "CG", 1)

u_f_only = Function(Vf)
u_a_only = Function(Vf)
u_s_only = Function(Vs)


# Python function to save solution for a given value
# of epsilon
def save(eps):
    u_f_only = project(u_f, Vf)
    u_a_only = project(u_a, Vf)
    u_s_only = project(u_s, Vs)

    u_f_only.rename("u_f", "u_f")
    u_a_only.rename("u_a", "u_a")
    u_s_only.rename("u_s", "u_s")

    output_f.write(u_f_only, eps)
    output_f.write(u_a_only, eps)
    output_s.write(u_s_only, eps)


# ---------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------

n = 0

# last converged value of epsilon
eps_conv = 0

# increment epsilon and solve
while eps_conv <= eps_max:

    print('-------------------------------------------------')
    print(f'attempting to solve problem with eps = {float(eps):.4e}')

    # make a prediction of the next solution based on how the solution
    # changed over the last two increments, e.g. using a simple
    # extrapolation formula
    if n > 1:
        X.block_vector()[:] = X_old.block_vector()[:] + (eps_try - eps_conv) * dX_deps

    (its, conv) = solver.solve()

    # if the solver converged...
    if conv:

        n += 1
        # update value of eps_conv and save
        eps_conv = float(eps)
        save(eps_conv)

        # update the value of epsilon to try to solve the problem with
        eps_try += de

        # copy the converged solution into old solution
        block_assign(X_old, X)

        # print some info to the screen
        print('Axial force on the particle: f_0 =', f_0.vector()[0])
        print('Translational speed of the particle: U_0 =', U_0.vector()[0])

        # approximate the derivative of the solution wrt epsilon
        if n > 0:
            dX_deps = (X.block_vector()[:] - X_old.block_vector()[:]) / de

    # if the solver diverged...
    if not (conv):

        # halve the increment in epsilon if not at smallest value
        # and use the previously converged solution as the initial
        # guess
        if de > de_min:
            de /= 2
            eps_try = eps_conv + de
            block_assign(X, X_old)
        else:
            print('min increment reached...aborting')
            save(eps_try)
            break

    # update the value of epsilon
    eps.assign(eps_try)
