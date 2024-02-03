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

"""

from dolfin import *
from multiphenics import *
from helpers import *
import numpy as np
from mesh_generation import *
import matplotlib.pyplot as plt

#---------------------------------------------------------------------
# Global variables
#---------------------------------------------------------------------

mr = Constant(0e-1)

# define the boundaries
circle = 1
fluid_axis = 2
inlet = 3
outlet = 4
wall = 5
solid_axis = 6

# define the domains
fluid = 7
solid = 8

#---------------------------------------------------------------------
# Setting up file names and paramerers
#---------------------------------------------------------------------

"""
Define file names
"""

# directory for file output
# dir = '/media/eg21388/data/fenics/stokes_elasticity/'
dir = '/home/simon/data/fenics/stokes_elasticity_axisym/'
# dir = '/home/matt/data/fenics/stokes_elasticity/'


output_eul_f = XDMFFile(dir + "eul_fluid.xdmf")
output_eul_f.parameters["rewrite_function_mesh"] = False
output_eul_f.parameters["functions_share_mesh"] = True
output_eul_f.parameters["flush_output"] = True

output_eul_s = XDMFFile(dir + "eul_solid.xdmf")
output_eul_s.parameters["rewrite_function_mesh"] = False
output_eul_s.parameters["functions_share_mesh"] = True
output_eul_s.parameters["flush_output"] = True

"""
    Physical parameters
"""

# the initial value of epsilon to try solving the problem with
eps = Constant(0.1)



"""
Solver parameters
"""
snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "absolute_tolerance": 1e-8,
                                          "error_on_nonconvergence": False}}

parameters["ghost_mode"] = "shared_facet"

parameters["form_compiler"]["quadrature_degree"] = 5
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True


def build_mesh(x_int, y_int):
    """
    Main function for mesh generation.  Calls the other functions
    """

    # create the mesh in pygmsh and save as xdmf
    ids = run_pygmsh(x_int, y_int)
    save_xdmf()

    # load in the .xdmf mesh
    mesh_file = XDMFFile(MPI.comm_world, "mesh/python_mesh.xdmf")
    mesh = Mesh()
    mesh_file.read(mesh)

    # create subdomains that separates fluid and solid domain
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile("mesh/python_mesh.xdmf") as infile:
        infile.read(mvc, "name_to_read")

    subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    # create bdry to separate different boundaries
    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile("mesh/python_facet_mesh.xdmf") as infile:
        infile.read(mvc, "name_to_read")

    bdry = cpp.mesh.MeshFunctionSizet(mesh, mvc)


    return mesh, subdomains, bdry


def ale_solve(eps_range, mesh, subdomains, bdry, n = 0, sol_eul = None):

    eps_ale = Constant(eps_range[0])

    # files for the fluid (f) and solid (s)
    output_ale_f = XDMFFile(dir + "ale_fluid.xdmf")
    output_ale_f.parameters["rewrite_function_mesh"] = True
    output_ale_f.parameters["functions_share_mesh"] = True
    output_ale_f.parameters["flush_output"] = True

    output_ale_s = XDMFFile(dir + "ale_solid.xdmf")
    output_ale_s.parameters["rewrite_function_mesh"] = True
    output_ale_s.parameters["functions_share_mesh"] = True
    output_ale_s.parameters["flush_output"] = True


    Of = generate_subdomain_restriction(mesh, subdomains, fluid)
    Os = generate_subdomain_restriction(mesh, subdomains, solid)
    Sig = generate_interface_restriction(mesh, subdomains, {fluid, solid})

    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
    ds = Measure("ds", domain=mesh, subdomain_data=bdry)
    dS = Measure("dS", domain=mesh, subdomain_data=bdry)
    dS = dS(circle)

    # print(f"Area of fluid: {assemble(Constant(1) * dx(fluid)):.2f}")
    # print(f"Area of solid: {assemble(Constant(1) * dx(solid)):.2f}")

    #---------------------------------------------------------------------
    # elements, function spaces, and test/trial functions
    #---------------------------------------------------------------------
    P2 = VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    DGT = VectorElement("DGT", mesh.ufl_cell(), 1)
    # DGT = VectorElement("CG", mesh.ufl_cell(), 1)
    P0 = FiniteElement("R", mesh.ufl_cell(), 0)

    mixed_element = BlockElement(P2, P1, DGT, P0, P0)
    Vf = BlockFunctionSpace(mesh, mixed_element, restrict=[Of, Of, Sig, Sig, Of])
    Vs = BlockFunctionSpace(mesh, BlockElement(P2, P1, P0), restrict=[Os, Os, Os])

    """
    Setting up the elements and solution: here is the notation:

    u_f/v_f: fluid velocity from Stokes equations
    p_f/q_f: fluid pressure from Stokes equations

    u_s/v_s: solid displacement from nonlinear elasticity
    p_s/q_s: solid pressure from nonlinear elasticity

    f_s/g_s: Lagrange multiplier corresponding to the force needed to pin the
    solid in place (should end up being zero since particle is free)

    lam/eta: Lagrange multiplier corresponding to the fluid traction
    acting on the solid
    
    U_0/V_0: the translational velocity of the solid

    lam_p/eta_p: Lagrange multiplier to ensure the mean fluid pressure is zero

    u_a/v_a: fluid "displacement" from the ALE method (e.g. how to deform
    the fluid geometry)

    lam_a: Lagrange multiplier to ensure continuity of fluid and solid
    displacement (ensures compatibility between fluid/solid domains)

    """

    # unknowns and test functions
    Xf = BlockFunction(Vf)
    (u_f, p_f, lam, U_0, lam_p) = block_split(Xf)
    Yf = BlockTestFunction(Vf)
    (v_f, q_f, eta, V_0, eta_p) = block_split(Yf)
    Xtf = BlockTrialFunction(Vf)

    Xs = BlockFunction(Vs)
    (u_s, p_s, f_s) = block_split(Xs)
    Ys = BlockTestFunction(Vs)
    (v_s, q_s, g_s) = block_split(Ys)
    Xts = BlockTrialFunction(Vs)

    # ALE
    V_ALE = BlockFunctionSpace(mesh, [P2], restrict=[Of])

    X_ALE = BlockFunction(V_ALE)
    (u_a,) = block_split(X_ALE)
    Y_ALE = BlockTestFunction(V_ALE)
    (v_a,) = block_split(Y_ALE)
    Xt_ALE = BlockTrialFunction(V_ALE)

    ##################

    # Placeholder for the last converged solution
    Xf_old = BlockFunction(Vf)
    # (u_f_old, p_f_old, lam_old, V_0_old, lam_p_old) = block_split(Xf_old)
    Xf_0 = BlockFunction(Vf)
    (u_f_0, p_f_0, lam_0, V_0_0, lam_p_0) = block_split(Xf_0)

    Xs_old = BlockFunction(Vs)
    # (u_s_old, p_s_old, f_s_old) = block_split(Xs_old)
    Xs_0 = BlockFunction(Vs)
    (u_s_0, p_s_0, f_s_0) = block_split(Xs_0)

    X_ALE_old = BlockFunction(V_ALE)
    # (u_a_old,) = block_split(X_ALE_old)
    X_ALE_0 = BlockFunction(V_ALE)
    (u_a_0,) = block_split(X_ALE_0)

    #---------------------------------------------------------------------
    # Physical boundary conditions
    #---------------------------------------------------------------------

    far_field = Expression(('(1 - x[1] * x[1] / 0.25) * t', '0'), degree=0, t=1)

    bc_inlet = DirichletBC(Vf.sub(0), far_field, bdry, inlet)
    bc_outlet = DirichletBC(Vf.sub(0), far_field, bdry, outlet)
    bc_fluid_axis = DirichletBC(Vf.sub(0).sub(1), Constant(0), bdry, fluid_axis)
    bc_wall = DirichletBC(Vf.sub(0), Constant((0, 0)), bdry, wall)

    bc_solid_axis = DirichletBC(Vs.sub(0).sub(1), Constant(0), bdry, solid_axis)

    bc_f = BlockDirichletBC([bc_inlet, bc_outlet, bc_fluid_axis, bc_wall])
    bc_s = BlockDirichletBC([bc_solid_axis])

    #---------------------------------------------------------------------
    # Define the model
    #---------------------------------------------------------------------

    I = Identity(2)

    """
    Extract Eulerian soln
    """
    if n > 0:
        u_s_eul = sol_eul[0]

    """
    Solids problem
    """

    if n > 0:
        u_s_0.interpolate(u_s_eul)
        F_0 = inv(I - grad(u_s_0))
        # F_0 = I
    else:
        F_0 = I

    # deformation gradient tensor
    F_1 = I + grad(u_s)
    H_1 = inv(F_1.T)

    F = F_1 * F_0
    H = inv(F.T)

    B = F * F.T

    # (non-dim) PK1 stress tensor and incompressibility condition
    sigma_s = 1 / eps_ale * ((B - I) - mr * (inv(B) - I)) - p_s * I
    Sigma_s = sigma_s * H_1
    # Sigma_s = 1 / eps * (F - H) - p_s * H
    ic_s = det(F_1) - 1

    """
    Fluids problem: mapping the current configuration to the
    initial configuration leads a different form of the
    incompressible Stokes equations
    """

    # Deformation gradient for the fluid
    F_a = I + grad(u_a)
    H_a = inv(F_a.T)

    # Jacobian for the fluid
    J_a = det(F_a)

    # PK1 stress tensor and incompressibility condition for the fluid
    sigma_f = J_a * (-p_f * I + grad(u_f) * H_a.T + H_a * grad(u_f).T) * H_a
    ic_f = div(J_a * inv(F_a) * u_f)


    """
    ALE problem: there are three different versions below
    """

    # Laplace
    # sigma_a = grad(u_a)

    # linear elasticity
    nu_a = Constant(0.1)
    E_a = 0.5 * (grad(u_a) + grad(u_a).T)
    sigma_a = nu_a / (1 + nu_a) / (1 - 2 * nu_a) * tr(E_a) * I + 1 / (1 + nu_a) * E_a

    # nonlinear elasticity
    # nu_a = Constant(0.1)
    # E_a = 0.5 * (F_a.T * F_a - I)
    # sigma_a = F_a * (nu_a / (1 + nu_a) / (1 - 2 * nu_a) * tr(E_a) * I + 1 / (1 + nu_a) * E_a)

    #---------------------------------------------------------------------
    # build equations
    #---------------------------------------------------------------------
    ez = as_vector([1,0])

    # Stokes equations for the fluid
    FUN1 = -inner(sigma_f, grad(v_f)) * dx(fluid) + inner(lam("+"), v_f("+")) * dS
    FUN2 = ic_f * q_f * dx(fluid) + lam_p * q_f * dx(fluid)

    # elasticity for the solid
    FUN3 = (-inner(sigma_s, grad(v_s)) * dx(solid) + inner(as_vector([f_s, 0]), v_s) * dx
            - inner(lam("-"), v_s("-")) * dS)
    FUN4 = ic_s * q_s * dx(solid)

    # Continuity of fluid velocity at the solid
    FUN5 = inner(avg(eta), u_f("+") - as_vector([U_0("+"), 0])) * dS
    # No total axial traction on the solid (ez . sigma_s . n = 0)
    FUN6 = dot(ez, lam("+")) * V_0("+") * dS

    # mean axial solid displacement is zero
    FUN7 = dot(ez, u_s) * g_s * dx(solid)
    # mean fluid pressure is zero
    FUN8 = p_f * eta_p * dx(fluid)

    # ALE problem
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


    # if n > 0:
    #     u_s.interpolate(u_s_eul)
    #     p_s.interpolate(p_s_eul)
    #     u_f.interpolate(u_f_eul)
    #     p_f.interpolate(p_f_eul)

    #if not(X_0) == None:
    #    block_assign(X, X_0)

    """
        Separate the meshes
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

    # Python functions to save solution for a given value
    # of epsilon
    def save_f(n):
        u_f_only = project(u_f, Vf)
        u_a_only = project(u_a, Vf)
        p_f_only = project(p_f, Vf0)

        u_f_only.rename("u_f", "u_f")
        u_a_only.rename("u_a", "u_a")
        p_f_only.rename("p_f", "p_f")

        output_ale_f.write(u_f_only, n)
        output_ale_f.write(u_a_only, n)
        output_ale_f.write(p_f_only, n)

    def save_s(n):
        u_s_only = project(u_s, Vs)
        u_s_only.rename("u_s", "u_s")
        output_ale_s.write(u_s_only, n)

        p_s_only = project(p_s, Vs0)
        p_s_only.rename("p_s", "p_s")
        output_ale_s.write(p_s_only, n)


    #---------------------------------------------------------------------
    # ALE BCs
    #---------------------------------------------------------------------
    ac_inlet = DirichletBC(V_ALE.sub(0), Constant((0, 0)), bdry, inlet)
    ac_outlet = DirichletBC(V_ALE.sub(0), Constant((0, 0)), bdry, outlet)
    ac_fluid_axis = DirichletBC(V_ALE.sub(0).sub(1), Constant((0)), bdry, fluid_axis)
    ac_wall = DirichletBC(V_ALE.sub(0).sub(1), Constant((0)), bdry, wall)
    ac_int = DirichletBC(V_ALE.sub(0), u_s, bdry, circle)

    bc_ALE = BlockDirichletBC([ac_inlet, ac_outlet, ac_fluid_axis, ac_wall, ac_int])

    problem_ALE = BlockNonlinearProblem(F_ALE, X_ALE, bc_ALE, J_ALE)
    solver_ALE = BlockPETScSNESSolver(problem_ALE)
    solver_ALE.parameters.update(snes_solver_parameters["snes_solver"])

    (u_a,) = X_ALE.block_split()

    #---------------------------------------------------------------------
    # Solve
    #---------------------------------------------------------------------

    print('solving fluid problem...')
    (its, conv_f) = solver_f.solve()
    save_f(0)
    save_s(0)
    print('f_0 =', f_s.vector()[:])
    print('zeta =', zeta.vector()[:])

    for e in eps_range:
        eps_ale.assign(e)

        print('-------------------------------------------------')
        print(f'solving problem with eps[{n:d}] = {e:.2e}')

        print('solving solid problem...')
        (its, conv_s) = solver_s.solve()

        save_s(n)

        if conv_s == False:
            block_assign(Xf, Xf_old)
            block_assign(Xs, Xs_old)
            block_assign(X_ALE, X_ALE_old)
            break
        else:
            block_assign(Xf_old, Xf)
            block_assign(Xs_old, Xs)
            block_assign(X_ALE_old, X_ALE)

        print('updating fluid geometry')
        (its, conv_ALE) = solver_ALE.solve()

        if conv_ALE == False:
            block_assign(Xf, Xf_old)
            block_assign(Xs, Xs_old)
            block_assign(X_ALE, X_ALE_old)
            break
        else:
            block_assign(Xf_old, Xf)
            block_assign(Xs_old, Xs)
            block_assign(X_ALE_old, X_ALE)

        if its > 4:
            break

        print('solving fluid problem...')
        (its, conv_f) = solver_f.solve()
        save_f(n)
        print('f_0 =', f_s.vector()[:])
        print('zeta =', zeta.vector()[:])

    x_new = np.array([x + u_s(x, y)[0] for (x, y) in zip(x_int, y_int)])
    y_new = np.array([y + u_s(x, y)[1] for (x, y) in zip(x_int, y_int)])


    return conv_ALE, x_new, y_new, float(U_0.vector()[0]), u_s, float(eps_ale)






#==============================================================================
# Define the Eulerian solver
#==============================================================================

def eulerian_solver(mesh, subdomains, bdry, U_0, u_s_ale = None):

    eps_eul = Constant(0.1)

    Of = generate_subdomain_restriction(mesh, subdomains, fluid)
    Os = generate_subdomain_restriction(mesh, subdomains, solid)
    Sig = generate_interface_restriction(mesh, subdomains, {fluid, solid})

    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
    ds = Measure("ds", domain=mesh, subdomain_data=bdry)
    dS = Measure("dS", domain=mesh, subdomain_data=bdry)
    dS = dS(circle)

    #---------------------------------------------------------------------
    # elements, function spaces, and test/trial functions
    #---------------------------------------------------------------------
    P2 = VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    DGT = VectorElement("DGT", mesh.ufl_cell(), 1)
    P0 = FiniteElement("R", mesh.ufl_cell(), 0)

    mixed_element = BlockElement(P2, P1, DGT, P0)
    Vf = BlockFunctionSpace(mesh, mixed_element, restrict = [Of, Of, Sig, Of])
    Vs = BlockFunctionSpace(mesh, BlockElement(P2, P1, P0), restrict=[Os, Os, Os])


    # unknowns and test functions


    Xf = BlockFunction(Vf)
    Xs = BlockFunction(Vs)
    Ys = BlockTestFunction(Vs)
    Yf = BlockTestFunction(Vf)

    (u_f, p_f, lam, lam_p) = block_split(Xf)
    (v_f, q_f, eta, eta_p) = block_split(Yf)
    (u_s, p_s, f_s) = block_split(Xs)
    (v_s, q_s, g_s) = block_split(Ys)

    Xtf = BlockTrialFunction(Vf)
    Xts = BlockTrialFunction(Vs)



    #---------------------------------------------------------------------
    # boundary conditions
    #---------------------------------------------------------------------

    far_field = Expression(('(1 - x[1] * x[1] / 0.25) * t', '0'), degree=0, t=1)

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

    # fluids
    sigma_f = -p_f * I + grad(u_f) + grad(u_f).T
    ic_f = div(u_f)

    # solids
    F = inv(I - grad(u_s))
    B = F * F.T
    sigma_s = 1 / eps_eul * ((B - I) - mr * (inv(B) - I)) - p_s * I
    ic_s = det(F) - 1

    #---------------------------------------------------------------------
    # build equations
    #---------------------------------------------------------------------

    ez = as_vector([1,0])

    FUN1 = -inner(sigma_f, grad(v_f)) * dx(fluid) + inner(lam("+"), v_f("+")) * dS
    FUN2 = ic_f * q_f * dx(fluid) + lam_p * q_f * dx(fluid)

    FUN3 = -inner(sigma_s, grad(v_s)) * dx(solid) + inner(as_vector([f_s, 0]), v_s) * dx - inner(lam("-"), v_s("-")) * dS
    FUN4 = ic_s * q_s * dx(solid)

    FUN5 = inner(avg(eta), u_f("+") - as_vector([Constant(U_0), 0])) * dS

    FUN7 = dot(ez, u_s) * g_s * dx(solid)
    FUN8 = p_f * eta_p * dx(fluid)

    FUN_f = [FUN1, FUN2, FUN5, FUN8]
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
    (u_f, p_f, lam, lam_p) = Xf.block_split()
    (u_s, p_s, f_s) = Xs.block_split()

    # class InitialSolidGuess(UserExpression):
    #     def eval(self, values, x):
    #         values[0] = 1e4
    #         values[1] = 1e4

    #     def value_shape(self):
    #         return (2,)

    # u_s.interpolate(InitialSolidGuess())
    # u_s.interpolate(Constant((1e5, 1e5)))
    # p_s.interpolate(Constant(1e5))


    # """
    #     quantities for separate meshes
    # """
    # mesh_f = SubMesh(mesh, subdomains, fluid)
    # mesh_s = SubMesh(mesh, subdomains, solid)
    # Vf = VectorFunctionSpace(mesh_f, "CG", 1)
    # Vs = VectorFunctionSpace(mesh_s, "CG", 1)
    # Vf0 = FunctionSpace(mesh_f, "CG", 1)
    # Vs0 = FunctionSpace(mesh_s, "CG", 1)

    # u_f_only = Function(Vf)
    # u_s_only = Function(Vs)
    # p_s_only = Function(Vs0)


    # def save_f(n):
    #     u_f_only = project(u_f, Vf)
    #     p_f_only = project(p_f, Vf0)

    #     u_f_only.rename("u_f", "u_f")
    #     p_f_only.rename("p_f", "p_f")

    #     output_eul_f.write(u_f_only, n)
    #     output_eul_f.write(p_f_only, n)

    # def save_s(n):
    #     u_s_only = project(u_s, Vs)
    #     u_s_only.rename("u_s", "u_s")
    #     output_eul_s.write(u_s_only, n)

    #     p_s_only = project(p_s, Vs0)
    #     p_s_only.rename("p_s", "p_s")
    #     output_eul_s.write(p_s_only, n)


    """
        solve
    """

    print('solving Eulerian fluid problem...')
    (its, conv_f) = solver_f.solve()
    (its, conv_s) = solver_s.solve()

    e_all = np.linspace(0.1, float(eps), 5)
    for e in e_all:
        eps_eul.assign(e)
        print(f'solving Eulerian solid problem with eps = {float(eps_eul):.2f}')
        (its, conv_s) = solver_s.solve()

    # save_f(0)
    # save_s(0)

    return u_s, p_s, u_f, p_f




#==================================================================

# generate points on the circle
R = 0.2
N_int = 40
theta = np.flip(np.linspace(0, np.pi, N_int))
x_int = R * np.cos(theta)
y_int = R * np.sin(theta)

x_int_circle = R * np.cos(theta)
y_int_circle = R * np.sin(theta)

# create mesh
mesh, subdomains, bdry = build_mesh(x_int, y_int)

# first solve of the ALE problem
eps_inc = 0.05
eps_range = eps_inc + np.linspace(0, eps_inc, 15)
conv, x_int, y_int, U_0, u_s_ale, eps_term = ale_solve(eps_range, mesh, subdomains, bdry)

if conv == True:

    for n in range(10):

        eps.assign(eps_range[-1])

        mesh, subdomains, bdry = build_mesh(x_int, y_int)
        sol_eul = eulerian_solver(mesh, subdomains, bdry, U_0, u_s_ale)

        # Test we recover the original circular shape
        plt.figure(2*n)
        mesh_s = SubMesh(mesh, subdomains, solid)
        plot(mesh_s)

        u_0_eul = -sol_eul[0]
        Vs = VectorFunctionSpace(mesh_s, "CG", 1)
        u_s_plot = project(u_0_eul, Vs)
        ALE.move(mesh_s, u_s_plot)
        plt.figure(2*n+1)
        plot(mesh_s)
        plt.plot(x_int_circle, y_int_circle)

        eps_range = eps_range + eps_inc
        conv_ale, x_int, y_int, U_0, u_s_ale, eps_term = ale_solve(eps_range, mesh, subdomains, bdry, n=1, sol_eul=sol_eul)

        if not(conv_ale):
            break
plt.show()
# eps_inc = 0.05
# for e in np.arange(0.2, 1, eps_inc):

#     mesh, subdomains, bdry = build_mesh(x_int, y_int)
#     sol_eul = eulerian_solver(mesh, subdomains, bdry, U_0, u_s_ale)
#     # x_int, y_int, U_0, u_s_ale, X_0 = ale_solve(mesh, subdomains, bdry, n = 1, sol_eul = sol_eul)

#     eps.assign(e)
#     print('================================================')
#     print(f'solving with eps = {e:.2f}')
#     conv_ale, x_int, y_int, U_0, u_s_ale, X_0 = ale_solve(mesh, subdomains, bdry, n = 1, sol_eul = sol_eul)

#     if conv_ale == False:
#         break

# mesh, subdomains, bdry = build_mesh(x_int, y_int)
# u_s = eulerian_solver(mesh, subdomains, bdry, U_0)

# x_int, y_int, U_0, u_ale_s = ale_solve(1, x_int, y_int)