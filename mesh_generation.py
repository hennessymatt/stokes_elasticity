import gmsh
import pygmsh
import meshio



def run_pygmsh(x_int, y_int):
    """
    Creates a mesh for the problem using pygmsh (a Python implementation
    of gmsh).  

    Inputs:
    -   x_int, y_int: positions of the fluid-solid interace

    Outputs:
    -   The function saves a .msh file
    -   The function returns a dictionary with info about the
        numerical ids of the domains and boundaries

    """

    # number of interface points
    N_int = len(x_int)

    # length and height of the domain
    L = 10
    H = 0.5

    # element size away from particle
    res = 0.1 / 2

    # element size at interface and in the solid
    res_int = res / 6 / 2

    
    geometry = pygmsh.geo.Geometry()
    model = geometry.__enter__()

    points = [
        model.add_point((x, y, 0), mesh_size = res_int) for (x, y) in zip(x_int, y_int)
    ]

    # corners of the fluid domain
    points.append(model.add_point((L/2, 0, 0), mesh_size = res))
    points.append(model.add_point((L/2, H, 0), mesh_size = res))
    points.append(model.add_point((-L/2, H, 0), mesh_size = res))
    points.append(model.add_point((-L/2, 0, 0), mesh_size = res))

    
    # create outline of the fluid domain
    lines = [
        model.add_line(points[i], points[i+1]) for i in range(len(points)-1)
    ]
    lines.append(model.add_line(points[-1], points[0]))

    # create line for solid axis
    lines.append(model.add_line(points[N_int-1], points[0]))

    # curve loop for fluid domain
    loop_f = model.add_curve_loop(lines[:-1])

    # curve loop for solid domain
    tmp = [lines[n] for n in range(N_int-1)]
    tmp.append(lines[-1])
    loop_s = model.add_curve_loop(tmp)

    # create surfaces for domains
    surface_f = model.add_plane_surface(loop_f)
    surface_s = model.add_plane_surface(loop_s)

    
    model.synchronize()

    # create labelled domains and boundaries
    model.add_physical([lines[n] for n in range(N_int-1)], "interface")
    model.add_physical([lines[-2], lines[N_int-1]], "fluid_axis")
    model.add_physical([lines[-3]], "inlet")
    model.add_physical([lines[N_int]], "outlet")
    model.add_physical([lines[N_int+1]], "wall")
    model.add_physical([lines[-1]], "solid_axis")

    model.add_physical([surface_f], "fluid")
    model.add_physical([surface_s], "solid")

    # create a dict that stores ids of the boundaries and domains
    domains = ["interface", 
               "fluid_axis", 
               "inlet", 
               "outlet", 
               "wall",
               "solid_axis",
               "fluid",
               "solid"]
    
    ids = {}
    for n in range(len(domains)):
        ids[domains[n]] = n+1


    mesh = geometry.generate_mesh(dim=2)
    gmsh.write("mesh/python_mesh.msh")
    gmsh.clear()
    geometry.__exit__()

    return ids


def create_mesh(mesh, cell_type, prune_z=False):
    """
    Helper function used for converting a .msh mesh
    into a .xdmf mesh
    """
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
                           "name_to_read": [cell_data]})
    return out_mesh


def save_xdmf():
    """
    Loads a .msh file created using run_pygmsh and converts it 
    into two xdmf files, one for the domains and another for
    the boundaries
    """

    # load .msh mesh from file
    mesh_from_file = meshio.read("mesh/python_mesh.msh")

    # find the boundaries in the mesh and save to a file
    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write("mesh/python_facet_mesh.xdmf", line_mesh)

    # find the domains in the mesh and save to a file
    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write("mesh/python_mesh.xdmf", triangle_mesh)