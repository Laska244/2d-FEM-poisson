from typing import Callable
import gmsh
import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.tri 
import matplotlib.pyplot
import importlib.util
import tkinter as tk
from tkinter import filedialog


def generate_mesh(input_coordinates: list | np.ndarray, mesh_size: float = 0.1) -> None:
    """
    prejde cyklom vsetky zadane body a vytvori im Point s tagom rovnym jeho poradiu
    a predpisanou velkostou siete mesh_density.
    potom medzi tieto points prida Lines s tagmi rovnymi tomu mensiemu bodu (ciara medzi bodmi 3 a 4 ma tag 3)
    potom spusti prikaz generate
    """
    gmsh.model.add("mesh")

    # prerobime vsetky zadane vrcholy na body, i je counter (na konci teda mame i-1 points a lines)
    i = 1
    for row in input_coordinates:
        gmsh.model.geo.addPoint(row[0], row[1], 0, meshSize=mesh_size, tag=i)
        i += 1

    # pridame medzi ne lines v poradi v akom boli na vstupe
    # prvych i-2 lines
    for j in range(1, i - 1):
        gmsh.model.geo.addLine(j, j + 1, j)
    # posledna line
    gmsh.model.geo.addLine(i - 1, 1, i - 1)

    # pridavam curve loop, plane surface, synchronizujem a generujem mesh
    a = list(range(1, i))
    gmsh.model.geo.addCurveLoop(a, 1)
    gmsh.model.geo.addPlaneSurface([1])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)


def extract_mesh_data() -> tuple[np.ndarray, dict[int, int], np.ndarray, int]:
    """
    z hotoveho meshu aktualneho modelu (my mame iba jeden model, a to "mesh") 
    vytiahne data o nodes, pripravi maticu ich suradnic,
    prekladovy slovnik tag to index,
    a maticu elementov 
    """
    # tymto kazdemu uzlu priradujeme kvalitu "index", ktora znamena, na ktorom indexe v matici su jeho suradnice
    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes(2, -1, True, False)
    node_coordinates = np.reshape(node_coordinates, (-1, 3))
    node_coordinates = np.delete(node_coordinates, 2, axis=1)
    pocet_uzlov = len(node_tags)
   
    # vytvarame slovnik ktory pomoze najst index uzla na zaklade jeho tagu
    tag_to_index = dict()
    ii = 0
    for tag in node_tags:
        tag_to_index[int(tag)] = ii
        ii += 1

    _, _, elem_node_tags = gmsh.model.mesh.getElements(2, 1)
    elements3 = np.reshape(elem_node_tags[0], (-1, 3))
    # elements3 je matica ktora ma v riadkoch trojice tagov nodes, ktore tvoria jeden element

    return node_coordinates, tag_to_index, elements3, pocet_uzlov


def assemble_system(
    pocet_uzlov: int,
    elements3: np.ndarray,
    node_coordinates: np.ndarray,
    tag_to_index: dict[int, int],
    f: Callable[[float, float], float]
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    assembly riedkej matice A a casti vektora b; objemove integraly
    """
    A = sp.dok_matrix((pocet_uzlov, pocet_uzlov))  # pouzivame format dok na assembly matice A
    b = np.zeros(pocet_uzlov)

    # konstantna matica, ktoru pouzivam na zostavenie matice M
    G2 = np.array([[0, 0], [1, 0], [0, 1]])

    for row in elements3: 
        # ziskanie suradnic uzlov elementu
        tag1, tag2, tag3 = row[0], row[1], row[2]
        x1 = node_coordinates[tag_to_index[tag1]][0]
        x2 = node_coordinates[tag_to_index[tag2]][0]
        x3 = node_coordinates[tag_to_index[tag3]][0]
        y1 = node_coordinates[tag_to_index[tag1]][1]
        y2 = node_coordinates[tag_to_index[tag2]][1]
        y3 = node_coordinates[tag_to_index[tag3]][1]

        # zostavenie matice M
        area = abs(((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2)  # obsah realny, nie orientovany
        G1 = np.array([np.ones(3), [x1, x2, x3], [y1, y2, y3]])
        G1_inverse = np.linalg.inv(G1)
        M = area * (G1_inverse @ G2 @ G2.T @ G1_inverse.T)

        # pricitanie prvkov matice M do matice A, ktoru skladam podla indexov, nie tagov
        index1 = tag_to_index[tag1]
        index2 = tag_to_index[tag2]
        index3 = tag_to_index[tag3]

        A[index1, index1] += M[0, 0]
        A[index2, index2] += M[1, 1]
        A[index3, index3] += M[2, 2]

        A[index1, index2] += M[0, 1]
        A[index2, index1] += M[1, 0]

        A[index1, index3] += M[0, 2]
        A[index3, index1] += M[2, 0]

        A[index2, index3] += M[1, 2]
        A[index3, index2] += M[2, 1]

        # takisto vektor b skladame podla indexov
        centre_of_mass = ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)
        add = 1 / 3 * area * f(*centre_of_mass)
        b[index1] += add
        b[index2] += add
        b[index3] += add

    return A.tocsr(), b         #maticu A konvertujem do csr formatu kvoli rieseniu sustavy neskor


def apply_neumann(
    b: np.ndarray,
    neumann_conditions: list,
    node_coordinates: np.ndarray,
    tag_to_index: dict[int, int]
) -> np.ndarray:
    """
    zoberie neumann conditions, prejde ich forcyklom a kazdu podmienku uplne spracuje a pricitat k vektoru b spravnu hodnotu
    """
    # cyklus ktory pricituje k vektoru b
    for row in neumann_conditions:
        line_tag = row[0]
        function = row[1]
        _, _, elem_node_tags = gmsh.model.mesh.getElements(1, line_tag)
        tags_array = elem_node_tags[0]
        for i in range(len(tags_array) // 2):
            tag1, tag2 = tags_array[2 * i], tags_array[2 * i + 1]

            # ziskanie spravnych indexov do matic
            index1 = tag_to_index[tag1]
            index2 = tag_to_index[tag2]

            # suradnice vytahujeme podla indexu
            x1, x2 = node_coordinates[index1][0], node_coordinates[index2][0]
            y1, y2 = node_coordinates[index1][1], node_coordinates[index2][1]

            centre_of_mass = (x1 + x2) / 2, (y1 + y2) / 2
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            # pricitanie do vektoru b
            b[index1] += function(*centre_of_mass) * distance / 2
            b[index2] += function(*centre_of_mass) * distance / 2

    return b


def apply_dirichlet(
    pocet_uzlov: int,
    dirichlet: list,
    tag_to_index: dict[int, int]
) -> tuple[np.ndarray, list[int]]:
    """
    prejde dirichlet podmienky, zlozi z nich znamu cast vektora U
    druhy cyklus zarucuje, ze na hranicnych uzloch bude podmienka aritmetickym priemerom dvoch "bijucich sa" podmienok
    druhy cyklus taktiez zapisuje do zoznamu ktore indexy su "dirichletove", pretoze poznat free nodes je klucove k rieseniu sustavy
    """
    U = np.zeros(pocet_uzlov)
    count = np.zeros(pocet_uzlov)

    for row in dirichlet:
        function = row[1]
        tag, coord, _ = gmsh.model.mesh.getNodes(1, row[0], includeBoundary=True)
        coord = np.reshape(coord, (-1, 3))[:, :2].flatten()
        for i in range(len(tag)):
            U[tag_to_index[tag[i]]] += float(function(coord[2 * i], coord[2 * i + 1]))
            count[tag_to_index[tag[i]]] += 1

    diri_nodes = list()
    for i in range(pocet_uzlov):
        if count[i] != 0:
            U[i] = U[i] / count[i]
            diri_nodes.append(i)

    return U, diri_nodes


def solve_system(
    A: sp.csr_matrix,
    b: np.ndarray,
    U: np.ndarray,
    diri_nodes: list[int],
    pocet_uzlov: int
) -> np.ndarray:
    """
    Najprv vyrobi "doplnok" zoznamu diri_nodes - teda zoznam free_nodes, ktory pouzije na to aby
    z matice A vyrezal podmaticu A_(11), a ulozi ju na lavu stranu sustavy,
    takisto aj z vektoru b odreze diri_nodes
    takisto aj z A*U po vynasobeni vyreze diri_nodes
    z tychto troch prvkov potom zlozi sustavu a vyriesi ju
    """
    free_nodes = np.setdiff1d(np.arange(pocet_uzlov), diri_nodes, assume_unique=True)
    A_11 = A[free_nodes, :][:, free_nodes]
    bb = b[free_nodes]

    c = A @ U
    cc = c[free_nodes]

    riesenie = spl.spsolve(A_11, bb - cc)
    U[free_nodes] = riesenie

    return U

def validate_input(data: dict) -> None:
    """
    Kontroluje ci vo vstupnych datach nie je chyba.
    Detekovane chyby su tieto: 
    vo vstupnom slovníku sa nachadzaju povinne kluce "nodes", "dirichlet", "right_side";
    Nodes - nejaky zadany node ma menej ako 2 suradnice, alebo nie su typu float/int, alebo je zadanych nodes menej ako 3;
    Dirichlet - nezadana ziadna podmienka, alebo nejaka podmienka ma menej ako 2 prvky, alebo tag hrany nema spravny tag,
                alebo je pre nejaku hranu podmienka duplicitna, alebo nie je zadana callable funkcia na druhom mieste
    Neumann - rovnake chyby ako pri Dirichlet, az na to, ze "neumann" nie je nutny kluc a teda nemusi byt vo vstupe vobec, 
              navyse kontrolujeme ci hrana nema zadanu aj Dirichletovu aj Neumannovu podmienku
    Right hand side - nie je zadana ako callable funkcia
    Mesh size - parameter nie je kladne cislo
    """

    # prava strana, dirichletove podmienky a uzly su nutne
    required_keys = ["nodes", "dirichlet", "right_side"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Vstupnym datam chyba povinny kluc: '{key}'")
        
    #dostatocny pocet uzlov
    num_lines = len(data["nodes"])
    if num_lines < 3:
        raise ValueError("Polygon musi obsahovat aspon 3 vrcholy.")
    
    # overenie ze su suradnice float alebo int, a ze su aspon dve
    for row in data["nodes"]:
        if len(row) < 2:
            raise ValueError("Kazdy bod v 'nodes' musi mat 2 suradnice [x, y].")
        
        if type(row[0]) not in (int, float) or type(row[1]) not in (int, float):
            raise TypeError("Suradnice bodu v 'nodes' musia byt cisla (int alebo float).")
        
    #aspon jedna dirichletova podmienka je nutna
    if len(data["dirichlet"]) == 0:
        raise ValueError("Musis zadat aspon jednu Dirichletovu okrajovu podmienku.")

    #kontrola, ci data zadane ako dirichlet splnaju format (tag,funkce), kde tag je int a funkce je callable, a ci nie su duplicity
    used_dirichlet = []
    for row in data["dirichlet"]:
        if len(row) < 2:
            raise ValueError("Kazdy zoznam v Dirichlet podmienkach musi obsahovat dva prvky: [cislo_hrany,funkce]")

        line_tag = row[0]
        if type(line_tag) is not int or line_tag < 1 or line_tag > num_lines:
            raise ValueError(f"Neplatny tag hrany {line_tag} v Dirichlet. Tag musi byt cele cislo od 1 do {num_lines}.")

        if line_tag in used_dirichlet:
            raise ValueError(f"Hrana cislo {line_tag} ma zadanu Dirichletovu podmienku viackrat.")
        used_dirichlet.append(line_tag)

        if not callable(row[1]):
            raise TypeError(f"Dirichletova podmienka pre hranu {line_tag} musi obsahovat funkciu.")

    #rovnaka kontrola ale pre nepovinne data Neumann
    used_neumann = []
    for row in data.get("neumann",[]):
        if len(row) < 2:
            raise ValueError("Kazdy zoznam v Neumann podmienkach musi obsahovat dva prvky: [cislo_hrany,funkce]")

        line_tag = row[0]
        if type(line_tag) is not int or line_tag < 1 or line_tag > num_lines:
            raise ValueError(f"Neplatny tag hrany {line_tag} v Neumann. Tag musi byt cele cislo od 1 do {num_lines}.")

        if line_tag in used_neumann:
            raise ValueError(f"Hrana cislo {line_tag} ma zadanu Neumannovu podmienku viackrat.")
        if line_tag in used_dirichlet:
            raise ValueError(f"Hrana cislo {line_tag} ma sucasne zadanu Dirichletovu aj Neumannovu podmienku.")
        used_neumann.append(line_tag)

        if not callable(row[1]):
            raise TypeError(f"Neumannova podmienka pre hranu {line_tag} musi obsahovat funkciu.")
        
    #zadana prava strana musi zobrazit (x,y) do R
    if not callable(data["right_side"]):
        raise TypeError("Polozka 'right_side' musi byt funkciou.")

    if "mesh_size" in data:
        ms = data["mesh_size"]
        if type(ms) is bool or type(ms) not in (int, float) or ms <= 0:
            raise ValueError("Parameter 'mesh_size' musí byť kladné číslo väčšie ako 0.")


def compute_flux(
    elements3: list[list[int]] | np.ndarray,
    node_coordinates: list[list[float]] | np.ndarray,
    U: list[float] | np.ndarray,
    tag_to_index: dict[int, int],
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Prechádza elementmi a pre každý trojuholník vypočíta súradnice ťažiska a vektor toku q = -grad(u).
    """
    centre_of_mass = []
    flux = []

    for triangle in elements3:
        #ziskanie tagov a indexov uzlov, ktore element tvoria
        tag1, tag2, tag3 = triangle[0], triangle[1], triangle[2]
        idx1, idx2, idx3 = tag_to_index[tag1], tag_to_index[tag2], tag_to_index[tag3]

        #ziskanie suradnic vrcholov trojuholnikoveho elementu
        x1, y1 = node_coordinates[idx1][0], node_coordinates[idx1][1]
        x2, y2 = node_coordinates[idx2][0], node_coordinates[idx2][1]
        x3, y3 = node_coordinates[idx3][0], node_coordinates[idx3][1]

        #tazisko
        xc, yc = (x1 + x2 + x3) / 3.0, (y1 + y2 + y3) / 3.0
        centre_of_mass.append([xc, yc])

        #vypocet gradientu pomocou Cramerovho pravidla
        u1, u2, u3 = U[idx1], U[idx2], U[idx3]
        D = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

        du_dx = (u1 * (y2 - y3) + u2 * (y3 - y1) + u3 * (y1 - y2)) / D
        du_dy = (u1 * (x3 - x2) + u2 * (x1 - x3) + u3 * (x2 - x1)) / D

        flux.append([-du_dx, -du_dy])

    return centre_of_mass, flux


def create_triangulation(
    node_coordinates: list[list[float]] | np.ndarray,
    elements3: list[list[int]] | np.ndarray,
    tag_to_index: dict[int, int],
) -> matplotlib.tri.Triangulation:
    """
    založenie objektu Triangulation zo zoznamov suradnic a elementov
    pre ľahšiu manipuláciu v matplotlib
    """
    # rozdelenie node_coordinates na osobitné zoznamy x a y
    x = []
    y = []
    for row in node_coordinates:
        x.append(row[0])
        y.append(row[1])

    # vytvorenie zoznamu triangles z matice elements3
    # indexy su osetrene tak, ze suhlasia medzi zoznamami x,y,triangles
    triangles = []
    for element in elements3:
        i1 = tag_to_index[element[0]]
        i2 = tag_to_index[element[1]]
        i3 = tag_to_index[element[2]]
        triangles.append([i1, i2, i3])

    # zalozenie objektu triangulation
    triangulace = matplotlib.tri.Triangulation(x, y, triangles)
    return triangulace


def plot_2d_heatmap(
    triangulace: matplotlib.tri.Triangulation, U: list[float] | np.ndarray) -> None:
    """
    2D heatmap funkce U(x,y) na mriežke originálneho meshu, pomocou funkce tricolor
    volba shading="gouraud" presne koresponduje s tym, ze nasa funkce U(x,y) je po částech lineární
    """
    matplotlib.pyplot.figure()
    matplotlib.pyplot.tripcolor(
        triangulace, U, shading="gouraud", cmap="coolwarm")
    matplotlib.pyplot.triplot(triangulace, color="black", lw=0.5, alpha=0.3)
    matplotlib.pyplot.colorbar(label="Hodnota U")
    matplotlib.pyplot.gca().set_aspect("equal")
    matplotlib.pyplot.xlabel("x")
    matplotlib.pyplot.ylabel("y")


def plot_2d_quiver(
    triangulace: matplotlib.tri.Triangulation,
    centre_of_mass: list[list[float]], flux: list[list[float]]) -> None:
    """
    2D vector plot funkce -grad(U) 
    """
    xc = []
    yc = []
    for pt in centre_of_mass:
        xc.append(pt[0])
        yc.append(pt[1])

    qx = []
    qy = []
    for f in flux:
        qx.append(f[0])
        qy.append(f[1])

    matplotlib.pyplot.figure()
    matplotlib.pyplot.triplot(triangulace, color="black", lw=0.5, alpha=0.3)
    matplotlib.pyplot.quiver(
        xc, yc, qx, qy, color="blue", angles="xy", scale_units="xy"
    )
    matplotlib.pyplot.gca().set_aspect("equal")
    matplotlib.pyplot.title("Vektorové pole -grad(U)")
    matplotlib.pyplot.xlabel("x")
    matplotlib.pyplot.ylabel("y")


def plot_3d_trisurf(
    triangulace: matplotlib.tri.Triangulation, U: list[float] | np.ndarray, subdiv: int = 2) -> None:
    """
    3D zobrazenie U(x,y) pomocou funkce trisurf s pridanym zjemnením triangulace pomocou metody refine_field
    """

    fig, ax = matplotlib.pyplot.subplots(subplot_kw={"projection": "3d"})

    # zjemnenie siete
    refiner = matplotlib.tri.UniformTriRefiner(triangulace)
    finer_triangulace, finer_U = refiner.refine_field(U, subdiv=subdiv)

    surface = ax.plot_trisurf(finer_triangulace, finer_U, cmap="coolwarm", edgecolor="none")
    fig.colorbar(surface, ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("U(x,y)")


vybrany_subor = None
def choose_file():
    global vybrany_subor
    path = filedialog.askopenfilename(title="vyberte subor so vstupom", filetypes=[("Python súbory", "*.py")])
    if path:
        vybrany_subor = path


def nacitaj_vstup(cesta_k_suboru: str) -> dict:
    spec = importlib.util.spec_from_file_location("docasny_nazov", cesta_k_suboru)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul.input_data


def start_fem(user_input: dict ) -> None:

    validate_input(user_input)

    gmsh.initialize()
    try:
        mesh_size = user_input.get("mesh_size", 0.1)
        generate_mesh(user_input["nodes"], mesh_size=mesh_size)
        
        node_coordinates, tag_to_index, elements3, pocet_uzlov = extract_mesh_data()

        A, b = assemble_system(pocet_uzlov, elements3, node_coordinates, tag_to_index, user_input["right_side"])

        if user_input.get("neumann"):
            b = apply_neumann(b, user_input["neumann"], node_coordinates, tag_to_index)

        U, diri_nodes = apply_dirichlet(pocet_uzlov, user_input["dirichlet"], tag_to_index)

        U = solve_system(A, b, U, diri_nodes, pocet_uzlov)

        centre_of_mass, flux = compute_flux(elements3,node_coordinates,U,tag_to_index)

        triangulace = create_triangulation(node_coordinates, elements3, tag_to_index)

        plot_2d_heatmap(triangulace, U)

        plot_2d_quiver(triangulace, centre_of_mass, flux)

        plot_3d_trisurf(triangulace, U)
        matplotlib.pyplot.show()

    finally:
        gmsh.finalize()


def start_solver():
    if vybrany_subor is None:
        return
    user_input = nacitaj_vstup(vybrany_subor)
    start_fem(user_input)


okno = tk.Tk()
okno.geometry("600x350")

tlacidlo_zvol = tk.Button(okno, text="zvolte subor", command=choose_file)
tlacidlo_zvol.pack(pady=10)

tlacidlo_spust = tk.Button(okno, text="spustit vypocet", command=start_solver)
tlacidlo_spust.pack(pady=10)

okno.mainloop()

