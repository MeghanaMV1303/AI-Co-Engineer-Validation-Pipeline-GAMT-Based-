import networkx as nx
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods, TopoDS_Shape
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface
import numpy as np

def get_surface_type(surface_adaptor):
    geom_type = surface_adaptor.GetType()
    if geom_type == GeomAbs_Plane:
        return "Plane"
    elif geom_type == GeomAbs_Cylinder:
        return "Cylinder"
    elif geom_type == GeomAbs_Cone:
        return "Cone"
    elif geom_type == GeomAbs_Sphere:
        return "Sphere"
    elif geom_type == GeomAbs_Torus:
        return "Torus"
    elif geom_type == GeomAbs_BSplineSurface:
        return "BSpline"
    return "Other"

def get_face_properties(face):
    """Compute features for a given TopoDS_Face."""
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    area = props.Mass()
    
    # Centroid
    cg = props.CentreOfMass()
    centroid = [cg.X(), cg.Y(), cg.Z()]
    
    # Surface type & Normal
    surf = BRepAdaptor_Surface(face)
    surf_type = get_surface_type(surf)
    
    # Evaluate normal at the center of the u, v bounds
    u_min, u_max, v_min, v_max = surf.FirstUParameter(), surf.LastUParameter(), surf.FirstVParameter(), surf.LastVParameter()
    u_mid, v_mid = (u_min + u_max) / 2.0, (v_min + v_max) / 2.0
    
    from OCC.Core.gp import gp_Pnt, gp_Vec
    _pnt = gp_Pnt()
    normal_dir = gp_Vec()
    from OCC.Core.GeomLProp import GeomLProp_SLProps
    
    BRep_Tool.Surface(face) # This gets the underlying Geom_Surface
    
    try:
        geom_surf = BRep_Tool.Surface(face)
        lprop = GeomLProp_SLProps(geom_surf, u_mid, v_mid, 1, 1e-6)
        if lprop.IsNormalDefined():
            d_dir = lprop.Normal()
            normal = [d_dir.X(), d_dir.Y(), d_dir.Z()]
        else:
            normal = [0.0, 0.0, 0.0]
    except Exception:
        normal = [0.0, 0.0, 0.0]
        
    # Standardize surface type to an index or one-hot later
    return {
        "area": area,
        "centroid": centroid,
        "normal": normal,
        "surface_type": surf_type
    }

def get_edge_properties(edge, face1=None, face2=None):
    """Compute features for a TopoDS_Edge, optionally considering adjacent faces."""
    props = GProp_GProps()
    brepgprop_LinearProperties(edge, props)
    length = props.Mass()
    
    curve = BRepAdaptor_Curve(edge)
    
    # Angle calculation between two faces would go here
    # For now, placeholder heuristics
    angle = 0.0
    convexity = 0.0 # 0 for flat, 1 for convex, -1 for concave
    
    if face1 and face2:
        # In a complete implementation, evaluate normals of both faces at a point on the edge
        # and compute the dihedral angle.
        # This is simplified for the prototype.
        pass
        
    return {
        "length": length,
        "angle": angle,
        "convexity": convexity
    }

def step_to_graph(filepath):
    """
    Reads a STEP file and builds a Face Adjacency Graph (FAG).
    Nodes = Faces
    Edges = Shared B-Rep Edges
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    if status != 1: # 1 means IFSelect_RetDone
        raise ValueError(f"Failed to read STEP file {filepath}")
        
    reader.TransferRoots()
    shape = reader.OneShape()
    
    G = nx.Graph()
    
    # 1. Extract all faces and build nodes
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    face_mapping = {} # TopoDS_Face hash to index
    
    idx: int = 0
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        faces.append(face)
        face_hash = face.__hash__()
        
        # Avoid duplicates
        if face_hash not in face_mapping:
            face_mapping[face_hash] = idx
            
            # Compute node features
            features = get_face_properties(face)
            G.add_node(idx, **features)
            idx += 1
            
        face_explorer.Next()
        
    # 2. Extract edges to find adjacencies
    # Create an edge-to-face map to find shared edges
    edge_to_faces: dict = {}
    
    for face in faces:
        f_idx = face_mapping[face.__hash__()]
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            e_hash = edge.__hash__()
            
            if e_hash not in edge_to_faces:
                edge_to_faces[e_hash] = {"edge": edge, "faces": []}
                
            edge_to_faces[e_hash]["faces"].append((f_idx, face))
            edge_explorer.Next()
            
    # 3. Build graph edges
    for e_hash, ef_data in edge_to_faces.items():
        attached_faces = ef_data["faces"]
        # If an edge is shared by exactly 2 faces, it's an adjacency
        if len(attached_faces) == 2:
            f1_idx, face1 = attached_faces[0]
            f2_idx, face2 = attached_faces[1]
            
            edge_geom = ef_data["edge"]
            edge_features = get_edge_properties(edge_geom, face1, face2)
            
            # Avoid multigraph if multiple edges connect same faces (just update or parallel)
            # We'll just add/overwrite for simple FAG
            G.add_edge(f1_idx, f2_idx, **edge_features)
            
    return G

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("step_file", help="Path to STEP file")
    args = parser.parse_args()
    
    print(f"Extracting FAG from {args.step_file}...")
    try:
        graph = step_to_graph(args.step_file)
        print(f"Graph constructed:")
        print(f"  Nodes (Faces): {graph.number_of_nodes()}")
        print(f"  Edges (Shared Edges): {graph.number_of_edges()}")
        
        if graph.number_of_nodes() > 0:
            print(f"  Sample Node 0 Features: {graph.nodes[0]}")
    except Exception as e:
        print(f"Error: {e}")
