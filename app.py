import os
import tempfile

import streamlit as st
import open3d as o3d
import numpy as np
import plotly.graph_objects as go

from mesh_pipeline import poisson_surface_reconstruction_fast

# — helpers to load an uploaded file vs a path —
@st.cache_data
def load_mesh_from_upload(file) -> o3d.geometry.TriangleMesh:
    suffix = os.path.splitext(file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.read())
    tmp.flush()
    tmp.close()
    mesh = o3d.io.read_triangle_mesh(tmp.name)
    return mesh

@st.cache_data
def load_mesh_from_path(path: str) -> o3d.geometry.TriangleMesh:
    return o3d.io.read_triangle_mesh(path)

st.title("3D Poisson Mesh Explorer")

# Upload or text‐input
uploaded = st.file_uploader("Upload a PLY/OBJ/STL point cloud or mesh", 
                            type=["ply","obj","stl","pcd","xyz"])
path = st.text_input("—or enter a local path to a mesh—")

mesh = None
if uploaded is not None:
    mesh = load_mesh_from_upload(uploaded)
elif path:
    mesh = load_mesh_from_path(path)

if mesh:
    # Ensure normals & colors
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.7,0.7,0.7])

    # Sidebar controls for Poisson
    st.sidebar.header("Poisson Reconstruction")
    run_poisson = st.sidebar.checkbox("Run Poisson on this cloud", value=False)
    depth            = st.sidebar.slider("Depth", 4, 12, value=8)
    voxel_size       = st.sidebar.number_input("Voxel size", value=5.0)
    hole_fill_size   = st.sidebar.number_input("Hole‐fill max diameter", value=1e6)
    target_tris      = st.sidebar.number_input("Target triangles", value=20000)

    if run_poisson:
        pts    = np.asarray(mesh.vertices)
        cols   = (np.asarray(mesh.vertex_colors)*255).astype(np.uint8)
        mesh = poisson_surface_reconstruction_fast(
            points_xyz=pts,
            colors_rgb=cols,
            depth=int(depth),
            voxel_size=voxel_size,
            mesh_target_triangles=int(target_tris),
            hole_fill_size=hole_fill_size,
            enable_hole_filling=True
        )

    # Convert to Plotly mesh
    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=tris[:,0], j=tris[:,1], k=tris[:,2],
            intensity=np.linalg.norm(verts, axis=1),
            colorscale="Viridis", opacity=1.0
        )
    ])
    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data"
        ),
        margin=dict(l=0,r=0,t=30,b=0)
    )

    st.plotly_chart(fig, use_container_width=True)
