#!/usr/bin/env python3
"""
Interactive 3D Mesh Viewer for Poisson Reconstruction
Integrates with the existing mesh_pipeline.py to provide CAD-like 3D exploration
"""

import cv2
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import webbrowser
import tempfile
import os
import json
import http.server
import socketserver
import threading
from urllib.parse import urlparse
from mesh_pipeline import (
    poisson_surface_reconstruction_fast,
    ball_pivoting_reconstruction_optimized
)

class InteractiveMeshViewer:
    def __init__(self, port=8080):
        self.port = port
        self.server = None
        self.temp_dir = tempfile.mkdtemp()
        self.html_content = self._get_html_template()
        self.mesh_data = None

    def _get_html_template(self):
        """
        Returns the HTML template with embedded Three.js viewer.
        """
        # Note: Closing triple quotes properly added below
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive 3D Mesh Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        /* CSS styles omitted for brevity */
    </style>
</head>
<body>
    <div id="container"></div>
    <script>
        // Three.js viewer logic omitted for brevity
    </script>
</body>
</html>'''

    def _start_server(self):
        """
        Starts a simple HTTP server to serve the HTML and mesh data.
        """
        os.chdir(self.temp_dir)
        handler = http.server.SimpleHTTPRequestHandler
        self.server = socketserver.TCPServer(("", self.port), handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

    def _write_files(self):
        """
        Writes the HTML and mesh_data.json to the temporary directory.
        """
        html_path = os.path.join(self.temp_dir, 'index.html')
        with open(html_path, 'w') as f:
            f.write(self.html_content)

        mesh_path = os.path.join(self.temp_dir, 'mesh_data.json')
        with open(mesh_path, 'w') as f:
            json.dump(self.mesh_data, f)

    def load_and_reconstruct(self, point_cloud_path, method='poisson', **kwargs):
        """
        Loads a point cloud, reconstructs a mesh, and stores it for the viewer.
        """
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        if method == 'poisson':
            verts, faces = poisson_surface_reconstruction_fast(
                points, colors, **kwargs
            )
        else:
            verts, faces = ball_pivoting_reconstruction_optimized(
                points, colors, **kwargs
            )

        # Flatten data for JSON serialization
        self.mesh_data = {
            'vertices': verts.flatten().tolist(),
            'triangles': faces.flatten().tolist(),
            'colors': colors.flatten().tolist() if colors is not None else []
        }

    def view(self):
        """
        Launches the HTTP server and opens the viewer in the default browser.
        """
        self._write_files()
        self._start_server()
        url = f'http://localhost:{self.port}/index.html'
        webbrowser.open(url)
        print(f"Viewer running at {url}")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Shutting down server...")
            self.server.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Interactive Mesh Viewer')
    parser.add_argument('point_cloud', help='Path to input point cloud file')
    parser.add_argument('--method', choices=['poisson','ball'], default='poisson', help='Reconstruction method')
    parser.add_argument('--port', type=int, default=8080, help='Port for HTTP server')
    parser.add_argument('--depth', type=int, default=9, help='Poisson depth parameter')
    parser.add_argument('--voxel_size', type=float, default=2.0, help='Poisson voxel_size parameter')
    args = parser.parse_args()

    viewer = InteractiveMeshViewer(port=args.port)
    viewer.load_and_reconstruct(
        args.point_cloud,
        method=args.method,
        depth=args.depth,
        voxel_size=args.voxel_size
    )
    viewer.view()
