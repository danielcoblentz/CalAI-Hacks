import stuff 

#turn 2D into a 3D encoding
 #transforms 2D into a 3D encoding matrix of size H x W x 4 (pixel size of img x encoding), stores it in a matrix called encoded_matrix. Each element is [R (range 0-255), G (range 0-255), B (0-255), relative depth [0.1- 10]]  output encoded_matrix. 

def encode():
    


    return encoding_matrix

#turn encoded_matrix into a 3D cloud of points
 # row i, column j, relative depth. (x,y,z) coordinate system. Point3D = (x,y,z) with color 0xRRGGBB. points_xyz = (N,3) array, each row will be (x,y,z). colors_rgb = (N,3) array, each row will be (R, G, B). normalized to 255. Output points_xyz, colors_rgb.

#make clusters
 #define close with euclidian distance. Connect nearby points, fit little triangle surfaces between them, estimate normals at each triangle 
# wrap the img around the triangles

#render
