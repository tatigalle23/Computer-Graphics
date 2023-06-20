from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os

width, height = 1080,1080
lastX, lastY = width / 2, height / 2
xoffset = 0
yoffset = 0
key_v = False
glCamAngx = 45
glCamAngy = 40
gCamZoom = 0.5
call = ''
move = False
left = 0
top = 0
glCamDist= 4.0
key_z = False
path = ''
flag = 0


camera_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 c_vin_pos;
layout (location = 1) in vec3 c_vin_color;

out vec4 c_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(c_vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    c_color = vec4(c_vin_color, 1.);
}
'''

camera_fragment_shader_src = '''
#version 330 core

in vec4 c_color;

out vec4 FragColor_c;

void main()
{
    FragColor_c = c_color;
}
'''

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos;
layout (location = 1) in vec3 vin_normal;

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;

const int MAX_LIGHTS = 3;  // Maximum number of lights

uniform vec3 light_positions[MAX_LIGHTS];
uniform vec3 light_colors[MAX_LIGHTS];
uniform vec3 material_color;

void main()
{
    float material_shininess = 32.0;

    // Light components
    vec3 ambient = vec3(0);
    vec3 diffuse = vec3(0);
    vec3 specular = vec3(0);

    // Calculate lighting for each light
    for (int i = 0; i < MAX_LIGHTS; i++) {
        vec3 light_dir = normalize(light_positions[i] - vout_surface_pos);
        vec3 normal = normalize(vout_normal);

        // Diffuse
        float diff = max(dot(normal, light_dir), 0);
        diffuse += diff * light_colors[i] * material_color;

        // Specular
        vec3 view_dir = normalize(view_pos - vout_surface_pos);
        vec3 reflect_dir = reflect(-light_dir, normal);
        float spec = pow(max(dot(view_dir, reflect_dir), 0), material_shininess);
        specular += spec * light_colors[i];
    }
    
    vec3 color = ambient + diffuse + specular;    
    FragColor=vec4(color, 1.0);
}
'''


class Node:
    def __init__(self, parent, shape_transform):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------

    # vertex shader
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object

    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())

    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object

    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def create_plane():
    positions = []
    colors = []
    for i in np.arange(-100, 100, 0.5):
        positions.append(glm.vec3(i, 0.0, 0.0))
        positions.append(glm.vec3(i, 0.0, 100.0))
        colors.append(glm.vec3(0.7, 0.7, 0.7))
        colors.append(glm.vec3(0.7, 0.7, 0.7))

        positions.append(glm.vec3(i, 0.0, 0.0))
        positions.append(glm.vec3(i, 0.0, -100.0))
        colors.append(glm.vec3(0.7, 0.7, 0.7))
        colors.append(glm.vec3(0.7, 0.7, 0.7))

        positions.append(glm.vec3(0.0, 0.0, i))
        positions.append(glm.vec3(100.0, 0.0, i))
        colors.append(glm.vec3(0.7, 0.7, 0.7))
        colors.append(glm.vec3(0.7, 0.7, 0.7))

        positions.append(glm.vec3(0.0, 0.0, i))
        positions.append(glm.vec3(-100.0, 0.0, i))
        colors.append(glm.vec3(0.7, 0.7, 0.7))
        colors.append(glm.vec3(0.7, 0.7, 0.7))

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # create a VBO for positions
    pos_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo)
    glBufferData(GL_ARRAY_BUFFER, np.array(positions, dtype=np.float32), GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # create a VBO for colors
    color_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
    glBufferData(GL_ARRAY_BUFFER, np.array(colors, dtype=np.float32), GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    return vao, positions


def draw_frame(plane_vao, plane_index_count):
    glBindVertexArray(plane_vao)
    glDrawArrays(GL_LINES, 0, len(plane_index_count))
    glBindVertexArray(0)


def scroll_callback(window, xoffsett, yoffsett):
    global gCamZoom
    gCamZoom+= yoffsett*0.3

def mouse_look_callback(window, xpos, ypos):
    global glCamAngx, glCamAngy, xoffset, yoffset, left, top, lastY, lastX

    if call == 'left' and move:
        xoffset = xpos - lastX
        yoffset = lastY - ypos

        glCamAngx += xoffset * 0.4
        glCamAngy += yoffset * 0.4

    elif call == 'right' and move:
        xoffset = xpos - lastX
        yoffset = lastY - ypos

        left -= xoffset * 0.006
        top += yoffset * 0.006

    lastX = xpos
    lastY = ypos

def button_callback(window, button, action, mod):
    global call, move
    if button==GLFW_MOUSE_BUTTON_LEFT:
        if action==GLFW_PRESS:
            call = 'left'
            move = True
        elif action==GLFW_RELEASE:
            move = False


    if button==GLFW_MOUSE_BUTTON_RIGHT:
        if action==GLFW_PRESS:
            call = 'right'
            move = True
        elif action==GLFW_RELEASE:
            move = False

def key_callback(window, key, scancode, action, mods):
    global key_v,flag, key_z

    if action == GLFW_PRESS or action == GLFW_REPEAT:
        if key == GLFW_KEY_V:
            key_v=not key_v 
        elif key == GLFW_KEY_H:
          flag=2
        elif key== GLFW_KEY_Z:
            key_z=not key_z
          

def Panning(left, top):
    a = glm.translate(glm.mat4(1), glm.vec3(-left, 0, 0))
    b = glm.translate(glm.mat4(1), glm.vec3(0, top, 0))
    translate = b*a
    return translate

def Orbit():
    global glCamDist, glCamAngx, glCamAngy

    flip_up_vector = 1
    # convert azimuth and elevation to radians
    azim = glm.radians(glCamAngx)
    elev = glm.radians(glCamAngy)
    # calculate the camera position in spherical coordinates
    cam_pos = glm.vec3(
        glCamDist * np.cos(elev) * np.sin(azim),
        glCamDist * np.sin(elev),
        glCamDist * np.cos(elev) * np.cos(azim)
    )
    # calculate the target position in cartesian coordinates
    target_pos = glm.vec3(0, 0, 0)
    #flipping at the top in y
    if np.cos(elev) <= 0.0:
        flip_up_vector = -1

    # construct the view matrix
    view = glm.lookAt(cam_pos + target_pos, target_pos, glm.vec3(0, flip_up_vector, 0))
    return view

def Zoom(V):
    global gCamZoom
    forward = glm.vec3(glm.inverse(glm.mat3(V))[2])
    zoom = glm.translate(glm.mat4(1), forward * gCamZoom)
    return zoom

#******************************************************************2nd part******************************************

def obj_load(path):
    v_coords = []
    n_coords = []
    v_index = []
    n_index = []
    tmp_v = []
    tmp_n = []
    gFace = [0, 0, 0]

    for line in open(path, 'r'):
        if line.startswith('#'):
            continue

        values = line.split()
        valNum = values[1:]
        
        if not values:
            continue

        if values[0]=='v':
            tmp = [float(values[1]), float(values[2]), float(values[3])]
            v_coords.append(tmp)
        if values[0]=='vn':
            tmp = [float(values[1]), float(values[2]), float(values[3])]
            n_coords.append(tmp)
        if values[0]=='f':

            if len(valNum) == 3:
                gFace[0] += 1
            elif len(valNum) == 4:
                gFace[1] += 1
            else:
                gFace[2] += 1

            for i in range(1, len(valNum)-1):
                face_i = []
                norm_i = []

                for v in values[i:i+2]:
                    w = v.split('/')
                    face_i.append(int(w[0])-1)
                    norm_i.append(int(w[2])-1)
                v = values[len(valNum)]
                w = v.split('/')
                face_i.append(int(w[0])-1)
                norm_i.append(int(w[2])-1)

                if i != 1: # n-polygon
                    tmp_v.append(face_i)
                    tmp_n.append(norm_i)
                else:
                    v_index.append(face_i)
                    n_index.append(norm_i)

    # n-polygon
    for i in range(len(tmp_v)-1, -1, -1):
        v_index.insert(len(n_coords), tmp_v[i])
        n_index.insert(len(n_coords), tmp_n[i])

    narr = np.zeros((len(v_coords), 3), dtype=np.float32)
    narr2 = np.zeros((len(v_coords), 3), dtype=np.float32)
    for i in range(len(v_index)):
        t1 = np.subtract(v_coords[v_index[i][1]], v_coords[v_index[i][0]])
        t2 = np.subtract(v_coords[v_index[i][2]], v_coords[v_index[i][0]])

        nv = np.cross(t1, t2)
        nv = nv / np.sqrt(np.dot(nv, nv))

        for j in range(3):
            narr[v_index[i][j]] = n_coords[n_index[i][j]]
            narr2[v_index[i][j]] += nv

    for i in range(len(v_coords)):
        narr2[i] = narr2[i] / np.sqrt(np.dot(narr2[i], narr2[i]))
    
    varr = np.array(v_coords, dtype=np.float32)
    iarr = np.array(v_index, dtype=np.uint32)
    narr = np.array(narr, dtype=np.float32)
    narr2 = np.array(narr2, dtype=np.float32)

    deal = path.replace('\\', '/')
    deal = deal.split('/')
    file_name = deal[-1]

    message = "File name: " + file_name +"\n" + "Number of vertices: " + str(len(v_coords))+"\n"+ "Total number of faces: " + str(gFace[0]+gFace[1]+gFace[2])+"\n"+"Number of faces with 3 vertices: " + str(gFace[0])+ "\n" + "Number of faces with 4 vertices: " + str(gFace[1]) + "\n" + "Number of faces with more than 4 vertices: " + str(gFace[2])+"\n"
    print(message)
    return varr, iarr, narr, narr2

def draw_individual(path,shader,MVP,M,V,node_true,node_name,color_material):
    #varr=vertex narr=normals
    first_vec = glm.vec3(V[0][0], V[1][0], V[2][0])

    varr, iarr, narr, narr2 = obj_load(path)

    if node_true:
        MVP=MVP * node_name.get_global_transform() * node_name.get_shape_transform()
    
    
    MVP_loc= glGetUniformLocation(shader,  'MVP')
    M_loc= glGetUniformLocation(shader, 'M')
    view_pos = glGetUniformLocation(shader, 'view_pos')
    light_pos_location = glGetUniformLocation(shader, "light_positions[0]")
    light_color_location = glGetUniformLocation(shader, "light_colors[0]")
    material=glGetUniformLocation(shader, 'material_color')
    
    # Draw
    lights = [
            {'position': (2, 2, 2), 'color': (1, 1, 0)},
            {'position': (-2, 2, -2), 'color': (0, 1, 0)},
            {'position': (5, 5, 5), 'color': (0,0,1)}
        ]
    MAX_LIGHTS = 3

    num_lights = min(len(lights), MAX_LIGHTS)
    light_positions = []
    light_colors = []

    for light in lights[:num_lights]:
        light_positions += list(light['position'])
        light_colors += list(light['color'])

    glUseProgram(shader)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(view_pos, first_vec.x, first_vec.y, first_vec.z)
    glUniform3fv(light_pos_location, num_lights, light_positions)
    glUniform3fv(light_color_location, num_lights, light_colors)
    glUniform3f(material,color_material.x,color_material.y,color_material.z)
    

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    vbo_normals = glGenBuffers(1)
    vbo_vertices= glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
    glBufferData(GL_ARRAY_BUFFER, narr2.nbytes, narr2, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
    glBufferData(GL_ARRAY_BUFFER, varr.nbytes, varr, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, iarr.nbytes, iarr, GL_STATIC_DRAW)
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, iarr.size, GL_UNSIGNED_INT, None)

#orbit,panning,zoom O*P*Z
def draw_node(shader,mvp,model,V):
    t = glfwGetTime()
    script_dir = os.path.dirname(__file__) 
    rel_path = "cup.obj"
    obj1 = os.path.join(script_dir, rel_path)
    
    script_dir2 = os.path.dirname(__file__) 
    rel_path2 = "spoon.obj"
    obj2 = os.path.join(script_dir2, rel_path2)

    script_dir3 = os.path.dirname(__file__) 
    rel_path3 = "plate.obj"
    obj3 = os.path.join(script_dir3, rel_path3)

    script_dir4 = os.path.dirname(__file__) 
    rel_path4 = "cereal.obj"
    obj4 = os.path.join(script_dir4, rel_path4)

    script_dir5 = os.path.dirname(__file__) 
    rel_path5 = "cookie.obj"
    obj5 = os.path.join(script_dir5, rel_path5)

    script_dir6 = os.path.dirname(__file__) 
    rel_path6 = "fly.obj"
    obj6 = os.path.join(script_dir6, rel_path6)

    script_dir7 = os.path.dirname(__file__) 
    rel_path7 = "fly.obj"
    obj7 = os.path.join(script_dir7, rel_path7)

    #cup base
    scale = glm.scale(glm.mat4(1.0), glm.vec3(0.1, 0.1, 0.1))
    translate=glm.translate(glm.mat4(1), glm.vec3(0., 0.1, 0))
    base = Node(None,translate*scale)    

    #spoon arm1
    translate=glm.translate(glm.mat4(1), glm.vec3(0, 1.1,-0.5))
    rotate=glm.rotate(glm.mat4(1), -np.radians(220), glm.vec3(1, 0, 0))
    scale = glm.scale(glm.mat4(1.0), glm.vec3(0.03, 0.03, 0.03))
    arm1 = Node(base, translate*rotate*scale)

    #plate arm2   
    scale = glm.scale(glm.mat4(1.0), glm.vec3(0.8, 0.8, 0.8))
    arm2 = Node(base, scale)

    #cereal arm1_1
    translate=glm.translate(glm.mat4(1), glm.vec3(0, 0.7, 0))
    rotate=glm.rotate(glm.mat4(1), np.radians(120), glm.vec3(1, 0, 0))
    scale = glm.scale(glm.mat4(1.0), glm.vec3(0.1, 0.1, 0.1))   
    arm1_1 = Node(arm1, translate*rotate*scale)

    #cookie arm2_1
    scale = glm.scale(glm.mat4(1.0), glm.vec3(0.03, 0.03, 0.03)) 
    rotate=glm.rotate(glm.mat4(1), -np.radians(360), glm.vec3(1, 0, 0))
    translate=glm.translate(glm.mat4(1), glm.vec3(0.6, 0.1, 0)) 
    arm2_1 = Node(arm2, translate*rotate*scale)

    #fly arm2_2
    scale = glm.scale(glm.mat4(1.0), glm.vec3(0.07, 0.07, 0.07))
    translate=glm.translate(glm.mat4(1), glm.vec3(0.7, 0.4, 0)) 
    arm2_2 = Node(arm2, translate*scale)

    #fly arm1_2
    translate=glm.translate(glm.mat4(1), glm.vec3(0, 1.2, -0.2))
    rotate=glm.rotate(glm.mat4(1), -np.radians(130), glm.vec3(1, 0, 0))
    scale = glm.scale(glm.mat4(1.0), glm.vec3(0.07, 0.07, 0.07))
    arm1_2 = Node(arm1, translate*rotate*scale)

    #special transformation (oscilation for the spoon)
    angle = -glm.radians(20) * glm.sin(t*2)
    rotation_matrix = glm.rotate(glm.mat4(1.0), angle, glm.vec3(1, 0, 0))
    #oscilation for plate
    angle1 = -glm.radians(20) * glm.sin(t*2)
    rotation_matrix1 = glm.rotate(glm.mat4(1.0), angle1, glm.vec3(1, 0, 0))

    base.set_transform(glm.rotate(t, (0,1,0))*glm.translate((0,glm.cos(t),0)))
    arm1.set_transform(rotation_matrix )
    arm2.set_transform(rotation_matrix1)
    arm2_2.set_transform(glm.rotate(t, (0,1,0))*glm.translate((0,0,glm.cos(t))))

    base.update_tree_global_transform()
    draw_individual(obj1,shader,mvp,model,V,True,base,glm.vec3(0.8,0.8,0.8))
    draw_individual(obj2,shader,mvp,model,V,True,arm1,glm.vec3(0.5,0.5,0.5))
    draw_individual(obj4,shader,mvp,model,V,True,arm1_1,glm.vec3(1,0,0.8))
    draw_individual(obj7,shader,mvp,model,V,True,arm1_2,glm.vec3(0.5, 0, 0.21))
    draw_individual(obj3,shader,mvp,model,V,True,arm2,glm.vec3(0.8,0.8,0.8))
    draw_individual(obj5,shader,mvp,model,V,True,arm2_1,glm.vec3(0.8, 0.6, 0.4))
    draw_individual(obj6,shader,mvp,model,V,True,arm2_2,glm.vec3(0.5, 0, 0.21))


def drop_callback(window, paths):
    global path, flag, should_print
    path = paths[0]
    flag = 1
    should_print = True

def main():
    global key_v, flag,path, key_z, glCamDist
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(width, height, 'Project_2', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetKeyCallback(window, key_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL)
    glfwSetCursorPosCallback(window, mouse_look_callback)
    glfwSetDropCallback(window,drop_callback)

    # load shaders
    camera_shader_program = load_shaders(camera_vertex_shader_src, camera_fragment_shader_src)
    x_shader_program= load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(camera_shader_program, 'MVP')

    # prepare vaos
    plane_vao, plane_index_count = create_plane()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(camera_shader_program)        
        
        rotate=Orbit()
        translate = Panning(left, top)
        zoom = Zoom(rotate)
        # Combine the transformations
        M = translate*rotate*zoom
        # MVP = P*M   
        
        if not key_v:
            P = glm.perspective(90, 1, 1, 100)
            MVP = P*M  
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        else:
            P = glm.ortho(-2*glCamDist, 2*glCamDist, -2*glCamDist, 2*glCamDist, -100, 100)
            MVP = P*M  
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))            

        # Draw the objects using the combined transformation
        draw_frame(plane_vao, plane_index_count)
    
        if not key_z:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # Set polygon mode to fill

        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # Set polygon mode to wireframe

        if flag == 1:
            draw_individual(path, x_shader_program, MVP, M,zoom,False,None,glm.vec3(1,0,0))
        elif flag == 2:
            draw_node(x_shader_program, MVP, M,zoom)            


        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()