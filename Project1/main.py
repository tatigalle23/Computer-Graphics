from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np


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

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

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

def prepare_vao_cube():
    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    vertices = glm.array(glm.float32,
        # position            color
        -0.5 ,  1 ,  0.5 ,  1, 1, 1, # v0
         0.5 , 0 ,  0.5 ,  1, 1, 1, # v2
         0.5 ,  1 ,  0.5 ,  1, 1, 1, # v1
                    
        -0.5 ,  1 ,  0.5 ,  1, 1, 1, # v0
        -0.5 , 0 ,  0.5 ,  1, 1, 1, # v3
         0.5 , 0 ,  0.5 ,  1, 1, 1, # v2
                    
        -0.5 ,  1 , -0.5 ,  1, 1, 1, # v4
         0.5 ,  1 , -0.5 ,  1, 1, 1, # v5
         0.5 , 0 , -0.5 ,  1, 1, 1, # v6
                    
        -0.5 ,  1 , -0.5 ,  1, 1, 1, # v4
         0.5 ,  0 , -0.5 ,  1, 1, 1, # v6
        -0.5 ,  0 , -0.5 ,  1, 1, 1, # v7
                    
        -0.5 ,  1 ,  0.5 ,  1, 1, 1, # v0
         0.5 ,  1 ,  0.5 ,  1, 1, 1, # v1
         0.5 ,  1 , -0.5 ,  1, 1, 1, # v5
                    
        -0.5 ,  1 ,  0.5 ,  1, 1, 1, # v0
         0.5 ,  1 , -0.5 ,  1, 1, 1, # v5
        -0.5 ,  1 , -0.5 ,  1, 1, 1, # v4
 
        -0.5 , 0 ,  0.5 ,  1, 1, 1, # v3
         0.5 , 0 , -0.5 ,  1, 1, 1, # v6
         0.5 , 0 ,  0.5 ,  1, 1, 1, # v2
                    
        -0.5 , 0 ,  0.5 ,  1, 1, 1, # v3
        -0.5 , 0 , -0.5 ,  1, 1, 1, # v7
         0.5 , 0 , -0.5 ,  1, 1, 1, # v6
                    
         0.5 ,  1 ,  0.5 ,  1, 1, 1, # v1
         0.5 , 0 ,  0.5 ,  1, 1, 1, # v2
         0.5 , 0 , -0.5 ,  1, 1, 1, # v6
                    
         0.5 ,  1 ,  0.5 ,  1, 1, 1, # v1
         0.5 , 0 , -0.5 ,  1, 1, 1, # v6
         0.5 ,  1 , -0.5 ,  1, 1, 1, # v5
                    
        -0.5 ,  1 ,  0.5 ,  1, 1, 1, # v0
        -0.5 , 0 , -0.5 ,  1, 1, 1, # v7
        -0.5 , 0 ,  0.5 ,  1, 1, 1, # v3
                    
        -0.5 ,  1 ,  0.5 ,  1, 1, 1, # v0
        -0.5 ,  1 , -0.5 ,  1, 1, 1, # v4
        -0.5 , 0 , -0.5 ,  1, 1, 1, # v7
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         10.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, 10.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, 0.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 10.0,  0.0, 0.0, 1.0, # z-axis end 
    )


    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO
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

    # draw the lines
    glDrawArrays(GL_LINES, 0, len(positions))
    return vao, positions

def draw_plane_vao(plane_vao, plane_index_count):
    glBindVertexArray(plane_vao)
    glDrawArrays(GL_LINES, 0, len(plane_index_count))
    glBindVertexArray(0)

def draw_frame(vao,MVP, MVP_loc):
    glBindVertexArray(vao)   
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))    
    glDrawArrays(GL_LINES, 0, 6)
    
def draw_cube(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_TRIANGLES, 0, 36)

def scroll_callback(window, xoffsett, yoffsett):
    global gCamZoom    
    gCamZoom+= yoffsett*0.3

def mouse_look_callback(window, xpos, ypos):
    global glCamAngx, glCamAngy, xoffset, yoffset, left, top, lastY, lastX

    if call == 'left' and move == True :       
        xoffset = xpos - lastX
        yoffset = lastY - ypos
        
        glCamAngx += xoffset*0.4
        glCamAngy += yoffset*0.4    
                                 
    elif call == 'right' and move == True :
        xoffset = xpos - lastX
        yoffset = lastY - ypos

        left -= xoffset*0.006
        top += yoffset*0.006
        
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
    global key_v

    if action == GLFW_PRESS or action == GLFW_REPEAT:
        if key == GLFW_KEY_V:
            key_v=not key_v 



def Panning(left, top):
    a = glm.translate(glm.mat4(1), glm.vec3(-left, 0, 0))
    b = glm.translate(glm.mat4(1), glm.vec3(0, top, 0))
    translate = a * b
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

def main():
    global key_v
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(1080, 1080, 'Project_1', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)
    glfwSetScrollCallback(window, scroll_callback)
    # register event callbacks
    glfwSetKeyCallback(window, key_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL)
    glfwSetCursorPosCallback(window, mouse_look_callback)


    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_cube = prepare_vao_cube()
    vao_frame = prepare_vao_frame()
    plane_vao, plane_index_count = create_plane()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # render in "wireframe mode"
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glUseProgram(shader_program)

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
        draw_frame(vao_frame, MVP, MVP_loc)
        draw_plane_vao(plane_vao, plane_index_count)
        draw_cube(vao_cube, MVP, MVP_loc)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
