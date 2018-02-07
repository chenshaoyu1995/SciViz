from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def linear_interp(n, x, f):
    # n = num discretization pts
    # [x[0], x[1]] interval
    # f[0] and f[1] = rgb triplets in x[0] and x[1]

    t = np.linspace(x[0], x[1], n)

    # prof solution
    # r = [ f[0] * (t[i] - x[0]) / (x[1] - x[0]) + f[1] * (x[1] - t[i]) / (x[1] - x[0]) for i in range(n)]
    # lreturn (np.asarray(r, dtype=float))

    # a little easier to parse
    left  = x[0]
    right = x[1]

    l_col = f[0]
    r_col = f[1]

    r2 = []
    for i in range(n):
        r2.append( l_col * (t[i] - left) / (right - left) + r_col * (right - t[i])/(right - left))
    
    return (np.asarray(r2, dtype=float))

def problem_1():

    n = 100 # discretization size
    x = np.asarray([0, 1], dtype=float)
    f = np.asarray([[1, 0, 0], [0, 0, 1]], dtype=float)

    f_linear = linear_interp(n, x, f)

    # generating a 2d image where the rows are copies of f_linear
    f2d_linear = np.zeros((n, n, 3))
    f2d_linear[:] = f_linear*255

    a_linear = Image.fromarray(f2d_linear.astype('uint8'), 'RGB')
    a_linear.show()

def problem_2_1():

    #defining corners
    n = 100 # discretization size
    x = np.asarray([0, 1],dtype=float)
    y = np.asarray([0, 1],dtype=float)
    f = np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=float)

    l_bottom = linear_interp(n, x, f[0:2])
    l_top    = linear_interp(n, x, f[2:])

    f_bilinear = np.zeros((n, n, 3))
    for i in range(n):
        f_bilinear[:,i] = linear_interp(n, y,[l_bottom[i], l_top[i]])*255

    a_bilinear = Image.fromarray(f_bilinear.astype('uint8'), 'RGB')
    a_bilinear.show()

def bilinear_interp(n, m, x, y, f):
    
    tx = np.linspace(x[0], x[1], n)
    ty = np.linspace(y[0], y[1], m)

    F =np.asarray([[f[0], f[2]], [f[1], f[3]]])
    def bilinear_formula(i, j):
        ratio = 1.0/((x[1] - x[0]) * (y[1]-y[0]))
        xa = np.asarray([x[1] - tx[i], tx[i]-x[0]])
        ya = np.asarray([y[1] - ty[j], ty[j]-y[0]])
        mij = np.asarray([ratio*np.dot(xa,np.dot(F[:,:,k],ya)) for k in range(3)])
        return(mij)

    M = [bilinear_formula(i, j) for i in range(n) for j in range(m)]
    return (np.asarray(M).reshape(n, m, 3))

def problem_2_2():
    #defining corners
    n = 100
    x = np.asarray([0, 1],dtype=float)
    y = np.asarray([0, 1],dtype=float)
    f = np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=float)

    f2_bilinear = bilinear_interp(n, n, x, y, f) * 255
    a2_bilinear = Image.fromarray(f2_bilinear.astype('uint8'), 'RGB')
    a2_bilinear.show()


###############################################################################################################

def baricentric_interpolate(triangle, pt, colors):

    AB = (triangle[0], triangle[1]) 
    AC = (triangle[0], triangle[2])

    # find area of triangle using cross product
    total_area = np.linalg.norm(np.cross(AB, AC)) / 2.0

    AP = (triangle[0], pt)

    # lambda value is proportional to area of inner triangle (OPPOSITE vertex)
    lambda2 = (np.linalg.norm(np.cross(AP, AB))) / (total_area * 2)
    lambda1 = (np.linalg.norm(np.cross(AP, AC))) / (total_area * 2)
    lambda0 = (1 - lambda1 - lambda2)

    # compute overall color by adding dot products (contrib from each vertex's color)
    return (np.dot(lambda0, colors[0]) + np.dot(lambda1, colors[1]) + np.dot(lambda2, colors[2]))


def problem_3():

    n = 100
    x = np.asarray([0, 1], dtype=float)
    y = np.asarray([0, 1], dtype=float)
    colors = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)

    """
    Diagram for the visually inclined:

                <--- j(0, n)--->
                A(r)           C(g)
                 ---------------
          ^     |       |       |
          |     |       |       |
          |     |       |       |
       i(0, n)   ---------------
          |     |       |       |
          |     |       |       |
          |     |       |       |
          v      ---------------
                B(b)           D(w)

    """

    tri1 = np.asarray([ [x[0], y[0]], [x[1], y[1]], [x[1], y[0]]], dtype=float) * n # A, D, B
    tri2 = np.asarray([ [x[0], y[0]], [x[0], y[1]], [x[1], y[1]]], dtype=float) * n # A, C, D

    result = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):

            """ 
                you can use a more robust method of determining which triangle
                the point lies in. for now, just comparing against slope of
                square bisection 
            """

            if i > j:
                result[i][j] = baricentric_interpolate(tri1, [i, j], [colors[0], colors[3], colors[2]])
            else:
                result[i][j] = baricentric_interpolate(tri2, [i, j], [colors[0], colors[1], colors[3]])
                
    result = result * 255   
    img_result = Image.fromarray(result.astype('uint8'), 'RGB')
    img_result.show()

def main():

    problem_1()

    problem_2_1()

    problem_2_2()

    problem_3()


if __name__ == "__main__":
    main()
