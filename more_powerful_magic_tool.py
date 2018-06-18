import pandas as pd  # 使用pandas读取和处理数据
import numpy as np   # 使用numpy进行矩阵运算
from matplotlib import pyplot as plt     # 二维作图
from mpl_toolkits.mplot3d import Axes3D  # 三维作图
import pickle           # 数据存储
import scipy.integrate as integrate      #用于积分运算

class Finite_Element_of_triangle():
    # 初始化方程、边界条件及读取数据
    def __init__(self):
        '输入文件夹名字'
        self.filename = 'mesh3'
        print('this function try to solve the following equation:')
        print('-∂（p∂u/∂x）/∂x - ∂（p∂u/∂y）/∂y + qu = f \n')
        print("it's boundary conditions is:")
        print("u = 0 ,(x, y)∈Γ0 ")
        print("p(∂u/∂n) + σu = β ,(x, y)∈Γ1 \n")
        print("you can set p(x,y),q(x,y),f(x,y),σ(x,y) and β(x,y)")
        print("now,the program had set p = 1,others = 0")
        print("if you don't want to change these functions,please entry!")
        print("or input anything to change them one by one")
        self.init_equation()
        print("complete input\n")
        self.init_essential_boundary_condition()
        print("complete input\n")
        print(self.p, self.q, self.fu, self.sigma, self.beta)

    def init_equation(self):
        mark = input("if you want to change parameter:\n")
        if not mark:
            self.p, self.q, self.fu, = '1', '0', '0'
            self.sigma, self.beta = '0', '0'
            return
        try:
            # 用户输入函数
            print("it only support constant or polynomial for now")
            self.p = input("p(x,y) = ")
            self.q = input("q(x,y) = ")
            self.fu = input("f(x,y) = ")
            self.sigma = input("σ(x,y) = ")
            self.beta = input("β(x,y) = ")

            # 由于dblquad函数要求输入为表达式，暂时不转换为sympy，但是需要判断输入是否合法
            # 即除x,y以外不能有其他字符,exp怎么办？
            all_function = [self.p, self.q, self.fu, self.sigma, self.beta]
            for i in range(len(all_function)):
                replaced = all_function[i].replace('x', '1').replace('y', '1')
                eval(replaced)
        except():
            print("please input legal values or variables")
            self.init_equation()

    def init_essential_boundary_condition(self):
        self.boundary_value = ['50', '', '100', '']
        # 记录有本质边界条件与自然边界条件的边
        self.essential_boundary_mark = [1, 3]
        self.nature_boundary_mark = [2, 4]
        print("now,the program had set boundary1 = 50,boundary3 = 100,others = 0")
        print("if you want to change essential boundary conditions:")
        mark = input("please input anything:\n")
        if not mark:
            return
        try:
            print("if you don't want to change one of them ,\nplease input entry")
            self.boundary_value[0] = input("boundary1 = ")
            self.boundary_value[1] = input("boundary2 = ")
            self.boundary_value[2] = input("boundary3 = ")
            self.boundary_value[3] = input("boundary4 = ")

            self.essential_boundary_mark = []
            self.nature_boundary_mark =[]
            # 判断输入的值是否合法并记录有本质边界条件的边
            for i in range(3):
                # 判断是否有输入
                if self.boundary_value[i]:
                    # 该输入是否合法
                    if_legal = float(self.boundary_value[i])
                    self.essential_boundary_mark.append(i + 1)
                # 没有输入则是自然边界条件
                else:self.nature_boundary_mark.append(i + 1)
        except():
            print("please input legal values ,")
            print("it must be constant or entry")
            self.init_essential_boundary_condition()

    def init_grid(self):
        '自动识别文件夹中三种数据文件，并以Pandas特征数据返回'
        filename_point = './' + self.filename + '/E.n'
        filename_element = './' + self.filename + '/E.e'
        filename_boundary = './' + self.filename + '/E.s'

        # 读取节点信息文件
        number_of_triangle_point = pd.read_csv(filename_point,header = None, delim_whitespace = True,
                                               nrows = 1).iat[0,0]
        list_of_no_read = [0, number_of_triangle_point+1, number_of_triangle_point+2]
        # 第一行与最后两行数据无需读取
        # pd.read_csv参数参考http://www.cnblogs.com/datablog/p/6127000.html
        self.point_of_global = pd.read_csv(filename_point, delim_whitespace = True, 
                                           names=["point NO","x","y","boundary mark"],
                                           skiprows = list_of_no_read)

        # 读取单元信息文件
        number_of_triangle_element = pd.read_csv(filename_element,header=None,
                                                 delim_whitespace = True,nrows=1).iat[0,0]
        list_of_no_read = [0, number_of_triangle_element + 1, number_of_triangle_element + 2]
        self.element_of_global = pd.read_csv(filename_element, delim_whitespace=True, 
                                             names=["element number", "point i", "point j", "point k",
                                                    "neighbor ele i",  "neighbor ele j",  "neighbor ele k",
                                                    "boundary i", "boundary j", "boundary k"],
                                             skiprows=list_of_no_read)

        # 读取边界信息文件
        number_of_triangle_boundary = pd.read_csv(filename_boundary,header = None, delim_whitespace = True,
                                               nrows = 1).iat[0,0]
        list_of_no_read = [0, number_of_triangle_boundary + 1, number_of_triangle_boundary + 2]
        self.boundary_of_global = pd.read_csv(filename_boundary, delim_whitespace = True,
                                               names=["boundary start", "boundary end",
                                                      "left element", "right element","mark"] ,
                                               skiprows = list_of_no_read)

        # 初始化总刚度矩阵 总载荷矩阵
        self.Ae = np.mat(np.tile([0.], (number_of_triangle_point, number_of_triangle_point)))
        self.f = np.mat(np.tile([0.], (number_of_triangle_point, 1)))


    # 先计算面积积分
    def add_all_element(self):
        for No_of_element in range(self.element_of_global.shape[0]):
            self.elemental_mat(No_of_element)

    def elemental_mat(self,No_of_element):
        """单元刚度矩阵计算并叠加到总刚度矩阵"""
        elemental_point = self.element_of_global.iloc[No_of_element, :]
        self.xi, self.yi = self.point_of_global.iloc[elemental_point[1], 1:3]
        self.xj, self.yj = self.point_of_global.iloc[elemental_point[2], 1:3]
        self.xk, self.yk = self.point_of_global.iloc[elemental_point[3], 1:3]

        self.get_area()
        # 常数矩阵
        BTB = self.get_BTB_of_element()
        # 面积坐标下对应的各函数
        p, q, fu = self.variable_transform_to_element()
        # print(p, q, fu)
        # print(eval(fu))
        for i in range(3):
            # 处理载荷矩阵
            self.f[elemental_point[i + 1]] += \
                integrate.dblquad(lambda L2, L1: self.get_N_of_element(L1, L2)[i] * eval(fu) * 2 * self.area_of_element,
                                  0.0, 1.0, lambda L2: 0.0, lambda L2: 1.0 - L2)[0]
            # print(self.f[elemental_point[i + 1]])
            # dblquad使用可参考http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.integrate.dblquad.html
            for j in range(3):
                self.Ae[elemental_point[i + 1], elemental_point[j + 1]] += \
                    integrate.dblquad(lambda L2, L1: (BTB[i, j] * eval(p) +
                                      eval(q) * self.get_N_of_element(L1, L2)[i] * self.get_N_of_element(L1, L2)[j])
                                      * 2 * self.area_of_element, 0.0, 1.0, lambda L2: 0.0, lambda L2: 1.0 - L2)[0]
                # print(self.Ae[elemental_point[i + 1], elemental_point[j + 1]])

    def get_area(self):
        area_of_element = np.mat([[self.xi, self.yi, 1],
                                  [self.xj, self.yj, 1],
                                  [self.xk, self.yk, 1]], )
        self.area_of_element = np.linalg.det(area_of_element) / 2

    def get_BTB_of_element(self):
        B = np.mat([[self.yj - self.yk, self.yk - self.yi, self.yi - self.yj],
                    [self.xk - self.xj, self.xi - self.xk, self.xj - self.xi]]) / (2 * self.area_of_element)
        BTB = (B.T * B)
        # print(BTB)
        return BTB

    def get_N_of_element(sel, L1, L2):
        """矩阵 N 与 x ，y 有关，而dblquad的输入必须为函数或者方法，故添加此方法 """
        Ni = L1
        Nj = L2
        Nk = 1 - L1 - L2
        N_of_element = np.mat([Ni, Nj, Nk]).T
        return N_of_element

    def variable_transform_to_element(self):
        x_on_element = '(self.xi - self.xk) * L1 + (self.xj - self.xk) * L2 + self.xk'
        y_on_element = '(self.yi - self.yk) * L1 + (self.yj - self.yk) * L2 + self.yk'
        p = self.p.replace( 'x', x_on_element).replace('y', y_on_element)
        q = self.q.replace( 'x', x_on_element).replace('y', y_on_element)
        fu = self.fu.replace( 'x', x_on_element).replace('y', y_on_element)
        return p, q, fu


    # 再计算线积分
    def add_all_line(self):
        # 若σ， β皆为0，不必计算线积分
        if self.sigma == 0 and self.beta == 0:
            return
        for No_of_element in range(self.boundary_of_global.shape[0]):
            self.boundary_mat(No_of_element)

    def boundary_mat(self,No_of_element):
        line_point = self.boundary_of_global.iloc[No_of_element, :]

        # 先判断是否属于自然边界条件所在边
        if line_point["mark"] not in self.nature_boundary_mark:
            return

        self.xi, self.yi = self.point_of_global.iloc[line_point["boundary start"], 1:3] # 0 0
        self.xj, self.yj = self.point_of_global.iloc[line_point["boundary end"], 1:3] # 0.13082502 0.13082502

        # 判断自然边界所在单元,也可以使用边界标志判断
        if line_point["left element"] < 0:
            element_point = self.boundary_of_global.iloc[line_point["right element"], 1:4]
        else:
            element_point = self.boundary_of_global.iloc[line_point["right element"], 1:4]

        # 最后一个节点的编号
        left_point = sum(element_point) - line_point["boundary start"] - line_point["boundary end"]
        self.xk, self.yk = self.point_of_global.iloc[left_point, 1:3]
        # 重新编号
        elemental_point = [line_point["boundary start"], line_point["boundary end"], left_point]

        # 边长
        lenth = np.sqrt((self.xi - self.xj) ** 2 + (self.yi - self.yj) ** 2)

        # 此处面积用于 get_N_of_line 求矩阵 N
        # 此时节点顺序不能保证是顺时针，行列式故需要做绝对值处理
        self.get_area()
        self.area_of_element.__abs__()

        sigma, beta = self.variable_transform_to_line
        for i in range(3):
            self.f[elemental_point[i]] += integrate.quad(lambda L1: lenth * eval(beta) *
                                                         self.get_N_of_line(self, L1)[i], (0, 1))
            for j in range(3):
                self.Ae[elemental_point[i], elemental_point[j]] += \
                    integrate.quad(lambda L1: lenth * eval(sigma) * self.get_N_of_line(self, L1)[i]
                                   * self.get_N_of_line(self, L1)[j], (0, 1))

    def variable_transform_to_line(self):
        x_on_line = 'L1 * self.xi + (1 - L1) * self.xj'
        y_on_line = 'L1 * self.yi + (1 - L1) * self.yj'
        sigma = self.sigma.replace('x', x_on_line).replace('y', y_on_line)
        beta = self.beta.replace('x', x_on_line).replace('y', y_on_line)
        return sigma, beta

    def get_N_of_line(self, L1):
        Ni = L1
        Nj = 1 - L1
        Nk = 0
        N_of_line = np.mat([Ni, Nj, Nk]).T
        return N_of_line


    # 处理本质边界条件
    def deal_with_boundary(self):
        for i in range(self.point_of_global.shape[0]):
            for j in self.essential_boundary_mark:
                # 仅处理有本质边界条件的边
                if self.point_of_global.iat[i, 3] == j:
                    boundary_value = float(self.boundary_value[j - 1])
                    self.f -= self.Ae[:, i] * boundary_value
                    self.Ae[:, i] = 0
                    self.Ae[i, :] = 0
                    self.Ae[i, i] = 1
                    self.f[i] = boundary_value

    # 解有限元方程
    def solve_mat(self,):
        self.u = self.Ae.I * self.f

    # 后处理
    def save_data(self, ):
        save_filename = input("please input filename for save data:")
        self.savefile = open(save_filename + '.pickle', 'wb')
        self.save_data = self.u
        pickle.dump(self.save_data, self.savefile)
        self.savefile.close()

        # 读取方式：
        # with open('pickle_example.pickle', 'rb') as file:
        #     a_dict1 = pickle.load(file)

    def plot_result(self,):
        plt.scatter(self.point_of_global["x"],
                    self.point_of_global["y"],
                    cmap="coolwarm",    #选择颜色 colormap
                    c=(self.u[:, 0] / (self.u[:, 0].max())).tolist()
                    #色彩或者颜色序列 先运行.max，此处类似于归一化操作
                     )
        plt.show()

    def plot2(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = self.point_of_global["x"]
        y = self.point_of_global["y"]
        x, y = np.meshgrid(x, y)
        z = self.u

        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
        plt.show()

    # 运行
    def run(self):
        self.init_grid()
        self.add_all_element()
        self.deal_with_boundary()
        self.solve_mat()
        print("一共有 %s 个节点，所有节点平均值：%s"  %(self.u.shape[0],self.u.mean()))
        self.plot2()


a = Finite_Element_of_triangle()
a.filename='mesh1'

a.run()

