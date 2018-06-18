import pandas as pd  # 使用pandas读取和处理数据
import numpy as np   # 使用numpy进行矩阵运算
from matplotlib import pyplot as plt     # 二维作图
from mpl_toolkits.mplot3d import Axes3D  # 三维作图
import pickle           # 数据存储
from sympy import *    # 数学符号计算库


class Finite_Element_of_triangle():
    # 初始化本质边界条件及读取网格剖分
    def __init__(self):
        '输入文件夹名字'
        self.filename = 'mesh3'
        # 记录有本质边界条件与自然边界条件的边及值
        self.essential_boundary_mark = [1, 3]
        self.boundary_value = ['50', '', '100', '']

    def init_grid(self):
        '自动识别文件夹中三种数据文件，并以Pandas特征数据返回'
        filename_point = './' + self.filename + '/E.n'
        filename_element = './' + self.filename + '/E.e'
        # 感觉.s文件没什么用就不读取了
        # filename_boundary = './' + self.filename + '/E.s'

        number_of_triangle_point = pd.read_csv(filename_point,header = None, delim_whitespace = True,
                                               nrows = 1).iat[0,0]
        # pd.read_csv参数参考http://www.cnblogs.com/datablog/p/6127000.html

        self.Ae = np.mat(np.tile([0.], (number_of_triangle_point, number_of_triangle_point)))
        self.f = np.mat(np.tile([0.], (number_of_triangle_point, 1)))
        # 初始化总刚度矩阵 总载荷矩阵

        list_of_no_read = [0, number_of_triangle_point+1, number_of_triangle_point+2]
        # 第一行与最后两行数据无需读取


        self.point_of_global = pd.read_csv(filename_point, delim_whitespace = True,
                                           names=["point NO","x","y","boundary mark"],
                                           skiprows = list_of_no_read)

        number_of_triangle_element = pd.read_csv(filename_element,header=None,
                                                 delim_whitespace = True,nrows=1).iat[0,0]
        list_of_no_read = [0, number_of_triangle_element + 1, number_of_triangle_element + 2]
        self.element_of_global = pd.read_csv(filename_element, delim_whitespace=True,
                                             names=["element number", "point i", "point j", "point k",
                                                    "neighbor ele i",  "neighbor ele j",  "neighbor ele j",
                                                    "boundary i", "boundary j", "boundary j"],
                                             skiprows=list_of_no_read)

        # 感觉.s文件没什么用就不读取了
        # number_of_triangle_boundary = pd.read_csv(filename_boundary,header = None, delim_whitespace = True,
        #                                        nrows = 1).iat[0,0]
        # list_of_no_read = [0, number_of_triangle_boundary + 1, number_of_triangle_boundary + 2]
        # self.boundary_of_triangle = pd.read_csv(filename_boundary, delim_whitespace = True,
        #                                        names=["boundary start", "boundary end",
        #                                               "left element", "right element","mark"] ,
        #                                        skiprows = list_of_no_read)

    # 计算总刚度矩阵：叠加单元刚度矩阵
    def add_all_element(self):
        for No_of_element in range(self.element_of_global.shape[0]):
            self.elemental_mat(No_of_element)

    # 计算单元刚度矩阵
    def elemental_mat(self,No_of_element):
        '单元刚度矩阵计算并叠加到总刚度矩阵'
        elemental_point = self.element_of_global.iloc[No_of_element,:]
        xi, yi = self.point_of_global.iloc[elemental_point[1],1:3]
        xj, yj = self.point_of_global.iloc[elemental_point[2],1:3]
        xk, yk = self.point_of_global.iloc[elemental_point[3],1:3]

        area_of_element = np.mat([[xi, yi, 1],
                                  [xj, yj, 1],
                                  [xk, yk, 1]],)
        area_of_element = np.linalg.det(area_of_element)/2

        B = np.mat([[yj-yk, yk-yi, yi-yj],
                    [xk-xj, xi-xk, xj-xi]],)/(2*area_of_element)

        # 单元刚度矩阵
        Ae_of_element =(B.T * B) * area_of_element

        # 将单元刚度矩阵叠加到总刚度矩阵
        for i in range(3):
            for j in range(3):
                self.Ae[elemental_point[i + 1], elemental_point[j + 1]] += Ae_of_element[i, j]

    # 处理本质边界条件
    def deal_with_boundary(self):
        for i in range(self.point_of_global.shape[0]):
            for j in self.essential_boundary_mark:
                # 仅处理有本质边界条件的边
                if self.point_of_global.iat[i, 3] == j :
                    boundary_value = float(self.boundary_value[j - 1])
                    self.f -= self.Ae[:, i] * boundary_value
                    self.Ae[:, i] = 0
                    self.Ae[i, :] = 0
                    self.Ae[i, i] = 1
                    self.f[i] = boundary_value

    # 求解方程
    def solve_mat(self,):
        self.u = self.Ae.I * self.f
        self.u = pd.DataFrame(self.u)
        self.u.columns = ["节点值"]

    # 存储数据
    def save_data(self, save_filename):
        self.savefile = open(save_filename + '.pickle', 'wb')
        self.save_data = self.u
        pickle.dump(self.save_data, self.savefile)
        self.savefile.close()
        # 读取方式：
        # with open('pickle_example.pickle', 'rb') as file:
        #     a_dict1 = pickle.load(file)

    # 作热点图
    def plot_result(self,):
        plt.scatter(self.point_of_global["x"],
                    self.point_of_global["y"],
                    cmap="rainbow",    #选择颜色 colormap
                    c=(self.u[:, 0] / (self.u[:, 0].max())).tolist()
                    #色彩或者颜色序列 先运行.max，此处类似于归一化操作
                     )
        plt.show()

    # 作三维图
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
        print(self.u.describe())
        self.plot2()

class Finite_Element_of_rect(Finite_Element_of_triangle):

    def __init__(self):
        self.x_size, self.y_size = 2., 2.
        self.row_size_of_point, self.series_size_of_point = 10, 10
        self.x = symbols("x", real=True)
        self.y = symbols("y", real=True)
        self.essential_boundary_mark = [1, 3]
        self.boundary_value = ['50', '', '100', '']

    def init_grid(self):
        # 初始化总刚度矩阵 总载荷矩阵
        self.x_segment = self.x_size / (self.row_size_of_point - 1)
        self.y_segment = self.y_size / (self.series_size_of_point - 1)
        self.Ae = np.mat(np.tile([0.], (self.row_size_of_point * self.series_size_of_point,
                                        self.row_size_of_point * self.series_size_of_point)))
        self.f = np.mat(np.tile([0.], (self.row_size_of_point * self.series_size_of_point, 1)))
        self.init_point()
        self.init_element()

    def init_point(self):
        # initialize point
        self.point_of_global = np.mat(np.tile([0, 0., 0., 0], (self.row_size_of_point * self.series_size_of_point, 1)))
        # point No.
        self.point_of_global[:, 0] = np.mat(range(self.row_size_of_point * self.series_size_of_point)).T
        # point y
        for i in range(self.series_size_of_point):
            self.point_of_global[self.row_size_of_point * i:self.row_size_of_point * (i+1), 2] = i * self.y_segment
        # point x
        for i in range(self.row_size_of_point):
            self.point_of_global[i:: self.row_size_of_point, 1] = i * self.x_segment
        # set boundary mark
        self.point_of_global[0:: self.series_size_of_point, 3] = 4
        self.point_of_global[self.row_size_of_point:: self.series_size_of_point, 3] = 2
        self.point_of_global[0: self.row_size_of_point, 3] = 1
        self.point_of_global[-self.row_size_of_point:, 3] = 3
        self.point_of_global = pd.DataFrame(self.point_of_global)
        self.point_of_global.columns = ["point NO","x","y","boundary mark"]

    def init_element(self):
        self.element_of_global = np.mat(np.tile([0], ((self.row_size_of_point - 1) * (self.series_size_of_point - 1), 5)))
        # element No.
        self.element_of_global[:, 0] = np.mat(range((self.row_size_of_point - 1) * (self.series_size_of_point - 1))).T
        # point No. of element (Counterclockwise)
        for i in range(self.series_size_of_point - 1):
            for j in range(self.row_size_of_point - 1):
                self.element_of_global[i * (self.row_size_of_point-1) + j,1:] \
                    = [(self.row_size_of_point) * i + j,
                       (self.row_size_of_point) * i + j + 1,
                       (self.row_size_of_point) * (i + 1) + j + 1,
                       (self.row_size_of_point) * (i + 1) + j]
        self.element_of_global = pd.DataFrame(self.element_of_global)
        self.element_of_global.columns = ["element number", "point i", "point j", "point k","point l"]

    def elemental_mat(self,No_of_element):
        '单元刚度矩阵生成并叠加到总刚度矩阵'
        elemental_point = self.element_of_global.iloc[No_of_element, :]
        x0, y0 = self.point_of_global.iloc[elemental_point[1], 1:3]
        x1, y1 = self.point_of_global.iloc[elemental_point[3], 1:3]
        area_of_element = self.x_segment * self.y_segment
        B = np.mat([[self.y - y0 - self.y_segment, self.y_segment + y0 - self.y, y0 - self.y, self.y - y0],
                    [self.x - x0 - self.x_segment, x0 - self.x, self.x_segment + x0 - self.x, self.x - x0]]) / area_of_element
        Ae_of_element = B.T * B
        for i in range(4):
            for j in range(4):
                self.Ae[elemental_point[i + 1], elemental_point[j + 1]] += \
                    integrate(Ae_of_element[i, j], (self.x, x0, x1), (self.y, y0, y1))


a = Finite_Element_of_rect()
a.row_size_of_point = 30
a.series_size_of_point = 30

# a = Finite_Element_of_triangle()
# a.filename = 'mesh2'

a.run()






