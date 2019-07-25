import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
import scipy.sparse
import scipy.sparse.linalg
from mpl_toolkits.mplot3d import axes3d


class HyperbolicSolver:
    def __init__(self, grid_s, grid_t, initial_cond, edges_cond, F, sigmas):
        self.sigma1 = sigmas[0]
        self.sigma2 = sigmas[1]
        self.space_dim = len(grid_s)
        self.F = F
        self.delta_s = []
        for i in grid_s:
            self.delta_s.append(i[1]-i[0])
        self.delta_t = grid_t[1] - grid_t[0]
        self.grid_t = grid_t
        self.grid_s = grid_s
        self.initial_cond = initial_cond
        self.edges_cond = edges_cond
        self.T = len(self.grid_t)
        self.N = [len(i) for i in grid_s] if self.space_dim == 2 else len(self.grid_s[0])
        self.S = []
        self.__propagateInitial()
        

    def __propagateInitial(self):
        if self.space_dim == 1:
            self.S.append([self.initial_cond[0](x) for x in self.grid_s[0]])
            self.S.append([self.initial_cond[0](x) + self.initial_cond[1](x)*self.delta_t for x in self.grid_s[0]])
            
        elif self.space_dim == 2:
            self.S.append([[self.initial_cond[0](x, y) for y in self.grid_s[1]] for x in self.grid_s[0]])
            self.S.append([[self.initial_cond[0](x, y) + self.initial_cond[1](x, y)*self.delta_t for y in self.grid_s[1]] for x in self.grid_s[0]])


    def __RHS(self, s, t):
        x = None
        y = None
        if self.space_dim == 1:
            i = s
            
            return (2*self.S[t][i] - self.S[t-1][i])/(self.delta_t**2) + self.sigma1*self.F(self.grid_s[0][i], self.grid_t[t+1]) + \
                 (1-self.sigma1-self.sigma2)*((self.S[t][i+1] - 2*self.S[t][i] + self.S[t][i-1])/(self.delta_s[0]**2) + \
                 self.F(self.grid_s[0][i], self.grid_t[t])) + self.sigma2*((self.S[t-1][i+1] -\
                 2*self.S[t-1][i] + self.S[t-1][i-1])/(self.delta_s[0]**2) + \
                 self.F(self.grid_s[0][i], self.grid_t[t-1]))

        elif self.space_dim == 2:
            i = s[0]
            j = s[1]

            return (2*self.S[t][i][j] - self.S[t-1][i][j])/(self.delta_t**2) + self.sigma1*self.F(self.grid_s[0][i], self.grid_s[1][j], self.grid_t[t+1]) + \
                 (1-self.sigma1-self.sigma2)*((self.S[t][i+1][j] - 2*self.S[t][i][j] + self.S[t][i-1][j])/(self.delta_s[0]**2) + \
                 (self.S[t][i][j+1] - 2*self.S[t][i][j] + self.S[t][i][j-1])/(self.delta_s[1]**2) + \
                 self.F(self.grid_s[0][i], self.grid_s[1][j], self.grid_t[t])) + \
                 self.sigma2*((self.S[t-1][i+1][j] - 2*self.S[t-1][i][j] + self.S[t-1][i-1][j])/(self.delta_s[0]**2) + \
                 (self.S[t-1][i][j+1] - 2*self.S[t-1][i][j] + self.S[t-1][i][j-1])/(self.delta_s[1]**2) + \
                 self.F(self.grid_s[0][i], self.grid_s[1][j], self.grid_t[t-1]))


    def solve(self):
        if self.space_dim == 1:
            delta_x = self.delta_s[0]
            A = scipy.sparse.diags([[-self.sigma1/(delta_x**2)], [1/(self.delta_t**2) + 2*self.sigma1/(delta_x**2)], [-self.sigma1/(delta_x**2)]], [1,0,-1], [self.N-2, self.N-2], "csc")
    
            for t in range(1, self.T-1):
                b = np.array([self.__RHS(i, t) for i in range(1,self.N-1)])
                x = scipy.sparse.linalg.spsolve(A, b)
                X = np.zeros(self.N)
                X[1:self.N-1] = x
                X[0] = self.edges_cond[0](self.grid_t[t])
                X[self.N-1] = self.edges_cond[1](self.grid_t[t])
                self.S.append(X)
        elif self.space_dim == 2:
            delta_x = self.delta_s[0]
            delta_y = self.delta_s[1]
            N = self.N[0]
            M = self.N[1]
            A =  scipy.sparse.diags([[-self.sigma1/(delta_x**2)], [-self.sigma1/(delta_y**2)], [1/(self.delta_t**2) + 2*self.sigma1/(delta_x**2) + 2*self.sigma1/(delta_y**2)], [-self.sigma1/(delta_y**2)], [-self.sigma1/(delta_x**2)]], [(M), 1, 0,-1, -(M)], [(N-2)*(M-2), (N-2)*(M-2)], "csc")

            for t in range(1, self.T-1):
                b = np.array([[self.__RHS((i, j), t) for j in range(1, M-1)] for i in range(1, N-1) ]).reshape((N-2)*(M-2))

                x = scipy.sparse.linalg.spsolve(A, b).reshape(N-2, M-2)
                #x = np.linalg.solve(A.todense(), b).reshape(N-2, M-2)
                X = np.zeros((N, M))
                X[1:N-1, 1:M-1] = x
                #X[0] = self.edges_cond[0](self.grid_t[t])
                #X[self.N-1] = self.edges_cond[1](self.grid_t[t])
                self.S.append(X)
            #print(np.array([[self.__RHS((i, j), 1) for j in range(1, M-1)] for i in range(1, N-1) ]))
        return self.S


    def animate(self, true_func=None, save=False):
        if self.space_dim == 1:
            fig = plt.figure()
            fig.suptitle("Propagation", fontsize=18)
            ax = plt.axes(xlim=(0, 1), ylim=(-0.5, 0.5))
            S_true = None
            if true_func:
                S_true = [[true_func(x, t) for x in self.grid_s[0]] for t in self.grid_t]
                ax2=plt.axes(xlim=(0, 1), ylim=(-0.5, 0.5))
            line, = ax.plot([], [], lw=1)
            if true_func:
                line2, = ax2.plot([], [], lw=1)
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
            def init():
                line.set_data([], [])
                time_text.set_text('')
                if true_func:
                    line2.set_data([], [])
                    return line, line2
                return line, 
            def animate(i):
                x = self.grid_s[0]
                y = self.S[i]
                line.set_data(x, y)
                time_text.set_text('time = %.3f' % self.grid_t[i])
                if true_func:
                    line2.set_data(x, S_true[i])
                    return line, line2, time_text
                return line, time_text
    
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=self.T, interval=self.delta_t*1000, blit=True)
            if save:
                anim.save('string.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            plt.show()

        elif self.space_dim ==2:
            fig = plt.figure()
            fig.suptitle("Propagation", fontsize=18)
            ax = axes3d.Axes3D(fig)
            X = self.grid_s[0]
            Y = self.grid_s[1]
            X, Y = np.meshgrid(X, Y)
            Z = np.array(self.S[0])
            wframe = ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
            ax.set_zlim(-1,1)

            def update(i, ax, fig):
                ax.cla()
                Z = np.array(self.S[i])
                wframe = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
                ax.set_zlim(-1,1)
                return wframe, 
            ani = animation.FuncAnimation(fig, update, frames=self.T, fargs=(ax, fig), interval=100)
            plt.show()


    def saveIntoImg(self, t, true_func):

        if self.space_dim == 1:
            for i in range(len(t)):
                fig = plt.figure()
                plt.plot(self.grid_s[0], self.S[int(t[i]/self.delta_t)])
                plt.title('Time: ' + str(t[i]))
                
                if true_func:
                    S_true = [[true_func(x, y) for x in self.grid_s[0]] for y in self.grid_t]
                    plt.plot(self.grid_s[0], S_true[int(t[i]/self.delta_t)])

                plt.axis([0, 1, -0.5, 0.5])
                plt.savefig('f' + str(i) + '.png')
                plt.close(fig)

        if self.space_dim == 2:
            for i in range(len(t)):
                fig = plt.figure()
                fig.suptitle("Time: " + str(t[i]), fontsize=18)
                ax = axes3d.Axes3D(fig)
                X = self.grid_s[0]
                Y = self.grid_s[1]
                X, Y = np.meshgrid(X, Y)
                Z = np.array(self.S[int(t[i]/self.delta_t)])
                wframe = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
                ax.set_zlim(-1,1)
                plt.savefig('d' + str(i) + '.png')
                plt.close(fig)


    def saveErrorIntoImg(self, t, true_func):
        if self.space_dim == 1:

            for i in range(len(t)):
                fig = plt.figure()
                plt.title('Time: ' + str(t[i]))
                S_true = [[true_func(x, y) - self.S[int(y/self.delta_t)][int(x/self.delta_s[0])] for x in self.grid_s[0]] for y in self.grid_t]
                plt.plot(self.grid_s[0], S_true[int(t[i]/self.delta_t)])
                plt.axis([0, 1, -0.05, 0.05])
                plt.savefig('fe' + str(i) + '.png')
                plt.close(fig)
            

    def reset(self):
        del self.S
        S = []
        self.__propagateInitial()   
