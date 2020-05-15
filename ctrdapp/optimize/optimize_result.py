import matplotlib.pyplot as plt
import matplotlib.animation as animation


class OptimizeResult:

    def __init__(self, best_q, best_solver, success, num_function_eval,
                 num_iterations, completion_time, optimize_process):
        self.best_q = best_q
        self.best_solver = best_solver
        self.success = success
        self.num_function_eval = num_function_eval
        self.num_iterations = num_iterations
        self.completion_time = completion_time
        self.optimize_process = optimize_process  # list of dicts{q, cost}

    def graph_process(self):  # todo
        fig = plt.figure()
        line, = plt.plot([], [], 'r-')
        optimize_iterator = [node_dict.get('q') for node_dict in self.optimize_process]
        optimize_iterator = iter(optimize_iterator)

        def update(data):
            line.set_xdata(data[0])
            line.set_ydata(data[1])
            return line,

        def data_gen():
            while True:
                yield next(optimize_iterator)

        plt.xlim(0, 0.1)
        plt.ylim(0, 0.1)
        plt.title('test')
        line_ani = animation.FuncAnimation(fig, update, data_gen, blit=True, interval=50)
        # writer = FFMpegWriter(fps=15, metadata=dict(artist='Conor'), bitrate=1800)
        # line_ani.save("mygraph.mp4", writer=writer)

        # plt.savefig("mygraph.png")
        plt.show()
