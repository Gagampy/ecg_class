from matplotlib import pyplot as plt


class Visualizator:
    def __init__(self, ecg_obj):
        self.__ecg_obj = ecg_obj

    @staticmethod
    def create_figure():
        plt.figure()
        return plt.gca()

    def visul_rr(self, fig):
        pass

    def visul_ecg(self, fig, r=False, c_rec=None, c_r='r'):
        fig.plot(self.__ecg_obj.filtered_record, c=c_rec)
        if r:
            ylim = fig.get_ylim()
            for cur_r in self.__ecg_obj.rpeaks:
                fig.plot((cur_r, cur_r), ylim, c=c_r, linewidth=0.7, linestyle='--')

    def visul_templates(self, fig):
        templates = self.__ecg_obj.templates.copy().reshape((1, -1))
        print(templates.shape)
        fig.plot(templates[0])
