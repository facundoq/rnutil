import matplotlib.pyplot as plt
import matplotlib

import matplotlib.pyplot as plt

plt.style.use('ggplot')


def get_categorical_colormap(n_classes):

    colors = ["tab:blue","tab:orange","tab:green", "tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:cyan","tab:olive","lime","gold","yellow","fuchsia"]
    if n_classes<=len(colors):
        colors = colors[:n_classes]
        cmap=matplotlib.colors.ListedColormap(colors)
    else:
        cmap = matplotlib.cm.get_cmap('jet')
    return cmap
    