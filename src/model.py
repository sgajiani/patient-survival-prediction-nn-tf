import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False

import seaborn as sns
dark2 = sns.color_palette('Dark2')


def plot_model_auc_loss(modelname, train_auc, val_auc, train_loss, val_loss, epochs):
    epoch_range = range(epochs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    ax1.plot(epoch_range, train_auc, label='Train AUC score', color=dark2[0])
    ax1.plot(epoch_range, val_auc, label='Validation AUC score', color=dark2[1])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('AUC')
    ax1.set_title(modelname + '- Training and Validation AUC')
    ax1.legend()

    ax2.plot(epoch_range, train_loss, label='Train Loss', color=dark2[0])
    ax2.plot(epoch_range, val_loss, label='Validation Loss', color=dark2[1])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(modelname + '- Training and Validation Loss')
    ax2.legend()

    plt.show()
