import matplotlib.pyplot as plt


def visualize_train_results(H):

    epoch = H.history['loss']
    plt.figure(figsize=(8,10))
    plt.subplot(121)
    plt.plot(epoch, H.history['loss'], label='train loss')
    plt.plot(epoch, H.history['val_loss'], label= 'validation loss')

    plt.subplot(122)
    plt.plot(epoch, H.history['accuracy'], label='train accuracy')
    plt.plot(epoch, H.history['val_accuracy'], label='validation accuracy')

    plt.legend()
    plt.show()
