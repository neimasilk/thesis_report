import numpy as np
import matplotlib.pyplot as plt
import thesis.ekstrak_csv as eks

kamus = {"Pre-training layer": "", "epoch": "", "cost": "", "\n": ""}


def replace_all(text, dic):
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text


def load_file_text(nama_file):
    text_file = open(nama_file, "r")
    lst = text_file.readlines()
    a = np.array([replace_all(lst[0], kamus).split(",")], float)
    for i in range(1,len(lst)):
        b = np.array([replace_all(lst[i], kamus).split(",")], float)
        a = np.r_[a,b]
    return a

def load_epch_layer(mat, jml_epoch, layer):
    return mat[(layer*jml_epoch):((layer+1)*jml_epoch),1:3]

def load_file_ekstrak_layer_epoch_cost(nama_file_log_test):
    c = load_file_text(nama_file_log_test)
    return c

if __name__ == '__main__':
    # contoh pemakaian load matrix
    # edit file log sampai hanya ada layer epoch dan cost saja
    # simpan dengan kode jml epoch layer
    # load file log dengan :

    # f = load_file_ekstrak_layer_epoch_cost("../thesis_test/dataset_test/log_test.log")
    # g = load_epch_layer(f, 2, 1)  # g = matrix dengan isi epoch dan cost pada layer 2


    f = load_file_ekstrak_layer_epoch_cost("../thesis/dataset/log1000e_3k_2k_1k_100.txt")
    ekstraktor = eks.Ekstraktor()

    # ekstraktor.simpan_data("../thesis/dataset/1000e_3k_2k_1k_100_lyr1",g)
    # plot layer 1
    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 3k_2k_1k_100 Layer 0 (3000 Hidden)")
    g_0 = load_epch_layer(f,1000,0)
    plt.plot(g_0[10:,0],g_0[10:,1])
    plt.show()

    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 3k_2k_1k_100 Layer 1 (2000 Hidden)")
    g_1 = load_epch_layer(f, 1000, 1)
    plt.plot(g_1[10:, 0], g_1[10:, 1])
    plt.show()

    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 3k_2k_1k_100 Layer 2 (1000 Hidden)")
    g_2 = load_epch_layer(f, 1000, 2)
    plt.plot(g_2[10:, 0], g_2[10:, 1])
    plt.show()

    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 3k_2k_1k_100 Layer 3 (100 Hidden)")
    g_3 = load_epch_layer(f, 1000, 3)
    plt.plot(g_3[10:, 0], g_3[10:, 1])
    plt.show()

    f = load_file_ekstrak_layer_epoch_cost("../thesis/dataset/log1000e_7k_10k_5k_1k.txt")
    ekstraktor = eks.Ekstraktor()
    #
    # # ekstraktor.simpan_data("../thesis/dataset/1000e_3k_2k_1k_100_lyr1",g)
    # # plot layer 1
    # ekstraktor.simpan_data("../thesis/dataset/1000e_3k_2k_1k_100_lyr1",g)
    # plot layer 1
    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 7k_10k_5k_1k Layer 0 (7000 Hidden)")
    g_0 = load_epch_layer(f, 1000, 0)
    plt.plot(g_0[10:, 0], g_0[10:, 1])
    plt.show()

    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 7k_10k_5k_1k Layer 1 (10000 Hidden)")
    g_1 = load_epch_layer(f, 1000, 1)
    plt.plot(g_1[10:, 0], g_1[10:, 1])
    plt.show()

    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 7k_10k_5k_1k Layer 2 (5000 Hidden)")
    g_2 = load_epch_layer(f, 1000, 2)
    plt.plot(g_2[10:, 0], g_2[10:, 1])
    plt.show()

    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 7k_10k_5k_1k Layer 3 (1000 Hidden)")
    g_3 = load_epch_layer(f, 1000, 3)
    plt.plot(g_3[10:, 0], g_3[10:, 1])
    plt.show()

    f = load_file_ekstrak_layer_epoch_cost("../thesis/dataset/log1000e_10k_5k_1k_500.txt")
    ekstraktor = eks.Ekstraktor()
    #
    # # ekstraktor.simpan_data("../thesis/dataset/1000e_3k_2k_1k_100_lyr1",g)
    # # plot layer 1
    # ekstraktor.simpan_data("../thesis/dataset/1000e_3k_2k_1k_100_lyr1",g)
    # plot layer 1
    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 10k_5k_1k_500 Layer 0 (10000 Hidden)")
    g_0 = load_epch_layer(f, 1000, 0)
    plt.plot(g_0[10:, 0], g_0[10:, 1])
    plt.show()

    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 10k_5k_1k_500 Layer 1 (5000 Hidden)")
    g_1 = load_epch_layer(f, 1000, 1)
    plt.plot(g_1[10:, 0], g_1[10:, 1])
    plt.show()

    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 10k_5k_1k_500 Layer 2 (1000 Hidden)")
    g_2 = load_epch_layer(f, 1000, 2)
    plt.plot(g_2[10:, 0], g_2[10:, 1])
    plt.show()

    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.title("Pretraining model 10k_5k_1k_500 Layer 3 (500 Hidden)")
    g_3 = load_epch_layer(f, 1000, 3)
    plt.plot(g_3[10:, 0], g_3[10:, 1])
    plt.show()
