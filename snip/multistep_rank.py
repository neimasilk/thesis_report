
# perkalian matrix rank weight
import numpy as np

def awal(w):
    return np.ones((w.shape[1],), dtype=np.float)

def jumlah_bobot(w, top_ke_n):
    # kalikan w dengan matrix 1
    return w.dot(top_ke_n)

def rank_hasil_jumlah(sum_w):
    # urutkan sum_w dan beri index
    """

    :rtype sum_w : numpy.array
    """
    swi = sum_w.shape[0]
    hsl = np.arange(swi)
    c = np.concatenate((hsl,sum_w))
    c = c.reshape(2,swi)
    c = c.T
    z = c[c[:,1].argsort()[::-1]] # urutkan descending berdasarkan bobot (indeks mengikuti)
    return z

def set_top_n(idx_sum_w, top_n = 2):
    # set = 0 semua yang bukan top n
    # kembalikan ke urutan semula
    z = idx_sum_w.copy()
    z[top_n:,1] = 0.
    z[0:top_n,1]= 1.
    # print 'z adalah'
    # print z
    d = z[z[:,0].argsort()[::]]
    # print 'd adalah'
    # print d
    return d

# set_rank : melakukan setting 1 untuk top n dan
def extract_top_n(n):
    return n[:,1]

def set_index_dengan_gen(bobot_akhir):
    # index gen dengan urutan perankingannya
    pass

def plot_diagram(a, b):
    # plot himpunan a dan b dan anggota keduanya
    pass


if __name__ == '__main__':
    # w1 adalah bobot untuk testing
    w1 = np.array([[0, 1, 2, 3, 4],
                   [5, 6, 7, 8, 9],
                   [10, 11, 12, 13, 14]])
    a = awal(w1)
    x = jumlah_bobot(w1,a)   # x = perhitungan bobot berdasarkan h ( 10, 35, 60)
    y = rank_hasil_jumlah(x) # (diberi index dan diranking)
    z = set_top_n(y,1)
    print y
    # print x.shape
    # print y  # matrix penjumlahan bobot diranking sebelum diambil top N
    # print z  # matrix penjumlahan bobot setelah diranking dan diset 0 untuk yg bukan top N
    # print extract_top_n(z)


