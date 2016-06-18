from sklearn import preprocessing
from sklearn import utils
import numpy as np
import gzip, cPickle
from utilitas import top_n_dataset

class Salah(Exception):
    pass

class Ekstraktor:
    nama_file = str
    data = np.empty
    target_file = str
    y = np.empty
    jumlah_data = int
    def norm_dataset(self,nama_file):
        self.nama_file = nama_file + ".csv"
        self.data = np.genfromtxt(self.nama_file, dtype=float, delimiter=",")
        min_max_scaler = preprocessing.normalize(self.data)
        #min_max_scaler = preprocessing.scale(self.data)
        #min_max_scaler = preprocessing.minmax_scale(self.data)
        np.savetxt(nama_file + "_norm.csv", min_max_scaler, delimiter=",")

    def generate_dataset(self, nama_file, target_file,train,valid,test, suffle = True):
        self.nama_file = nama_file + ".csv"
        self.target_file = target_file + ".csv"
        self.data = np.genfromtxt(self.nama_file, dtype=float, delimiter=',')
        self.y = np.genfromtxt(self.target_file, dtype=float, delimiter=',')
        self.data = self.data.transpose()
        self.jumlah_data = self.ambil_jumlah_dataset(self.data)
        jml_train,jml_valid,jml_test =self.ambil_train_valid_test(self.jumlah_data,train,valid,test)
        if suffle:
            self.data, self.y = utils.shuffle(self.data,self.y,random_state = 5)
        train_set_x = self.data[0:jml_train]
        valid_set_x = self.data[jml_train+1:jml_train+1+jml_valid]
        test_set_x = self.data[jml_train+1+jml_valid+1:jml_train+1+jml_valid+1+jml_test]
        train_set_y = self.y.transpose()[2][0:jml_train]
        valid_set_y = self.y.transpose()[2][jml_train+1:jml_train+1+jml_valid]
        test_set_y = self.y.transpose()[2][jml_train+1+jml_valid+1:jml_train+1+jml_valid+1+jml_test]
        train_set = train_set_x, train_set_y
        valid_set = valid_set_x, valid_set_y
        test_set = test_set_x, test_set_y
        dataset = [train_set, valid_set, test_set]
        self.simpan_data(self.nama_file + '_dataset.pkl.gz',dataset)
        return dataset

    def ambil_jumlah_dataset(self,data):
        return data.shape[0]

    def ambil_train_valid_test(self,jml_dataset,train,valid,test):
        # ambil train valid test dalam %
        if int(round(train+valid+test)) != 100 :
            raise Salah("train+valid+test harus = 100%")
        jml_train_set = int(round(float(jml_dataset)*(float(train)/100.)))
        jml_valid_set = int(round(float(jml_dataset)*(float(valid)/100.)))
        jml_test_set  = int(round(float(jml_dataset)*(float(test)/100.)))
        return jml_train_set,jml_valid_set,jml_test_set

    def simpan_data(self, n_file, data_simpan):
        f = gzip.open(n_file, 'wb')
        cPickle.dump(data_simpan, f, protocol=2)
        f.close()
        return data_simpan

    def load_data(self, data):
        # model_hasil = load cpickel
        f = gzip.open(data, 'rb')
        model_hasil = cPickle.load(f)
        return model_hasil

class Generator:
    ekstraktor = Ekstraktor()
    # data_rank adalah array dari ranking data
    def top_n_dataset(self, data_rank,dataset, namafile):
        data_hasil = top_n_dataset(data_rank, dataset)
        np.savetxt(namafile + ".csv", data_hasil, delimiter=",")
        return data_hasil

if __name__ == '__main__':
    ekstraktor = Ekstraktor()
    generator = Generator()
    array_rank= np.array([2, 3])
    train = 80.5
    valid = 14.5
    test = 5
    ekstraktor.norm_dataset("./dataset/iris_dataset")
    dataset_iris = np.genfromtxt("./dataset/iris_dataset_norm.csv", dtype=float, delimiter=",")
    generator.top_n_dataset(array_rank, dataset_iris,"./dataset/iris_dataset_rank")
    dataset_iris = ekstraktor.generate_dataset("./dataset/iris_dataset_rank",
                                          "./dataset/iris_target", train, valid, test, True)

    print dataset_iris
    # ekstraktor.norm_dataset("./dataset/GSE10072_dataset")
