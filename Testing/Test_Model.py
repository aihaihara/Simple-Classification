#++++++++++++++++++++++++++++++Trying Model++++++++++++++++++++++++++++++++++++
#now time to try our model to new data
#import the preprocessing libraries
import pickle, numpy
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#create stopword remover
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

#create stemmer
stem = StemmerFactory()
stemmer = stem.create_stemmer()

#making function for preprocessing
def prep(text):
  text = stopword.remove(text)
  text = stemmer.stem(text)
  return " ".join(text.split())

#try new input
#defining label (you can change it according to your data)
#PS: in this case, we classify data into 3 classes (Hotel, Food, and Travel)
label = {0:'Hotel',1:'Food',2:'Travel'}
#example=['BANDUNG, KOMPAS.com - Kabar gembira bagi Anda yang gemar makan bakso. Pekan ini akan kembali hadir Festival Baso Juara yang diadakan di Jalan Dr Sukarno (Cikapundung Timur), Kota Bandung, Jawa Barat. Festival Baso Juara akan diadakan selama tiga hari yaitu 7-9 Desember 2018 mulai pukul 09.00-21.00 WIB. Selama tiga hari penyelenggaraan, setiap harinya Festival Baso Juara akan memiliki tema yang berbeda-beda. Misalnya saja di hari Jumat mengusung tema Jumat Berkah, Sabtu temanya tentang kebudayaan, dan Minggu temanya tentang family day. Baca juga: Rekomendasi 5 Kuliner Bakso di Bandung Wajib Dicoba Ketua Penyelenggara Festival Baso Juara, Ratih mengatakan, adanya tema yang berbeda-beda ini dilakukan supaya adanya keunikan di festival kuliner baso tahunan ini. "Di Jumat Berkah, kami akan mengajak 100 anak yatim, lalu keesokan harinya akan ada pasanggiri abah ambu, dan di hari terakhir akan ada zumba," ujar Ratih saat menggelar konferensi pers di Hotel Bidakara Grand Savoy Homann, Jalan Asia Afrika No 112, Kota Bandung, Rabu (5/12/2018). Baca juga: Bakso Raksasa 55 Kilogram Harga Rp 6 Juta-Rp 8 Juta, Apa Saja Isinya? Festival Baso Juara akan diisi oleh 40 tenan bakso yang berasal dari Bandung, Garut, dan Tasikmalaya. Ratih mengatakan, untuk pemilihan tenan bakso yang ikut festival ini tentunya sudah masuk standar kualifikasi. Syarat yang disetujui di antaranya adalah tenan tersebut telah berjualan bakso selama 6 bulan, memiliki kedai baso, dan ada jam operasional untuk kedai basonya. Konferensi pers Festival Baso Juara di Hotel Bidakara Grand Savoy Homann, Jalan Asia Afrika No 112, Kota Bandung, Jawa Barat, Rabu (5/12/2018). Konferensi pers Festival Baso Juara di Hotel Bidakara Grand Savoy Homann, Jalan Asia Afrika No 112, Kota Bandung, Jawa Barat, Rabu (5/12/2018). (TRIBUNJABAR.ID/PUTRI PUSPITA) "Hal ini untuk memudahkan pengunjung yang datang ke festival. Ketika mereka ingin coba makan basonya lagi selain di festival kan bisa datang langsung ke kedai basonya," ujar Ratih. Berbagai jenis bakso yang hits dan sudah teruji rasanya di Bandung bisa Anda temukan di sini, seperti mie ayam cipaganti, lomie dan bakmi imam bonjol, baso sumsum, baso beranak durian, dan lainnya. Hal yang perlu diperhatikan bagi pengunjung adalah transaksi Festival Baso Juara masing-masing menggunakan vocer bernilai Rp 5.000, Rp 10.000, Rp 20.000 dan Rp 50.000. (Tribun Jabar)']

#call function to tell user to input new document directly from keyboard
example = [] #new variable for input list
doc = input('Enter ur document:\n') #call the input function
example.append(doc) #append the string input to list

#preprocessing step for the input data
for line in example:
    test= prep(line)
    a =[]
    a.append(test)

#extracting the features function
with open ('features.pkl', 'rb') as features:
    count_feat = pickle.load(features)
test_s = count_feat.transform(a) #converting the document to matrix
#print(test_s)

#open the model
with open ('your pickled model *.pkl', 'rb') as clf:
    model = pickle.load(clf)

#predict the new input doc using the model
print("The text that u have input is: %s with probability: %.2f%%" % (label[model.predict(test_s)[0]], numpy.max(model.predict_proba(test_s))*100))

#+++++++++++++++++++++++++++++Finish+++++++++++++++++++++++++++++++++++++++++++