import os
import cv2
import numpy as np
import pandas as pd
import pickle
import webbrowser
import joblib
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from shutil import copy2
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from PIL import ImageTk
from PIL import Image

#define root
root=Tk()
root.geometry("800x300")
root.title("Skripsi")
#import model
data80=pd.read_csv('models/word_220.csv')
with open('neigh220.pkl', 'rb') as file:
	neigh = pickle.load(file)
with open('models/mo,pkl', 'rb') as file:
	mp = pickle.load(file)

maps={'bent':'https://www.google.com/maps/place/Fort+Vredeburg+Museum/@-7.8002713,110.3641111,17z/data=!3m1!4b1!4m5!3m4!1s0x2e7a5788c0b3eecf:0xb9611ce0232a9ff8!8m2!3d-7.8002713!4d110.3662998',
'boro':'https://www.google.com/maps/place/Borobudur+Temple/@-7.6078738,110.2015626,17z/data=!3m1!4b1!4m5!3m4!1s0x2e7a8cf009a7d697:0xdd34334744dc3cb!8m2!3d-7.6078738!4d110.2037513',
'pinus':'https://www.google.com/maps/place/Hutan+Pinus+Mangunan+Dlingo/@-7.9267837,110.429808,17z/data=!3m1!4b1!4m5!3m4!1s0x2e7a536355abb129:0x9fb567811ef62e4e!8m2!3d-7.9267837!4d110.4319967',
'ts': 'https://www.google.com/maps/place/Taman+Sari/@-7.8100812,110.3571798,17z/data=!3m1!4b1!4m5!3m4!1s0x2e7a57923d58046b:0x9fbd6cc9617191f4!8m2!3d-7.8100812!4d110.3593685',
'tugu':'https://www.google.com/maps/place/Tugu/@-7.782984,110.3648463,17z/data=!3m1!4b1!4m5!3m4!1s0x2e7a591a4d553bd5:0xc0f964003add568b!8m2!3d-7.782984!4d110.367035',
'unk': 'http://google.com/unknown',
}

def knning():
	dico = extract_ciri()

	diki = []
	for n in range(len(dico)):
		r = neigh.predict(dico[n].reshape(1, -1))
		diki.append(r)

	dataset = pd.DataFrame(np.zeros([1, 220], dtype=int))
	
	buatkol = []
	for i in range(220):
		buatkol.append('p_' + str(i))
	dataset.columns = (buatkol)

	asain = []
	for a in dataset.columns:
		lel = 0
		for i in diki:
			if i == a:
				lel += 1
		asain.append(lel)
	
	dataset = dataset.append(pd.Series(asain, index=dataset.columns), ignore_index=True)
	lol = dataset.columns
	pre = pd.DataFrame(dataset.loc[1, lol], index=dataset.columns)
	pre = pre.transpose()
	hasil = mp.predict(pre)
	print(hasil)
	pro=mp.predict_proba(pre)
	prob=max(pro[0])
	return hasil, prob

def extract_ciri():
	dico = []
	nem = ('pict.jpg')
	img = cv2.imread(nem)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img, None)

	for d in des:
		dico.append(d)
	return dico
	
#print(lis)

def doNothing():
	global lis, prob
	lis, prob = knning()
	print(lis)
	if lis=='boro':
		hasil="Borobudur,"
		haha="terletak di kawasan Magelang, sekitar 40 km dari kota Yogyakarta. \nBorobudur adalah candi atau kuil Buddha terbesar di dunia, sekaligus salah satu \nmonumen Buddha terbesar di dunia. Monumen ini terdiri atas enam teras berbentuk \nbujur sangkar yang di atasnya terdapat tiga pelataran melingkar, pada dindingnya \ndihiasi dengan 2.672 panel relief dan aslinya terdapat 504 arca Buddha. (sumber: Wikipedia)"
	elif lis=='pinus':
		hasil='Hutan Pinus Mangunan'
		haha="secara administrasi terletak di Desa Sudimoro, Kelurahan Muntuk, \nKecamatan Dlingo, Kabupaten Bantul, Provinsi Daerah Istimewa Yogyakarta.\nDahulunya kawasan ini merupakan kawasan tanah kering dan berkapur yang tingkat kesuburannya rendah. \nKemudian oleh pemerintah melalui Perhutani wilayah ini dijadikan sebagai \nResort Pengelolaan Hutan (RPH) dengan program utamanya yaitu melakukan reboisasi.(sumber: Siswapedia)"
	elif lis=='ts':
		hasil='Taman Sari'
		haha="adalah situs bekas taman atau kebun istana Keraton Ngayogyakarta. \nKebun ini dibangun pada zaman Sultan Hamengku Buwono I (HB I) pada tahun 1758-1765/9. \nAwalnya, taman yang mendapat sebutan The Fragrant Garden ini memiliki \nluas lebih dari 10 hektare dengan sekitar 57 bangunan baik berupa gedung, kolam pemandian, \njembatan gantung, kanal air, maupun danau buatan beserta pulau buatan dan lorong bawah air. \nKebun yang digunakan secara efektif antara 1765-1812 ini pada mulanya membentang dari \nbarat daya kompleks Kedhaton sampai tenggara kompleks Magangan.(sumber: Wikipedia)"
	elif lis=='tugu':
		hasil='Tugu Yogyakarta'
		haha="adalah sebuah tugu atau monumen yang sering dipakai sebagai simbol \natau lambang dari kota Yogyakarta. Tugu ini dibangun oleh pemerintah Belanda setelah \ntugu sebelumnya runtuh akibat gempa yang terjadi waktu itu. \nTugu sebelumnya yang bernama Tugu Golong-Gilig dibangun oleh Hamengkubuwana I. \nTugu yang terletak di perempatan Jalan Jenderal Sudirman dan Jalan Margo Utomo ini, \nmempunyai nilai simbolis dan merupakan garis yang bersifat magis menghubungkan laut selatan, \nkraton Jogja dan gunung Merapi. Pada saat melakukan meditasi, konon Sultan Yogyakarta pada waktu itu \nmenggunakan tugu ini sebagai patokan arah menghadap puncak gunung Merapi.(sumber: Wikipedia)"
	elif lis=='unk':
		hasil='UNKNOWN'
		haha='Mohon maaf, lokasi tidak diketahui atau belum masuk pustaka sistem'
	else:
		hasil='Benteng Vredeberg'
		haha="Benteng Vredeburg Yogyakarta terkait erat dengan lahirnya Kasultanan Yogyakarta. \nPerjanjian Giyanti 13 Februari 1755 yang berrhasil menyelesaikan perseteruan \nantara Susuhunan Pakubuwono III dengan Pangeran Mangkubumi (Sultan Hamengku Buwono I kelak)\nadalah merupakan hasil politik Belanda yang selalu ingin \nikut campur urusan dalam negeri raja-raja Jawa waktu itu.(sumber: Wikipedia)"
	
	return hasil, haha, prob

def openweb():
	chromedir= 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
	link=maps.get(lis[0])
	webbrowser.get(chromedir).open(link)

def peking():
	hasil, haha, prob=doNothing()
	labe.configure(text=hasil) 
	hah.configure(text=haha)
	if prob<=0.60:
		prob=str(int(prob*100))+'%'
		heh.configure(text=prob, foreground='red')
	else:
		prob=str(int(prob*100))+'%'
		heh.configure(text=prob)
	conf.configure(text="Nilai Probabilitas: ")

def browse_file():
	global source
	source = filedialog.askopenfilename(initialdir="/", title="Select file")
	target_dir = os.getcwd()
	copy2(source, target_dir+'/'+'pict.jpg' )
	im=PhotoImage('pict.jpg')
	showImg()
	labe.configure(text=" ")	
	hah.configure(text=" ")
	heh.configure(text=" ")
	conf.configure(text=" ")

def showImg():
    load = Image.open("pict.jpg")
    x=int(load.width//300)
    y=int(load.height//x)
    image2 = load.resize((300,y), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(image2)
    global img
    img.configure(image=render)
    img.image = render
    root.geometry("1050x500")	

#***Toolbar
toolbar=Frame(root,bg='#fcba03')
medium=Frame(root)

insertButt=ttk.Button(toolbar,text='Insert Image', command=browse_file)
insertButt.pack(side=LEFT,padx=2, pady=2)
submitButt=ttk.Button(toolbar,text='Submit', command=peking)
submitButt.pack(side=LEFT,padx=2, pady=2)
findButt=ttk.Button(toolbar,text='Find Location', command=openweb)
findButt.pack(side=LEFT,padx=2, pady=2)

labe=ttk.Label(medium,font='Helvetica 18 bold', anchor="e")
labe.grid(row = 0, column = 0, sticky = W, pady = 2)
hah=ttk.Label(medium, anchor="e")
hah.grid(row = 1, column = 0, sticky = W, pady = 2)
conf=ttk.Label(medium, anchor='e')
conf.grid (row = 2, column = 0, sticky = W, pady = 2)
heh=ttk.Label(medium, font='Helvetica 12 bold', anchor="e")
heh.grid(row = 3, column = 0, sticky = W, pady = 2)

img= ttk.Label(root)
img.place(y=120, x=80, width=300)

toolbar.grid(row = 0, sticky = W, pady = 2) 
medium.grid(row = 1, column = 1, sticky = W, pady = 85, padx=200) 

#medium.pack(fill=X)
root.mainloop()
