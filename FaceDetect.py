import cv2
import os
from matplotlib import pyplot as plt


def save_face(face, name_file):
    print("Salvando Frame Face: {}".format(name_file))
    cv2.imwrite(name_file + '.png', face)


def histograma_face(frame_face, name_file):
    plt.clf()
    print("Salvando Histograma da face: {}".format(name_file)+".jpg")
    plt.hist(frame_face.ravel(), 256, [0, 256])
    plt.savefig(name_file+".jpg")


def histograma_vide(frame, name_file):
    plt.cla()
    print("Salvando Histograma do frame do Video: {}".format(name_file)+".jpg")
    plt.hist(frame.ravel(), 256, [0, 256])
    plt.savefig(name_file+".jpg")


def read_video(file_video):
    plt.rcParams['figure.figsize'] = (224, 224)
    face_cascade = cv2.CascadeClassifier(
        'modelo/haarcascade_frontalface_default.xml')
    name_file_face = "Pessoas/face{}"

    if not os.path.exists("Hist"):
        os.makedirs("Hist")
    if not os.path.exists("HistFace"):
        os.makedirs("HistFace")
    if not os.path.exists("Pessoas"):
        os.makedirs("Pessoas")

    cap = cv2.VideoCapture(file_video)
    i, j = 0, 0
    print("Comecando lendo o video: {}".format(file_video))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            histograma_vide(frame, "Hist/histogramaVideo{}".format(i))
            i += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            for (x, y, w, h) in faces:
                frame = cv2.rectangle(
                    frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                save_face(roi_color, name_file_face.format(j))
                histograma_face(
                    roi_color, "HistFace/histograma_face{}".format(j))

                j += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Acabou')


read_video('faces.mp4')
