from tensorflow.keras import models
import cv2
import numpy as np
import simpleaudio as sa


class AudioPlayer:
    def __init__(self):
        self.play_obj = None

    def play(self, son):
        wave_obj = son
        self.play_obj = wave_obj.play()
    
    def is_done(self):
        if self.play_obj:
            return not self.play_obj.is_playing()
        return True

player = AudioPlayer()
emoji = cv2.imread('C:/Users/utilisateur/Documents/microsoft_ia/emotions/images/base.png')
model = models.load_model('C:/Users/utilisateur/Documents/microsoft_ia/emotions/data/emotions_model')
cap = cv2.VideoCapture(0)

dic_emo = {
    0: "colere",
    1: "dégout",
    2: "peur",
    3:"joie",
    4:"neutre",
    5:"triste",
    6:"surprise"   
}

dic_emojis = { 
    0:'C:/Users/utilisateur/Documents/microsoft_ia/emotions/images/colere.png',
    1:'C:/Users/utilisateur/Documents/microsoft_ia/emotions/images/degout.png',
    2:'C:/Users/utilisateur/Documents/microsoft_ia/emotions/images/peur.png',
    3:'C:/Users/utilisateur/Documents/microsoft_ia/emotions/images/joie.png',
    4:'C:/Users/utilisateur/Documents/microsoft_ia/emotions/images/neutre.png',
    5:'C:/Users/utilisateur/Documents/microsoft_ia/emotions/images/triste.png',
    6:'C:/Users/utilisateur/Documents/microsoft_ia/emotions/images/surprise.png'
}


dic_sons = {    
 0: sa.WaveObject.from_wave_file('C:/Users/utilisateur/Documents/microsoft_ia/emotions/sons/colere.wav'),
 1: sa.WaveObject.from_wave_file('C:/Users/utilisateur/Documents/microsoft_ia/emotions/sons/degout.wav'),
 2: sa.WaveObject.from_wave_file('C:/Users/utilisateur/Documents/microsoft_ia/emotions/sons/peur.wav'),
 3: sa.WaveObject.from_wave_file('C:/Users/utilisateur/Documents/microsoft_ia/emotions/sons/content.wav'),
 4: sa.WaveObject.from_wave_file('C:/Users/utilisateur/Documents/microsoft_ia/emotions/sons/neutre.wav'),
 5: sa.WaveObject.from_wave_file('C:/Users/utilisateur/Documents/microsoft_ia/emotions/sons/triste.wav'),
 6: sa.WaveObject.from_wave_file('C:/Users/utilisateur/Documents/microsoft_ia/emotions/sons/surprise.wav')
}


while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('C:/Users/utilisateur/Documents/microsoft_ia/emotions/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    frame = cv2.flip(frame, 1)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
            #placement emoji        
        rows,cols,channels = emoji.shape
        roi_smiley = frame[0:rows, 0:cols]
        frame[0:rows, 0:cols ] = emoji
            #son
        if player.is_done():
            player.play(dic_sons[maxindex])

        emoji = cv2.imread(dic_emojis[maxindex])
        cv2.putText(frame, dic_emo[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    
    #window = np.concatenate((emoji, frame), axis=0)
    cv2.imshow("Detecteur d'emotions", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()