from tkinter import messagebox, simpledialog, filedialog
from tkinter.filedialog import askopenfilename
import tkinter as tk
import math, time, threading, os, copy, itertools, sys

import mediapipe as mp
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from attention import attention
from keras.utils import to_categorical
from keras.layers import (MaxPooling2D, Dense, Dropout, Activation, Flatten,
                          LSTM, RepeatVector, Input, Conv2D, UpSampling2D, Convolution2D)
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

if sys.platform == "win32":
    import subprocess

# ── globals ──────────────────────────────────────────────────
labels = ['I','apple','can','get','good','have','help','how',
          'like','love','my','no','sorry','thank-you','want','yes','you','your']

sc = None; encoder_model = None; decoder_model = None
vectorizer = None; sign_label = None
X = Y = words = signs = None
X_train = X_test = y_train = y_test = None
words_X_train = words_X_test = words_y_train = words_y_test = None

if os.path.exists("model/scaler.pkl"):
    sc = pickle.load(open("model/scaler.pkl","rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawingModule = mp.solutions.drawing_utils

# ── hand helpers ─────────────────────────────────────────────
def calc_bounding_rect(image, landmarks):
    iw, ih = image.shape[1], image.shape[0]
    arr = np.empty((0,2), int)
    for _, lm in enumerate(landmarks.landmark):
        arr = np.append(arr, [[min(int(lm.x*iw),iw-1), min(int(lm.y*ih),ih-1)]], axis=0)
    x, y, w, h = cv2.boundingRect(arr)
    return [x, y, x+w, y+h]

def calc_landmark_list(image, landmarks):
    iw, ih = image.shape[1], image.shape[0]
    return [[min(int(lm.x*iw),iw-1), min(int(lm.y*ih),ih-1)]
            for _, lm in enumerate(landmarks.landmark)]

def pre_process_landmark(landmark_list):
    tmp = copy.deepcopy(landmark_list)
    bx, by = tmp[0]
    for i in range(len(tmp)):
        tmp[i][0] -= bx; tmp[i][1] -= by
    flat = list(itertools.chain.from_iterable(tmp))
    mv = max(map(abs, flat))
    return [v/mv for v in flat]

# ═══════════════════════════════════════════════════════════
#  COLOUR PALETTE  (mirrors HTML exactly)
# ═══════════════════════════════════════════════════════════
C = {
    "bg":      "#0f1117",
    "card":    "#1a1f2e",
    "card2":   "#252d3d",
    "border":  "#2a3040",
    "accent":  "#00e5ff",
    "accent2": "#00b8cc",
    "green":   "#00c853",
    "red":     "#ff4444",
    "orange":  "#ff6600",
    "text":    "#e0e0e0",
    "dim":     "#888888",
    "tabline": "#1e2533",
    "btnbg":   "#2a3040",
    "inputbg": "#0f1117",
    "header":  "#1a1f2e",
}

# ═══════════════════════════════════════════════════════════
#  WIDGET HELPERS
# ═══════════════════════════════════════════════════════════
def make_card(parent, title=None, expand=False):
    outer = tk.Frame(parent, bg=C["card"], bd=0,
                     highlightthickness=1, highlightbackground=C["border"])
    outer.pack(fill=tk.X if not expand else tk.BOTH,
               expand=expand, padx=0, pady=(0,10))
    if title:
        tk.Label(outer, text=title, bg=C["card"], fg=C["accent"],
                 font=("Consolas",10,"bold"), pady=6, padx=14).pack(anchor="w")
        tk.Frame(outer, bg=C["border"], height=1).pack(fill=tk.X)
    inner = tk.Frame(outer, bg=C["card"], padx=14, pady=10)
    inner.pack(fill=tk.BOTH, expand=expand)
    return inner

def make_btn(parent, text, cmd, style="normal", state=tk.NORMAL):
    styles = {
        "normal":  (C["btnbg"],  C["text"],   C["card2"]),
        "primary": (C["accent"], C["bg"],     C["accent2"]),
        "danger":  (C["red"],    "#ffffff",   "#cc0000"),
        "success": (C["green"],  "#ffffff",   "#009900"),
    }
    bg, fg, hover = styles.get(style, styles["normal"])
    b = tk.Button(parent, text=text, command=cmd, state=state,
                  bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
                  font=("Consolas",9,"bold"), bd=0, padx=14, pady=7,
                  cursor="hand2", relief="flat")
    b.pack(side=tk.LEFT, padx=(0,6), pady=2)
    b.bind("<Enter>", lambda e: b.config(bg=hover))
    b.bind("<Leave>", lambda e: b.config(bg=bg))
    return b

def make_text(parent, height=4, fg_color=None, font_size=11):
    f = tk.Frame(parent, bg=C["inputbg"],
                 highlightthickness=1, highlightbackground=C["border"])
    f.pack(fill=tk.X, pady=(0,4))
    t = tk.Text(f, bg=C["inputbg"], fg=fg_color or C["text"],
                font=("Consolas", font_size), height=height, bd=0, wrap=tk.WORD,
                insertbackground=C["accent"], padx=10, pady=8,
                selectbackground=C["accent"], selectforeground=C["bg"])
    sb = tk.Scrollbar(f, command=t.yview, bg=C["card"])
    t.configure(yscrollcommand=sb.set)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    t.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    return t

# ── confidence bar ───────────────────────────────────────────
class ConfBar:
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg=C["card"])
        self.frame.pack(fill=tk.X, pady=(4,2))
        self.canvas = tk.Canvas(self.frame, bg=C["border"], height=6,
                                bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.X)
        self.bar = self.canvas.create_rectangle(0, 0, 0, 6, fill=C["accent"], width=0)
        self.lbl = tk.Label(self.frame, text="Confidence: —",
                            bg=C["card"], fg=C["dim"], font=("Consolas",8))
        self.lbl.pack(anchor="w")
        self.canvas.bind("<Configure>", self._redraw)
        self._pct = 0

    def _redraw(self, e=None):
        w = self.canvas.winfo_width()
        self.canvas.coords(self.bar, 0, 0, w * self._pct / 100, 6)

    def set(self, pct):
        self._pct = max(0, min(100, pct))
        color = C["red"] if pct < 40 else C["orange"] if pct < 70 else C["accent"]
        self.canvas.itemconfig(self.bar, fill=color)
        self._redraw()
        self.lbl.config(text=f"Confidence: {pct:.0f}%")

# ═══════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════
root = tk.Tk()
root.title("SignSpeak – Real-Time Sign Language to Speech Converter")
root.geometry("1340x860")
root.configure(bg=C["bg"])
root.resizable(True, True)

# ── HEADER ───────────────────────────────────────────────────
hdr = tk.Frame(root, bg=C["header"], bd=0,
               highlightthickness=1, highlightbackground="#1a2a2e")
hdr.pack(fill=tk.X)

tk.Label(hdr, text="🤟", bg=C["header"],
         font=("Segoe UI Emoji",22), pady=12).pack(side=tk.LEFT, padx=16)
hd_sub = tk.Frame(hdr, bg=C["header"])
hd_sub.pack(side=tk.LEFT, pady=10)
tk.Label(hd_sub, text="Sign Language System",
         bg=C["header"], fg=C["accent"],
         font=("Consolas",16,"bold")).pack(anchor="w")
tk.Label(hd_sub, text="Sign Language · Real-time Recognition & Translation",
         bg=C["header"], fg=C["dim"], font=("Consolas",9)).pack(anchor="w")

clock_lbl = tk.Label(hdr, text="", bg=C["header"], fg=C["dim"], font=("Consolas",9))
clock_lbl.pack(side=tk.RIGHT, padx=20)
def _tick():
    clock_lbl.config(text=time.strftime("Time:  %H:%M:%S"))
    root.after(1000, _tick)
_tick()

# ── PIPELINE BAR ─────────────────────────────────────────────
pipe_bar = tk.Frame(root, bg=C["card2"], bd=0,
                    highlightthickness=1, highlightbackground=C["border"])
pipe_bar.pack(fill=tk.X, padx=12, pady=(8,0))

pipe_status = tk.Label(pipe_bar, text="READY", bg=C["card2"], fg=C["green"],
                       font=("Consolas",9,"bold"), padx=14)
pipe_status.pack(side=tk.RIGHT, pady=6)

_step_btns = []

def _unlock_next(idx):
    if idx + 1 < len(_step_btns):
        nb = _step_btns[idx+1]
        nb.config(state=tk.NORMAL, fg=C["text"])
        nb.bind("<Enter>", lambda e, b=nb: b.config(bg=C["card"]))
        nb.bind("<Leave>", lambda e, b=nb: b.config(bg=C["card2"]))

# Steps 0 (upload) and 4,5 (video/webcam) need main thread for dialogs/cv2.
# Steps 1,2,3 (preprocess/split/train) run in background thread.
_MAIN_THREAD_STEPS = {0, 4, 5}

def _run_step(idx, fn):
    pipe_status.config(text="⏳ Running…", fg="#FFD700")
    root.update()
    if idx in _MAIN_THREAD_STEPS:
        # run directly on main thread
        try:
            fn()
            _unlock_next(idx)
            pipe_status.config(text="✅ Done", fg=C["green"])
        except Exception as e:
            _log(f"❌  Step error: {e}")
            pipe_status.config(text="❌ Error", fg=C["red"])
    else:
        def _go():
            try:
                fn()
                root.after(0, lambda: _unlock_next(idx))
                root.after(0, lambda: pipe_status.config(text="✅ Done", fg=C["green"]))
            except Exception as e:
                root.after(0, lambda: _log(f"❌  Step error: {e}"))
                root.after(0, lambda: pipe_status.config(text="❌ Error", fg=C["red"]))
        threading.Thread(target=_go, daemon=True).start()

step_defs = []  # filled after functions defined

# ── TAB BAR ──────────────────────────────────────────────────
tab_line = tk.Frame(root, bg=C["tabline"], height=1)
tab_line.pack(fill=tk.X, padx=12, pady=(6,0))

tab_bar = tk.Frame(root, bg=C["bg"])
tab_bar.pack(fill=tk.X, padx=12)

_active_tab = [None]
_tab_btns   = {}
_tab_pages  = {}

def show_tab(name):
    for n, p in _tab_pages.items():
        p.pack_forget()
        _tab_btns[n].config(fg=C["dim"])
    _tab_pages[name].pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
    _tab_btns[name].config(fg=C["accent"])
    _active_tab[0] = name

def add_tab(name, label_text):
    b = tk.Button(tab_bar, text=label_text, bg=C["bg"], fg=C["dim"],
                  font=("Consolas",10), bd=0, padx=20, pady=10,
                  activebackground=C["bg"], activeforeground=C["accent"],
                  cursor="hand2", relief="flat",
                  command=lambda n=name: show_tab(n))
    b.pack(side=tk.LEFT)
    _tab_btns[name] = b
    p = tk.Frame(root, bg=C["bg"])
    _tab_pages[name] = p
    return p

# ── CONSOLE (bottom, always visible) ─────────────────────────
console_outer = tk.Frame(root, bg=C["bg"],
                          highlightthickness=1, highlightbackground=C["border"])
console_outer.pack(fill=tk.X, side=tk.BOTTOM, padx=12, pady=(0,8))
console_hdr_f = tk.Frame(console_outer, bg=C["card2"])
console_hdr_f.pack(fill=tk.X)
tk.Label(console_hdr_f, text="  ◉  CONSOLE OUTPUT", bg=C["card2"], fg=C["accent"],
         font=("Consolas",9,"bold"), pady=5).pack(side=tk.LEFT)

console_txt = tk.Text(console_outer, bg=C["inputbg"], fg=C["accent"],
                      font=("Consolas",9), height=5, bd=0, wrap=tk.WORD,
                      insertbackground=C["accent"], padx=10, pady=6,
                      state=tk.DISABLED)
console_txt.pack(fill=tk.X)
tk.Button(console_hdr_f, text="CLEAR ×", bg=C["card2"], fg=C["dim"],
          font=("Consolas",8), bd=0, cursor="hand2", padx=10,
          command=lambda: [console_txt.config(state=tk.NORMAL),
                           console_txt.delete("1.0",tk.END),
                           console_txt.config(state=tk.DISABLED)]
          ).pack(side=tk.RIGHT, pady=4)

def _log(msg=""):
    console_txt.config(state=tk.NORMAL)
    ts = time.strftime("%H:%M:%S")
    console_txt.insert(tk.END, f"[{ts}]  {msg}\n" if msg else "\n")
    console_txt.see(tk.END)
    console_txt.config(state=tk.DISABLED)
    root.update_idletasks()

# ═══════════════════════════════════════════════════════════
#  CORE ML FUNCTIONS  (names & logic unchanged)
# ═══════════════════════════════════════════════════════════
def uploadDataset():
    global X, Y, words, signs, sign_label, vectorizer, labels
    filename = filedialog.askdirectory(initialdir=".")
    if not filename: return
    _log(f"📂  {filename} loaded")
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
        words = np.load("model/words.npy", allow_pickle=True)
        signs = np.load("model/signs.npy")
        sign_label = np.load("model/sign_label.npy", allow_pickle=True)
    else:
        words=[]; signs=[]; sign_label=[]; old="old"
        for rdir, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                if 'Thumbs.db' not in directory[j]:
                    cur = os.path.basename(rdir)
                    if cur != old:
                        img = cv2.imread(rdir+"/"+directory[j], 0)
                        img = cv2.resize(img, (128,128))
                        sign_label.append(img); old = cur
                    name = int(os.path.basename(rdir))
                    data = labels[name].strip().lower()
                    words.append(data); signs.append(img)
        words = np.asarray(words); signs = np.asarray(signs)
        np.save("model/words", words); np.save("model/signs", signs)
    _log(f"✅  Total images: {signs.shape[0]}")
    _log(f"🏷   Classes: {labels}")
    try:
        _, count = np.unique(Y, return_counts=True)
        plt.figure(figsize=(12,3))
        plt.bar(np.arange(len(labels)), count)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.xlabel("Class Labels"); plt.ylabel("Count")
        plt.tight_layout(); plt.show()
    except Exception as e:
        _log(f"Chart skipped: {e}")

def processDataset():
    global X, Y, words, signs, sign_label, vectorizer, sc
    _log("⚙️  Preprocessing…")
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None,
                                  decode_error='replace', max_features=39)
    words_vec = vectorizer.fit_transform(words).toarray()
    XX = []
    for i in range(len(words_vec)):
        w = np.reshape(words_vec[i], (13,3)); w = cv2.resize(w,(128,128)); XX.append(w)
    words = np.asarray(XX).reshape(-1,128,128,1).astype('float32') / 255
    signs = np.reshape(signs, (signs.shape[0],128,128,1)).astype('float32') / 255
    sc = StandardScaler()
    X = sc.fit_transform(X)
    pickle.dump(sc, open("model/scaler.pkl","wb"))
    idx = np.arange(X.shape[0]); np.random.shuffle(idx)
    X = X[idx]; Y = Y[idx]; Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    _log("✅  Preprocessing complete")

def splitDataset():
    global X_train, X_test, y_train, y_test, X, Y, words, signs
    global words_X_train, words_X_test, words_y_train, words_y_test
    _log("🔀  Splitting…")
    words_X_train, words_X_test, words_y_train, words_y_test = train_test_split(
        words, signs, test_size=0.2, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    _log(f"   Train: {X_train.shape[0]}   Test: {X_test.shape[0]}")

def trainSignnet():
    global X_train, X_test, y_train, y_test
    global words_X_train, words_X_test, words_y_train, words_y_test
    global encoder_model, decoder_model, labels
    _log("🧠  Building encoder…")
    encoder_model = Sequential()
    encoder_model.add(Convolution2D(64,(1,1),
        input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]),activation='relu'))
    encoder_model.add(MaxPooling2D(pool_size=(1,1)))
    encoder_model.add(Convolution2D(32,(1,1),activation='relu'))
    encoder_model.add(MaxPooling2D(pool_size=(1,1)))
    encoder_model.add(Flatten()); encoder_model.add(RepeatVector(3))
    encoder_model.add(attention(return_sequences=True, name='attention'))
    encoder_model.add(LSTM(32)); encoder_model.add(Dense(64, activation='relu'))
    encoder_model.add(Dense(y_train.shape[1], activation='softmax'))
    encoder_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    encoder_model.load_weights("model/encoder_weights.keras")
    inp = Input(shape=(128,128,1))
    x = Conv2D(64,(3,3),activation='relu',padding='same')(inp)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(16,(3,3),activation='relu',padding='same')(x)
    enc = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(16,(3,3),activation='relu',padding='same')(enc); x = UpSampling2D((2,2))(x)
    x = Conv2D(32,(3,3),activation='relu',padding='same')(x);  x = UpSampling2D((2,2))(x)
    x = Conv2D(64,(3,3),activation='relu',padding='same')(x);  x = UpSampling2D((2,2))(x)
    dec = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)
    decoder_model = Model(inp, dec)
    decoder_model.compile(optimizer='adam', loss='binary_crossentropy')
    if not os.path.exists("model/decoder_weights.keras"):
        cb = ModelCheckpoint('model/decoder_weights.keras', verbose=1, save_best_only=True)
        hist = decoder_model.fit(words_X_train, words_y_train, batch_size=32, epochs=380,
                                  validation_data=(words_X_test, words_y_test),
                                  callbacks=[cb], verbose=1)
        pickle.dump(hist.history, open('model/decoder_history.pckl','wb'))
    else:
        decoder_model.load_weights("model/decoder_weights.keras")
    pred = encoder_model.predict(X_test)
    pred = np.argmax(pred, axis=1); yt = np.argmax(y_test, axis=1)
    ts = " ".join(labels[yt[i]] for i in range(len(pred)))
    ps = " ".join(labels[pred[i]] for i in range(len(pred)))
    score = sentence_bleu([ts.split()], ps.split()) / 2
    _log(f"🏆  BLEU Score = {score:.4f}")

def getSignImage(word, index):
    td = word.lower().strip()
    td = vectorizer.transform([td]).toarray()
    td = np.reshape(td,(13,3)); td = cv2.resize(td,(128,128))
    tmp = np.asarray([td]).reshape(1,128,128,1).astype('float32') / 255
    decoder_model.predict(tmp)
    img = sign_label[index]; return cv2.resize(img,(300,300))

def predict():
    global encoder_model, decoder_model, sc, labels
    _log("▶  Video prediction")
    filename = askopenfilename(initialdir="testVideo")
    if not filename: return
    camera = cv2.VideoCapture(filename); detected = 0
    while True:
        grabbed, frame = camera.read()
        if frame is None: break
        img = cv2.flip(frame,1); dbg = copy.deepcopy(img)
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                drawingModule.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)
                llist = calc_landmark_list(dbg, hl)
                pp = pre_process_landmark(llist)
                d = np.asarray([pp]); d = sc.transform(d)
                d = np.reshape(d,(d.shape[0],d.shape[1],1,1))
                p = encoder_model.predict(d); pt = int(np.argmax(p))
                cv2.putText(img,'Sign: '+labels[pt],(10,25),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,229,255),2)
                _log(f"   {labels[pt]}")
                sign_img = getSignImage(labels[pt], pt); detected = 1
        cv2.imshow("Sign Language Prediction", img)
        if detected: cv2.imshow("Text to Sign", sign_img); detected = 0
        if cv2.waitKey(500) & 0xFF == ord('q'): break
    camera.release(); cv2.destroyAllWindows()

def predictfromWebcam():
    global sign_label, vectorizer, sc, encoder_model, decoder_model, labels
    _log("📷  Webcam — press Q to stop")
    camera = cv2.VideoCapture(0); count = 0
    while True:
        grabbed, frame = camera.read()
        if frame is None: break
        img = cv2.flip(frame,1); dbg = copy.deepcopy(img)
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                drawingModule.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)
                llist = calc_landmark_list(dbg, hl)
                pp = pre_process_landmark(llist)
                d = np.asarray([pp]); d = sc.transform(d)
                d = np.reshape(d,(d.shape[0],d.shape[1],1,1))
                p = encoder_model.predict(d); pt = int(np.argmax(p))
                cv2.putText(img,'Sign: '+labels[pt],(10,25),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,229,255),2)
                _log(f"   {labels[pt]}")
                sign_img = getSignImage(labels[pt], pt)
                cv2.imshow("Text to Sign", sign_img); count += 1
        cv2.imshow("Sign Language Prediction", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or count > 1500: break
    camera.release(); cv2.destroyAllWindows()

ASSETS_PATH = "assets"

def play_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): _log(f"❌  Cannot open {path}"); return
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Sign Animation", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyWindow("Sign Animation")

def play_word(word):
    p = os.path.join(ASSETS_PATH, f"{word.capitalize()}.mp4")
    if os.path.exists(p):
        _log(f"▶  {word}"); play_video(p)
    else:
        _log(f"Spelling: {word}")
        for c in word:
            if c.isalpha():
                cp = os.path.join(ASSETS_PATH, f"{c.upper()}.mp4")
                if os.path.exists(cp): play_video(cp)

def text_to_sign_sentence(sentence):
    for w in sentence.split(): play_word(w)

def textToSignUI():
    s = simpledialog.askstring("Text → Sign", "Enter sentence:")
    if s: _log(f"Input: {s}"); text_to_sign_sentence(s)

def speechToSignUI():
    if not SR_AVAILABLE: _log("❌  speech_recognition not installed"); return
    r = sr.Recognizer()
    with sr.Microphone() as src:
        _log("🎙  Listening…"); audio = r.listen(src)
    try:
        s = r.recognize_google(audio)
        _log(f"🗣  You said: {s}")
        text_to_sign_sentence(s)
    except: _log("❌  Speech not recognized")

# ── speak helper ──────────────────────────────────────────────
def _speak_text(txt):
    if not txt or not txt.strip(): return
    if sys.platform == "win32":
        subprocess.Popen(['PowerShell','-Command',
            f'Add-Type -AssemblyName System.Speech; '
            f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{txt}")'])
    else:
        os.system(f'espeak "{txt}" 2>/dev/null || say "{txt}" 2>/dev/null')

# ═══════════════════════════════════════════════════════════
#  PIPELINE BAR BUTTONS  (now that fns are defined)
# ═══════════════════════════════════════════════════════════
step_defs = [
    ("01 Upload",     uploadDataset),
    ("02 Preprocess", processDataset),
    ("03 Split",      splitDataset),
    ("04 Train",      trainSignnet),
    ("05 Video",      predict),
    ("06 Webcam",     predictfromWebcam),
]

for i, (lbl_text, fn) in enumerate(step_defs):
    enabled = (i == 0)
    b = tk.Button(pipe_bar, text=lbl_text,
                  font=("Consolas",9,"bold"),
                  bg=C["card2"], fg=C["text"] if enabled else C["dim"],
                  bd=0, padx=16, pady=8,
                  activebackground=C["card"], activeforeground=C["accent"],
                  cursor="hand2" if enabled else "arrow",
                  state=tk.NORMAL if enabled else tk.DISABLED,
                  relief="flat",
                  command=lambda i=i, fn=fn: _run_step(i, fn))
    b.pack(side=tk.LEFT)
    _step_btns.append(b)
    if i < len(step_defs) - 1:
        tk.Label(pipe_bar, text=" › ", bg=C["card2"], fg=C["dim"],
                 font=("Consolas",12)).pack(side=tk.LEFT)

# ═══════════════════════════════════════════════════════════
#  PAGE 1 — ✋ Sign → Text
# ═══════════════════════════════════════════════════════════
p1 = add_tab("sign-to-text", "✋  Sign → Text")

p1_cols = tk.Frame(p1, bg=C["bg"])
p1_cols.pack(fill=tk.BOTH, expand=True)
p1_left  = tk.Frame(p1_cols, bg=C["bg"])
p1_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,8))
p1_right = tk.Frame(p1_cols, bg=C["bg"])
p1_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8,0))

# Camera card
cam_card = make_card(p1_left, "📷  Camera Feed", expand=False)

badge_row = tk.Frame(cam_card, bg=C["card"])
badge_row.pack(fill=tk.X, pady=(0,6))
mode_badge = tk.Label(badge_row, text=" LETTER ", bg=C["card2"], fg=C["accent"],
                       font=("Consolas",8,"bold"), padx=6, pady=3)
mode_badge.pack(side=tk.LEFT, padx=(0,6))
hand_badge = tk.Label(badge_row, text=" No Hand ", bg="#2d1a1a", fg=C["red"],
                       font=("Consolas",8,"bold"), padx=6, pady=3)
hand_badge.pack(side=tk.LEFT)

cam_status = tk.Label(cam_card, text="Click Start Camera to begin",
                       bg=C["card"], fg=C["dim"], font=("Consolas",8), anchor="w")
cam_status.pack(fill=tk.X, pady=(0,6))

cam_ctrl = tk.Frame(cam_card, bg=C["card"])
cam_ctrl.pack(fill=tk.X)

_cam_running = [False]; _cam_cap = [None]
_cam_mode    = ["LETTER"]

def _toggle_camera():
    if _cam_running[0]:
        _cam_running[0] = False
        cam_btn.config(text="▶  Start Camera")
        mode_btn.config(state=tk.DISABLED)
        cam_status.config(text="Camera stopped")
        if _cam_cap[0]: _cam_cap[0].release()
        cv2.destroyAllWindows()
        return
    threading.Thread(target=_run_camera, daemon=True).start()

def _run_camera():
    _cam_running[0] = True
    cam_btn.config(text="⏹  Stop")
    mode_btn.config(state=tk.NORMAL)
    cam_status.config(text="Camera running — show your hand!")
    cap = cv2.VideoCapture(0); _cam_cap[0] = cap
    last_ch = [""]
    while _cam_running[0]:
        ret, frame = cap.read()
        if not ret: break
        img = cv2.flip(frame, 1); dbg = copy.deepcopy(img)
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            hand_badge.config(text=" Hand ✓ ", bg="#1a2d1a", fg=C["green"])
            for hl in res.multi_hand_landmarks:
                drawingModule.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)
                if encoder_model and sc:
                    llist = calc_landmark_list(dbg, hl)
                    pp = pre_process_landmark(llist)
                    d = np.asarray([pp]); d = sc.transform(d)
                    d = np.reshape(d,(1,d.shape[1],1,1))
                    p = encoder_model.predict(d, verbose=0)
                    pt = int(np.argmax(p)); conf = float(np.max(p)) * 100
                    cv2.putText(img, 'Sign: '+labels[pt], (10,25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,229,255), 2)
                    if labels[pt] != last_ch[0]:
                        last_ch[0] = labels[pt]
                        _p1_append(labels[pt])
                        p1_conf.set(conf)
                        cam_status.config(text=f"Detected: {labels[pt]}  ({conf:.0f}%)")
        else:
            hand_badge.config(text=" No Hand ", bg="#2d1a1a", fg=C["red"])
        cv2.imshow("SignSpeak — Camera", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    _cam_running[0] = False
    cap.release(); cv2.destroyAllWindows()
    cam_btn.config(text="▶  Start Camera")
    hand_badge.config(text=" No Hand ", bg="#2d1a1a", fg=C["red"])
    cam_status.config(text="Camera stopped")

cam_btn  = make_btn(cam_ctrl, "▶  Start Camera", _toggle_camera, "primary")
mode_btn = make_btn(cam_ctrl, "🔄  Word Mode",
                   lambda: [
                       _cam_mode.__setitem__(0, "WORD" if _cam_mode[0]=="LETTER" else "LETTER"),
                       mode_badge.config(text=f" {_cam_mode[0]} "),
                       mode_btn.config(text=f"🔄  {'Word' if _cam_mode[0]=='LETTER' else 'Letter'} Mode")
                   ], state=tk.DISABLED)

# Recognized text card
rec_card = make_card(p1_right, "📝  Recognized Text")

p1_out = tk.Text(rec_card, bg=C["inputbg"], fg=C["accent"],
                 font=("Consolas",18,"bold"), height=3, bd=0, wrap=tk.WORD,
                 insertbackground=C["accent"], padx=12, pady=10,
                 highlightthickness=1, highlightbackground=C["border"],
                 state=tk.DISABLED)
p1_out.pack(fill=tk.X)

p1_conf = ConfBar(rec_card)

def _p1_append(ch):
    p1_out.config(state=tk.NORMAL)
    cur = p1_out.get("1.0", tk.END).strip()
    if cur == "Perform signs to see output here…": cur = ""
    p1_out.delete("1.0", tk.END)
    p1_out.insert(tk.END, cur + ch)
    p1_out.config(state=tk.DISABLED)

p1_out.config(state=tk.NORMAL)
p1_out.insert(tk.END, "Perform signs to see output here…")
p1_out.config(state=tk.DISABLED)

p1_btns = tk.Frame(rec_card, bg=C["card"])
p1_btns.pack(fill=tk.X, pady=(8,0))
make_btn(p1_btns, "⎵ Space",  lambda: _p1_append(" "))
make_btn(p1_btns, "⌫ Delete",
         lambda: [p1_out.config(state=tk.NORMAL),
                  p1_out.delete("end-2c","end-1c"),
                  p1_out.config(state=tk.DISABLED)])
make_btn(p1_btns, "🔊 Speak",
         lambda: _speak_text(p1_out.get("1.0",tk.END).strip()), "success")
make_btn(p1_btns, "📋 Copy",
         lambda: [root.clipboard_clear(),
                  root.clipboard_append(p1_out.get("1.0",tk.END).strip())])
make_btn(p1_btns, "🗑 Clear",
         lambda: [p1_out.config(state=tk.NORMAL),
                  p1_out.delete("1.0",tk.END),
                  p1_out.insert(tk.END,"Perform signs to see output here…"),
                  p1_out.config(state=tk.DISABLED),
                  p1_conf.set(0)], "danger")

# How to use card
how_card = make_card(p1_right, "ℹ️  How to use")
how_txt = ("LETTER mode: Hold each letter sign still for 1-2 seconds.\n"
           "WORD mode: Perform word gesture while progress bar fills.\n"
           "Words trained: Sorry, Thank You, Please, Help, Want, Yes, No")
tk.Label(how_card, text=how_txt, bg=C["card"], fg=C["dim"],
         font=("Consolas",8), justify=tk.LEFT, wraplength=500).pack(anchor="w")

# ═══════════════════════════════════════════════════════════
#  PAGE 2 — 🔊 Text → Speech
# ═══════════════════════════════════════════════════════════
p2 = add_tab("text-to-speech", "🔊  Text → Speech")

tts_card = make_card(p2, "🔊  Text → Speech")
tts_input = make_text(tts_card, height=5)

tts_ctrl = tk.Frame(tts_card, bg=C["card"])
tts_ctrl.pack(fill=tk.X, pady=(6,0))
make_btn(tts_ctrl, "🔊  Speak",
         lambda: _speak_text(tts_input.get("1.0",tk.END).strip()), "primary")
make_btn(tts_ctrl, "⏹  Stop",
         lambda: os.system("pkill espeak 2>/dev/null || taskkill /f /im powershell.exe 2>nul"),
         "danger")

phrases_card = make_card(p2, "⚡  Quick Phrases")
phrases = ["Hello, how are you?", "Thank you very much",
           "Please help me", "I am sorry", "Good morning", "My name is ..."]
for row_phrases in [phrases[:3], phrases[3:]]:
    row = tk.Frame(phrases_card, bg=C["card"])
    row.pack(fill=tk.X, pady=2)
    for ph in row_phrases:
        def _set(p=ph):
            tts_input.delete("1.0",tk.END); tts_input.insert(tk.END, p)
        make_btn(row, ph, _set)

# ═══════════════════════════════════════════════════════════
#  PAGE 3 — 🎤 Speech → Sign
# ═══════════════════════════════════════════════════════════
p3 = add_tab("speech-to-sign", "🎤  Speech → Sign")

p3_cols = tk.Frame(p3, bg=C["bg"])
p3_cols.pack(fill=tk.BOTH, expand=True)
p3_left  = tk.Frame(p3_cols, bg=C["bg"])
p3_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,8))
p3_right = tk.Frame(p3_cols, bg=C["bg"])
p3_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8,0))

# Mic card
mic_card = make_card(p3_left, "🎤  Record Speech")

mic_canvas = tk.Canvas(mic_card, width=80, height=80,
                        bg=C["card"], highlightthickness=0)
mic_canvas.pack(pady=6)
mic_ring = mic_canvas.create_oval(4,4,76,76, outline=C["accent"], width=3, dash=(6,3))
mic_icon = mic_canvas.create_text(40,40, text="🎤",
                                   font=("Segoe UI Emoji",22), fill=C["accent"])
_mic_anim_id = [None]
_mic_angle    = [0]

def _animate_mic():
    if not _mic_anim_id[0]: return
    _mic_angle[0] = (_mic_angle[0] + 8) % 360
    r = 38 + 3 * math.sin(math.radians(_mic_angle[0]))
    cx, cy = 40, 40
    mic_canvas.coords(mic_ring, cx-r, cy-r, cx+r, cy+r)
    _mic_anim_id[0] = root.after(30, _animate_mic)

_mic_recording = [False]

def _toggle_mic():
    if _mic_recording[0]: return
    _mic_recording[0] = True
    mic_canvas.itemconfig(mic_ring, outline=C["red"])
    mic_canvas.itemconfig(mic_icon, fill=C["red"])
    _mic_anim_id[0] = root.after(30, _animate_mic)
    mic_status_lbl.config(text="🔴  Listening…", fg=C["red"])
    threading.Thread(target=_do_speech_record, daemon=True).start()

def _do_speech_record():
    speechToSignUI()
    _mic_recording[0] = False
    if _mic_anim_id[0]: root.after_cancel(_mic_anim_id[0]); _mic_anim_id[0] = None
    mic_canvas.itemconfig(mic_ring, outline=C["accent"])
    mic_canvas.itemconfig(mic_icon, fill=C["accent"])
    mic_canvas.coords(mic_ring, 4,4,76,76)
    mic_status_lbl.config(text="✅  Done — click to record again", fg=C["green"])
    convert_btn.config(state=tk.NORMAL)

mic_canvas.bind("<Button-1>", lambda e: _toggle_mic())
mic_canvas.config(cursor="hand2")

mic_status_lbl = tk.Label(mic_card, text="Click mic to start",
                            bg=C["card"], fg=C["dim"], font=("Consolas",9))
mic_status_lbl.pack()

transcript_box = make_text(mic_card, height=3, fg_color=C["text"])
transcript_box.insert(tk.END, "Transcript appears here…")

p3_ctrl = tk.Frame(mic_card, bg=C["card"])
p3_ctrl.pack(fill=tk.X, pady=(6,0))
convert_btn = make_btn(p3_ctrl, "📤  Convert to Sign",
                        lambda: text_to_sign_sentence(
                            transcript_box.get("1.0",tk.END).strip()),
                        "primary", state=tk.DISABLED)

# Or type
type_card = make_card(p3_left, "⌨️  Or Type")
sts_input = make_text(type_card, height=3, fg_color=C["text"])
p3_type_ctrl = tk.Frame(type_card, bg=C["card"])
p3_type_ctrl.pack(fill=tk.X, pady=(6,0))
make_btn(p3_type_ctrl, "📤  Convert",
         lambda: text_to_sign_sentence(sts_input.get("1.0",tk.END).strip()),
         "primary")
make_btn(p3_type_ctrl, "🔊  Speak",
         lambda: _speak_text(sts_input.get("1.0",tk.END).strip()),
         "success")

# Playback info card (right)
play_card = make_card(p3_right, "📽  Sign Playback", expand=True)
play_info = tk.Label(play_card, text="Sign videos will play in a separate OpenCV window.",
                     bg=C["card"], fg=C["dim"], font=("Consolas",9),
                     wraplength=340, justify=tk.LEFT)
play_info.pack(anchor="w", pady=4)
tk.Label(play_card,
         text=("Make sure your  assets/  folder contains\n"
               "word .mp4 files and/or letter A-Z .mp4 files."),
         bg=C["card"], fg=C["dim"], font=("Consolas",8),
         justify=tk.LEFT).pack(anchor="w", pady=4)
make_btn(play_card, "▶  Play from Transcript",
         lambda: text_to_sign_sentence(transcript_box.get("1.0",tk.END).strip()),
         "primary")
make_btn(play_card, "🔊  Speak Transcript",
         lambda: _speak_text(transcript_box.get("1.0",tk.END).strip()),
         "success")

# ═══════════════════════════════════════════════════════════
#  PAGE 4 — 📝 Text → Sign
# ═══════════════════════════════════════════════════════════
p4 = add_tab("text-to-sign", "📝  Text → Sign")

t2s_top = make_card(p4, "📝  Text → Sign")
t2s_input = make_text(t2s_top, height=4, fg_color=C["text"])

t2s_ctrl = tk.Frame(t2s_top, bg=C["card"])
t2s_ctrl.pack(fill=tk.X, pady=(6,0))
make_btn(t2s_ctrl, "🔄  Convert & Play",
         lambda: text_to_sign_sentence(t2s_input.get("1.0",tk.END).strip()),
         "primary")
make_btn(t2s_ctrl, "🔊  Speak",
         lambda: _speak_text(t2s_input.get("1.0",tk.END).strip()),
         "success")
make_btn(t2s_ctrl, "🗑 Clear",
         lambda: [t2s_input.delete("1.0",tk.END)], "danger")

p4_cols = tk.Frame(p4, bg=C["bg"])
p4_cols.pack(fill=tk.BOTH, expand=True, pady=(4,0))
p4_left  = tk.Frame(p4_cols, bg=C["bg"])
p4_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,8))
p4_right = tk.Frame(p4_cols, bg=C["bg"])
p4_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8,0))

info_card = make_card(p4_left, "ℹ️  Playback Info")
tk.Label(info_card,
         text=("Sign animations play word-by-word in an OpenCV window.\n\n"
               "If a whole-word video is not found, the system\n"
               "will spell it out letter by letter.\n\n"
               "Assets folder: ./assets/  (e.g. Hello.mp4, A.mp4)"),
         bg=C["card"], fg=C["dim"], font=("Consolas",9),
         justify=tk.LEFT).pack(anchor="w")

seq_card = make_card(p4_right, "📋  Quick Phrases", expand=True)
quick_phrases = [
    "I love you", "Thank you", "Sorry", "Please help me",
    "Good morning", "How are you", "Yes", "No",
]
seq_inner = tk.Frame(seq_card, bg=C["card"])
seq_inner.pack(fill=tk.BOTH, expand=True)
for ph in quick_phrases:
    row = tk.Frame(seq_inner, bg=C["card"])
    row.pack(fill=tk.X, pady=2)
    tk.Label(row, text=ph, bg=C["card"], fg=C["text"],
             font=("Consolas",10), width=22, anchor="w").pack(side=tk.LEFT)
    def _play(p=ph):
        t2s_input.delete("1.0",tk.END); t2s_input.insert(tk.END,p)
        text_to_sign_sentence(p)
    make_btn(row, "▶ Play",  _play, "primary")
    make_btn(row, "🔊 Speak", lambda p=ph: _speak_text(p), "success")

# ═══════════════════════════════════════════════════════════
#  LAUNCH
# ═══════════════════════════════════════════════════════════
show_tab("sign-to-text")

root.after(300, lambda: [
    _log("╔══════════════════════════════════════════╗"),
    _log("║  SignSpeak – Real-Time Sign Language to Speech Converter ║"),
    _log("╚══════════════════════════════════════════╝"),
    _log("🔷  Use the pipeline bar (top) to run steps 01 → 06"),
    _log("🔷  Or use the tabs to jump directly to any feature"),
])

root.mainloop()