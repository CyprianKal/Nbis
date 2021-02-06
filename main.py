#importyVVVVV

import nltk #zestaw bibliotek i programów do symbolicznego i statystycznego przetwarzania języka naturalnego.
from nltk.stem.lancaster import LancasterStemmer #wykonuje on steaming na słowach.  jest to proces usunięcia ze słowa końcówki fleksyjnej pozostawiając tylko temat wyrazu. Proces stemmingu może być przeprowadzany w celu zmierzenia popularności danego słowa.
stemmer = LancasterStemmer() #tu poporostu inicjujemy naszego stemmera pod zmienną stemmer
import numpy #Moduł Numpy jest podstawowym zestawem narzędzi dla języka Python umożliwiającym zaawansowane obliczenia matematyczne
import tflearn #a deep learning library featuring a higher-level API for TensorFlow
import tensorflow #to biblioteka programów do obliczeń numerycznych, w niej wykonujemy właśnie obliczenia dla sieci neuronowych, z pomocą tflearn
import random #losowość
import json #pozwala pracować z plikami json
import pickle
from os import system, name 

nltk.download('punkt') #naprawia błąd "LookupError: Resource punkt not found. Please use the NLTK Downloader to obtain the resource"


#otwiera plik z naszymi wypowiedziami i zapisuje je w "data" jako batch plików 
#(batch czyli taka tabela w której są tabele, w tych tabelach mogą być jeszcze inne tabele. Wygoogluj "shape machine learnig" to się wyjaśni )
with open('intents.json', encoding='utf-8') as file:
    data = json.load(file, )


try:
    with open("data.pickle", "rb") as f:
        words, labels, training,output = pickle.load(f)
except:
    #tabeleVVVV

    words = [] #lista wszyskich unikalnych słów w naszych patternach
    labels = [] #miejsce przechowywania wszystkich tagów, nie liczy się tutaj powiązanie tylko unikalność. Mamy w ten sposób dostęp do każdego tagu. Przyda się to później podczas tworzenia "worka wyrazów"
    docs_x = [] #tutaj przechowywane są patterny powiązane z tagami
    docs_y = [] #tutaj przechowywane są tagi powiązane z patternami



    #ta pętla przechodzi przez cały plik JSON z danymi, i selekcojnuje nasze patterny,
    #potem tokenanize je zamienia w listy słów, czyli np. jak się masz, to od teraz ['Jak', 'się', 'masz'] i usuwa znaki specjalne typu "!"
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            
            wrds = nltk.word_tokenize(pattern) #nltk.word_tokenize właśnie tu są upraszczane nasze zdania, i dzielone na listy
            words.extend(wrds)#w tym momencie te słowa dodawane są do tabeli words
            docs_x.append(wrds)#to wkleja do tabeli docsX słowo, które będzie na tej samej pozycji w tej tabeli co tag w tabeli docsY. Czyniąc je tym samym powiązanymi
            docs_y.append(intent["tag"])#to wkleja do tabeli docsY tag, któru będzie na tej samej pozycji w tej tabeli co słowo w tabeli docsX. Czyniąc je tym samym powiązanymi

            #Jak powiązanymi? To proste. Wyobraź sobie że mamy słowa "cześć", "witaj". I te słowa zapisujemy w tabeli X w pozycji X[0]. Mamy też tag "powitania", jego zapisujemy w 
            #tabeli Y w pozycji Y[0]. Teraz zawsze możemy się odwołać do Y[0] i X[0] i wiemy że to będą powiązane tagi i słowa.


        #ten if zapisuje nam wszystie tagi niezważając na powiązania, upewnia się że są one unikatowe
        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    #Tutaj wykonujemy steaming na słowach, czyli jeszcze raz przypominam - upraszczamy je maksymalnie, usuwając końcówki fleksyjne. 
    #Tak że już często nawet nie przypomiają słów którymi były w zapisie, ale wciąż da się je rozpoznać

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] #Ta linia właśnie przeprowadza steaming, robi to dla wszystkich słów, chyba że to "słowo" to znak zapytania, jego zostawia
    words = sorted(list(set(words))) #ta linia sortuje nasze słowa, upewnia się że są wciąż listą, i dzięki "set" usuwa niepotrzebne duplikaty
    labels = sorted(labels) #ta linia sortuje nasze tagi w liście labels




    #kod poniżej aż do lini 88 jest odpowiedzialny za "przetłumaczenie" naszych słów na słowa zrozumiałe dla komputera. Czyli na 0 i 1.
    #ten kod reprezentuje każde zdanie jako listę tej samej długości co ilość naszych słów w słowniku LABELS. Każda pozycja w tej utworzonej liście będzie reprezentować 
    #jedno słowo z naszego słownika LABELS. Jeśli pozycja w liście będzie równa 1, to znaczy że słowo z naszego słownika istniało w tym wyrazie, jesli 0 to znaczy że nie.
    #Nazywamy to rozwiązanie "workiem wyrazów", bo w tym momencie mamy taki jakby worek. Straciliśmy kolejnosć słów i ich logike, za to wiemy że te słowa są.

    training = [] #lista na informacje treningowe
    output = [] #lista na informacje po treningowe

    out_empty = [0 for _ in range(len(labels))] #pusty wynik, używamy go przy output w linijce 81, jeśli słowo z naszych labels nie zostało wykryte w naszym wyrazie.

    for x, doc in enumerate(docs_x): #enumerate przypisuje każdemu słowu (w tym przypadku x, doc) numerek. Zaczyna liczyć od 0 i idzie w górę
        bag = [] #Lista na nasze zera i jedynki, które reprezentują nasze słowa

        wrds = [stemmer.stem(w.lower()) for w in doc]

        #to iteruje przez tabele words. Potem sprawdza czy nasze słowo jest w tabeli wrds. Jeśli jest to zapisuje je jako jedynke w bag[]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:] #jeśli któreś z labels nie znajduje się w naszym wyrazie, wtedy zostaje tutaj na tej pozycji w tabeli wpsisane 0
        output_row[labels.index(docs_y[x])] = 1 #przechodzi przez tabele labels, i jeśli widzi że informacja z outputu jest równa jakiejś pozycji z tamtej tabeli, to wstawia w tej tabeli output jedynkę, jeśli nie to zostaje 0 z powyższej linii

        training.append(bag) #wsadza do naszej tabeli treningowej te liste słów
        output.append(output_row) #wsadza do outputów nasz output

        with open("data.pickle", "wb") as f:
            pickle.dump((words,labels,training,output), f)

training = numpy.array(training) #konweruje te tabele na tabele numpy
output = numpy.array(output)#konweruje te tabele na tabele numpy


#####################
# V BUDOWA MODELU V #
#####################

 
tensorflow.compat.v1.reset_default_graph() #ta lina koda upewnia się że nie ma ustawionych żadnych przestażałych ustawien dla naszego modelu

net = tflearn.input_data(shape=[None, len(training[0])]) #ustanawia shape dla modelu. To będzie jedna długa linia. To nasza warstwa INPUT
net = tflearn.fully_connected(net, 16) #ustanawia fully connected, czyli w pełni połączoną z poprzednią warstwą, 8 neuronową warstwę ukrytą HIDDEN LAYER
net = tflearn.fully_connected(net, 16) #ustanawia DRUGĄ fully connected, czyli w pełni połączoną z poprzednią warstwą, 8 neuronową warstwę ukrytą HIDDEN LAYER
net = tflearn.fully_connected(net, 16) #ustanawia DRUGĄ fully connected, czyli w pełni połączoną z poprzednią warstwą, 8 neuronową warstwę ukrytą HIDDEN LAYER
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #warstwa ostatnia, odpowiada za OUTPUT. Jako funckjii aktywacyjnej użyliśmy softmax
net = tflearn.regression(net)

model = tflearn.DNN(net) #DNN to typ sieci neurnowej, bierze net(naszą sieć którą przed chwilą pisaliśmy) i jej używa

try:
    model.load(model.tflearn)
except:
    model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True) #rozpoczyna "trenowanie" naszego modelu
    model.save("model.tflearn")#zapisuje nasz model

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s) 
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


def chat():
    #system('cls')
    print("Start talking! (type quit to stop)")
    while True:
        inp = input("Ty: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)]) 
        results_index = numpy.argmax(results)
        tag = labels[results_index]



    
        y = numpy.max(results)
        if y < 0.7:
            print("nie rozumiem :(")
        else:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            
            print(random.choice(responses)) #tutaj jest losowo generowana odpowiedź

       
        

chat()