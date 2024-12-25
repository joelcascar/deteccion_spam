# Detector de SPAM en correos electronicos

# imports
from html.parser import HTMLParser
import email
from nltk import download 
from nltk import PorterStemmer
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# descargamos los data de nltk
download('stopwords')
download('punkt_tab')

# Primero Creamos una funcion que lea los correos y si es span o legitimo
def leer_correos(indice, numero_correos):
    with open(indice, "r") as f:
        labels = f.read().splitlines()
    # Leemos los correos de disco
    x = []
    y = []
    for i in labels[:numero_correos]:
        label, email_path = i.split(" ../")
        y.append(label)
        with open(email_path, "r", errors="ignore") as f:
            x.append(f.read())
    return x, y


# Obtenemos una tupla con una lista de emails y una lista de tipo email
x,y = leer_correos("full/index", 20)
# Obtenemos el primer correo
x[0]
# obtenemos el primer tipo de email
y[0]


# Creamos la clase que eliminara el codigo HTML de los emails 
class HTMLStripper(HTMLParser):
    def __init__(self, *, convert_charrefs = True):
        super().__init__(convert_charrefs=convert_charrefs)
        # Atributo que almacena los datos
        self.data = []
        
    def handle_data(self, data):
        self.data.append(data) 

"""
# instanciamos la clase HTMLStripper
html_strpper = HTMLStripper()
# pasamos el email para eliminar el HTML
html_strpper.feed(x[0])
# Convertimos la lista en string
email = "".join(html_strpper.data)
"""
# 2 Funcion que me retorna el string de un email sin html
def strip_tags(text):
    # instanciamos la clase HTMLStripper
    html_stripper = HTMLStripper()
    # pasamos el email para eliminar el HTML
    html_stripper.feed(text)
    # Convertimos la lista en string
    return "".join(html_stripper.data)

"""
# Llamada a la funcion strip_tags
correo1 = strip_tags(x[0])
"""

# Preprocesamos el email para sacar solo el cuerpo del email y no lo demas
"""
# obtenemos el mensaje desde un string 
pcorreo = email.message_from_string(correo1)
# Obtenemos el cuerpo del email en una lista
cuerpo_email = pcorreo.get_payload()
# Obtenemos el cuerpo del email en string
print(cuerpo_email[0].get_payload())
"""

# Funcion para obtener uno o varios cuerpos de un correo 
def get_body(correo):
    # definimos una funcion interna que procese el cuerpo
    def parse_body(payload):
        body = []
        if type(payload) is str:
            return [payload]
        elif type(payload) is list:
            for p in payload:
                body += parse_body(p.get_payload())
        return body
    pcorreo = email.message_from_string(correo)
    return parse_body(pcorreo.get_payload())

"""
cuerpo = get_body(correo1)
cuerpo_str = strip_tags("".join(cuerpo))
print(cuerpo_str)
"""

# Usos para optimizar el email
"""
# instanciamos la clase PorterStemmer
stemmer = PorterStemmer()
# quitamos los subfijos a una palabra en ingles
raiz = stemmer.stem("doing")
print(raiz)

# conseguimos la lista de las palabras que no significan nada por si mismas en un idioma
stop_words = stopwords.words("english")
print(stop_words)

# me devuelve un string con los signos de puntuacion
signos_puntuacion = string.punctuation
print(signos_puntuacion)

# me devuelve una lista de cada palabra, caracter especial o signos de puntuacion
separacion_string = word_tokenize("Hola como estas; Me llamo joel@ Castellanos...@")
"""

# Juntamos todas las funciones en una clase 
class EmailParser:
    def parse(self, correo):
        # Obtenemos el cuerpo del correo electronico
        pcorreo = " ".join(self.get_body(correo))
        # Eliminamos los tags HTML
        pcorreo = self.strip_tags(pcorreo)
        # Eliminamos los urls
        pcorreo = self.remove_urls(pcorreo)
        # Transformamos el texto en tokens
        pcorreo = word_tokenize(pcorreo)
        # Eliminamos stopwords
        # Eliminamos puntuacion
        # Hacemos stemming
        pcorreo = self.clean_text(pcorreo)
        return " ".join(pcorreo) 
        
    def get_body(self, correo):
        pcorreo = email.message_from_string(correo)
        return self._parse_body(pcorreo.get_payload())
    
    def _parse_body(self, payload):
        body = []
        if type(payload) is str:
            return [payload]
        elif type(payload) is list:
            for p in payload:
                body += self._parse_body(p.get_payload())
        return body
    
    def strip_tags(self,correo):
        # instanciamos la clase HTMLStripper
        html_stripper = HTMLStripper()
        # pasamos el email para eliminar el HTML
        html_stripper.feed(correo)
        # Convertimos la lista en string
        return "".join(html_stripper.data)
    
    def remove_urls(self,correo):
        return re.sub(r"http\S+", "", correo)
    
    def clean_text(self, correo):
        pcorreo = []
        st = PorterStemmer()
        punct = list(string.punctuation) + ["\n", "\t"] 
        for word in correo:
            if word not in stopwords.words("english") and word not in punct:
                # Aplicamos stemming
                pcorreo.append(st.stem(word))
        return pcorreo
    
"""   
parser = EmailParser()
correo = parser.parse(x[2])
print(correo)    
"""
# Me crea el subconjunto de datos con los correos procesados
def crear_datasaet(indice, num):
    # creamos la instancia de la clase EmailParser
    email_parser = EmailParser()
    # Llamamos a la funcion leer_correos() que me devuelve las listas de los tags y los correos
    x, y = leer_correos(indice, num)
    # creamos una lista que almacenara cada correo con su tranformacion
    x_proc = []
    for i, mail in zip(range(len(x)), x):
        print("\rParsing email: {0}".format(i+1), end="")
        x_proc.append(email_parser.parse(mail))
    return x_proc, y

# Convertir cada correo eque es texto en numeros
vectorizer = CountVectorizer()
# metodo fit_transform aprende el vocabulario y los tranforma
X_vect = vectorizer.fit_transform(["un ejemplo", "otro ejemplo"])
# Convertimos el vector en un arreglo
X_vect.toarray()

# Entrenamiento del algoritmo LogisticRegression
# Primero obtenemos el conjunto de datos
X, y = crear_datasaet("full/index", 200)
# segundo dividimos el conjunto de datos en subconjunto de entrenamiento y subconjunto de pruebas
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
"""
x_train, y_train = X[:120], y[:120]
x_test, y_test = X[120:],y[120:]
"""
# Tercero vectorizamos el subconjunto de entrenamiento
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_train = x_train.toarray()
# Cuarto Entrenamos el algoritmo LogisticRegression
clf = LogisticRegression()
clf.fit(x_train, y_train)
# Quinto vectorizamos el conjunto de pruebas
x_test = vectorizer.transform(x_test)
x_test.toarray()
# Sexto Predecimos con el subconjunto de pruebas
y_pred = clf.predict(x_test)
# Septimo vemos la probabilidad de exito que tuvo
print("\nexito: {}".format(accuracy_score(y_test, y_pred)))
