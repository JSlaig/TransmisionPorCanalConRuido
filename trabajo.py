from asyncio.windows_events import NULL
import math
from textwrap import wrap
import numpy as np
import itertools as it
from scipy.spatial.distance import hamming
import random

#Variables globales:
alf = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ "

msg = "ESTO ES UN MENSAJE DE PRUEBA PARA EL TRABAJO DE SEGURIDAD INFORMATICA "

matrix = np.array([[1, 0, 1, 0, 1, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1, 0, 0, 1]])

#Funciones Auxiliares:
def binary2int(binary): 
    int_val, i, n = 0, 0, 0
    while(binary != 0): 
        a = binary % 10
        int_val = int_val + a * pow(2, i) 
        binary = binary//10
        i += 1
    return int_val

def calculateSyndrome(word, matrixH, mod):
    # Se obtiene el sindrome como producto matricial de la matriz de control por la palabra
    syndrome = np.dot(matrixH , getWordAsTransponseMatrix(word)).transpose()
    # Se convierte de matriz con valores en N a lista con valores en su modulo
    syndromeList = []
    for j in range(len(syndrome[0])):
        syndromeList.append(syndrome[0][j] % mod)
        
    return syndromeList

def getWordAsTransponseMatrix(word):
    matrix = np.zeros((len(word),1), dtype=object)
    for i in range(len(word)):
        matrix[i][0] = int(word[i])
    return matrix

def generateErrorsPatternBoard(wordLength, weight, mod):
    # Se añade la palabra de peso 0
    board = np.zeros((1, wordLength))
    
    # Se busca entre todos los posibles errores (De peso 1 peso maximo) los de peso <= a la capacidad correctora
    for error in list(it.product(range(mod), repeat=wordLength)):
        if getWeight(error) in range(1, weight+1):
                board = np.concatenate((board, np.array(error)[np.newaxis]),0)
                
    return board

def listSubstraction(list1, list2, mod):
    resultList = []
    for i in range(len(list1)):
        resultList.append(int((mod - (int(list1[i]) - int(list2[i]))) % mod))
    return resultList

def getNumberAsList(number):
    numberList = []
    numberStr = str(number)
    for element in numberStr:
        numberList.append(int(element))
    return numberList

def getWeight(word):
    weight = 0
    # El peso de una palabra es igual a los digitos distintos de 0 que posea
    for digit in word:
        if(digit != 0):
            weight += 1
    return weight
#Funciones:

#Encode:

#Codificacion de la fuente
def codificacionFuente(alf, msg):
    
    #Creamos lista con los simbolos del alfabeto
    alfSymbols = list(alf)

    #Creamos un diccionario vacio
    alfDictionary = dict()

    #Establecemos un contador y rellenamos el diccionario
    counter = 0
    for i in alfSymbols:        
        alfDictionary[i] = counter
        counter += 1
        
    #Hacemos una lista con los caracteres del mensaje
    msgSymbols = list(msg)
    
    #Hacemos un bucle para obtener los valores que tomara cada caracter del mensaje
    codedMsg = {}

    counter = 0
    for i in msgSymbols:
        codedMsg[counter] = alfDictionary[i]
        counter+=1

    #Calculamos el numero de bits que se necesitaran para pasar los digitos del alfabeto a binario
    nBits = math.floor(math.log(len(alfDictionary), 2) + 1)

    #Pasamos el mensaje codificado a binario con el numero de bits correspondiente
    binCodedMsg = {}

    getbinary = lambda x, n: format(x, 'b').zfill(n)

    counter = 0
    for i in codedMsg:
        binCodedMsg[counter] = getbinary(codedMsg[i], nBits)
        counter += 1

    #Concatenamos toda la lista en una cadena
    binCodedMsgString = ''.join(map(str, binCodedMsg.values()))

    return binCodedMsgString

#Codificacion lineal
def codificacionLineal(msg, matrix):
    #Para realizar la codificacion lineal debemos multiplicar matrices formadas por fragmentos del msg que tengan el mismo numero de columnas que la matriz tiene de filas y concatenarlos
    #Obtenemos el numero de filas de la matriz generadora
    nRows = len(matrix)

    #Separamos el mensaje en bloques de el numero de filas de la matriz generadora
    blocks = wrap(msg, nRows)

    #Multiplicamos cada bloque por la matriz controladora
    linearCodedMsg = ""
    for elem in blocks:
        elem = [int(x) for x in list(elem)]
        blockMatrix = np.array(elem)
        newBlock = np.matmul(blockMatrix, matrix)

        current = np.array2string(newBlock).replace(" ","").replace("[", "").replace("]", "").replace("2", "0")
        linearCodedMsg += current
        
    return linearCodedMsg

#Generacion de ruido
def generarRuido(msg, hamming):

    correctiveCap = (hamming - 1) / 2

    numError = random.randint(0, correctiveCap)

    
    errorPos = {}
    count = 0
    for i in range(numError):
        errorPos[count] = random.randint(0, len(msg))
        count += 1

    count = 0
    for i in range(numError):
        if msg[errorPos[count]] == "1":
            msg = msg[:(errorPos[count])] + "0" + msg[(errorPos[count]+1):]
        else:
            msg = msg[:(errorPos[count])] + "1" + msg[(errorPos[count]+1):]
        count += 1
    return msg

#Decode:

#Correccion de errores
def corregirErrores(code, hammingDistance, matrix, mod):

    # Se generan las variables necesarias para corregir el ruido 
    
    # G sera la composicion de I (tamaño de filas de A) concatenada con A 
    matrixG = matrix
    nRowsG = len(matrixG)
    nColsG = len(matrixG[0])

    matrixA = matrixG[:nRowsG, nRowsG:]
    
    # H sera la composicion de -A transpuesta concatenada con la identidad de tamaño columnas de A
    matrixH = np.concatenate((matrixA.transpose(), np.identity(len(matrixA[0]))), 1)
    
    # Se calcula la capacidad correctora en base a la distancia de Hamming
    t = math.floor((hammingDistance - 1) / 2)
    
    # Se separa el dato en palabras de longitud columnas de G y se convierte a un array numpy para poder trabajar con el
    words = []
    wordLength = len(matrixG[0])
    for i in range(0, len(code), wordLength):
        words.append(code[i:i+wordLength])
    words = np.array(words, dtype=object)  

    # Se muestra la informacion basica del codigo dividido y se comprueba si hay cola para no calcular su sindrome
    if(len(words[-1]) != wordLength):
        validWords =  len(words) - 1
    else:
        validWords =  len(words)
    
    # Se aplica el algoritmo del lider

    # Paso 1: Calcular el sindrome de cada palabra
    syndromes = []
    for i in range(validWords):
        # Se utiliza una funcion auxiliar para calcular los sindromes
        syndromes.append(calculateSyndrome(words[i], matrixH, mod))

    # Paso 2.1: Obtener el tablero de errores de patron y de sindromes incompleto
    errorsPatternBoard = generateErrorsPatternBoard(wordLength, t, mod)
    incompleteSyndromesBoard = []
    for i in range(0, len(errorsPatternBoard)):
        incompleteSyndromesBoard.append(calculateSyndrome(errorsPatternBoard[i], matrixH, mod))
        
    # Paso 2.2 y 3: Buscar el sindrome en el tablero y restar la palabra del sindrome con el error del sindrome del tablero
    for i in range(len(syndromes)):
        for j in range(len(incompleteSyndromesBoard)):
            if(syndromes[i] == incompleteSyndromesBoard[j]):
                words[i] = listSubstraction(getNumberAsList(words[i]), errorsPatternBoard[j], mod)

    # Se reagrupa el codigo sin ruido para devolverlo
    freeCode = ""
    for word in words:
        for digit in word:
            freeCode += str(digit)
    return freeCode 

#Decodificacion lineal
def decodificacionLineal(code, matrix):

    nRows = len(matrix)
    nColumns = len(matrix[0])
   
    #Separamos el mensaje en bloques de nBits
    codedBlocks = wrap(code, nColumns)

    linearDecoded = ""
    for block in codedBlocks:
        linearDecoded += block[:2]
    
    return linearDecoded

#Decodificacion de la fuente
def decodificacionFuente(alf, binCodedMsgString):
    #Creamos lista con los simbolos del alfabeto
    alfSymbols = list(alf)

    #Creamos un diccionario vacio
    alfDictionary = dict()

    #Establecemos un contador y rellenamos el diccionario
    counter = 0
    for i in alfSymbols:        
        alfDictionary[counter] = i
        counter += 1
    
    #Setteamos el numero de bits del codigo
    nBits = math.floor(math.log(len(alfDictionary), 2) + 1)

    #Separamos el mensaje en bloques de nBits
    binCodedMsg = wrap(binCodedMsgString, nBits)

    #Creamos una lista vacia para el mensaje codificado    
    codedMsg = {}    

    #Pasamos los bloques del mensaje a decimal
    counter = 0
    for i in binCodedMsg:
        codedMsg[counter] = binary2int(int(i))
        counter += 1

    #Terminamos de decodificar el mensaje
    msg = {}

    counter = 0
    for i in codedMsg:
        msg[counter] = alfDictionary[codedMsg[i]]
        counter += 1

    #Concatenamos el mensaje y lo devolvemos
    msgString = ''.join(map(str, msg.values()))

    return msgString

#Distancia Hamming
def distanciaHamming(matrix):

    currentDis = NULL
    dis = NULL

    for row in range(0, len(matrix) - 1):
        # matrix[row] - is your FIRST list
        for column in range(0, len(matrix[row])):
            x = matrix[row]

            y = matrix[row + 1]


            dis = hamming(x, y) * len(x)
        
        if(dis < currentDis or currentDis == NULL):
            currentDis = dis


    return dis










#Ejecucion de ejemplo
hammingDistance = distanciaHamming(matrix)

print("----------------Encode----------------")
#Codificacion de la fuente
print("Codificacion de la fuente:")
fontCodedMsg = codificacionFuente(alf, msg)
print(fontCodedMsg)
#Codificacion lineal
print("Codificacion lineal:")
linearCodedMsg = codificacionLineal(fontCodedMsg, matrix)
print(linearCodedMsg)
#Generacion de ruido
print("Generacion de ruido:")
noisyMsg = generarRuido(linearCodedMsg, hammingDistance)
print(noisyMsg)


print("\n----------------Decode----------------")
#Correccion de errores
print("Correccion de errores:")
correctedMsg = corregirErrores(noisyMsg, hammingDistance, matrix, 2)
print(correctedMsg)
#Decodificacion lineal
print("Decodificacion lineal:")
linearDecodedMsg = decodificacionLineal(correctedMsg, matrix)

print(linearDecodedMsg)
#Decodificacion de la fuente
print("Decodificacion de la fuente:")
msgDecoded = decodificacionFuente(alf, linearDecodedMsg)
print(msgDecoded)

#Fin de la ejecucion de ejemplo


