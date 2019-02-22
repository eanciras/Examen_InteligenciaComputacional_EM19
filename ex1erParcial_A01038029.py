""" 
Eduardo Ancira A01038029
Inteligencia computacional (Ene 19 Gpo 1)
Primer Examen Parcial
Profesor: Alejandro Rosales Pérez

	“Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
	          me comprometo a que mi actuacion en este examen este regida 
	                            por la honestidad academica.”
"""
from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import random
import queue 
from Ann import Ann

#Funciones para operador de cruza o Crossover

"""Funcion que se encarga del crossover One point
Parametros redes padre y madre
retorna red hijo con los pesos y bias de la capa oculta del padre y 
los pesos y bias de la neurona final de la madre """
def crossOne_point(padre, madre):
	""" 
	“Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
	me comprometo a que mi actuacion en este examen este regida por la honestidad
	academica.”
	"""
	hijo = Ann(padre.size) #iniciamos el hijo como una nueva red dado el tamaño del padre
	#igualamos los pesos y bias de la capa oculta del padre
	hijo.pesosOculta = padre.pesosOculta
	hijo.biasOculta = padre.biasOculta

	#igualamos los pesos y bias de la neurona final del la madre
	hijo.pesosFinal = madre.pesosFinal
	hijo.biasFinal = madre.biasFinal
	return hijo

"""Funcion que se encarga del crossover aritmetico
Parametros redes  padre y madre
retorna red hijo con los pesos, bias de la capa oculta, 
los pesos y bias de la neurona final de ambos padres con una
proporcion de sus atributos dados por alpha"""
def crossAritmetico(padre, madre):
	""" 
	“Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
	me comprometo a que mi actuacion en este examen este regida por la honestidad
	academica.”
	"""
	alpha = .4 #proporcion de cruza de cada padre
	hijo = Ann(padre.size)#iniciamos el hijo como una nueva red dado el tamaño del padre

	#asignamos cada atributo de la red hijo a la suma del porcentaje 
	#de cada atributo que tiene el padre y la madre
	hijo.pesosOculta = padre.pesosOculta*alpha + madre.pesosOculta*(1-alpha)
	hijo.biasOculta = padre.biasOculta*alpha + madre.biasOculta*(1-alpha)
	hijo.pesosFinal = padre.pesosFinal*alpha + madre.pesosFinal*(1-alpha)
	hijo.biasFinal = padre.biasFinal*alpha + madre.biasFinal*(1-alpha)

	
	return hijo

#Funciones para operador de mutacion

def mutacion_Uniforme(original):
	""" 
	“Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
	me comprometo a que mi actuacion en este examen este regida por la honestidad
	academica.”
	"""
	limiteInf = -.5
	original.pesosOculta = (original.pesosOculta - limiteInf)*random.random() + limiteInf
	original.biasOculta = (original.biasOculta - limiteInf)*random.random() + limiteInf
	original.pesosFinal = (original.pesosFinal - limiteInf)*random.random() + limiteInf
	original.biasFinal = (original.biasFinal - limiteInf)*random.random() + limiteInf
	

def mutacion_limite(padre, madre):
	""" 
	“Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
	me comprometo a que mi actuacion en este examen este regida por la honestidad
	academica.”
	"""
	limiteInf = -.5
	s= random.random()
	original.pesosOculta = original.pesosOculta + (1-s)*limiteInf
	original.biasOculta = original.biasOculta + (1-s)*limiteInf
	original.pesosFinal = original.pesosFinal + (1-s)*limiteInf
	original.biasFinal = original.biasFinal + (1-s)*limiteInf

#Funciones de activacion

def lineal(valor):
	""" 
	“Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
	me comprometo a que mi actuacion en este examen este regida por la honestidad
	academica.”
	"""
	return .9*valor


def sigmoid(valor):
	""" 
	“Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
	me comprometo a que mi actuacion en este examen este regida por la honestidad
	academica.”
	"""
	return 1/(1+np.exp(-valor))

def tanh(valor):
	""" 
	“Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
	me comprometo a que mi actuacion en este examen este regida por la honestidad
	academica.”
	"""
	return 2/(1+np.exp(-valor)) - 1

def trainANN(red, valores_X, valores_Y, funcion):
	""" 
	“Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
	me comprometo a que mi actuacion en este examen este regida por la honestidad
	academica.”
	"""

	#realizamos la multiplicacion de los pesos de la capa oculta y el resultado
	#se le aplica la funcion de activacion
	if funcion == 1:
		act = sigmoid(np.dot(valores_X,red.pesosOculta))
	elif funcion == 2:
		act = lineal(np.dot(valores_X,red.pesosOculta))
	elif funcion == 3:
		act = tanh(np.dot(valores_X,red.pesosOculta))
	
	#se le suma el bias de la red a los valores calculados hasta el momento
	act+=red.biasOculta

	#realizamos la multiplicacion de los pesos a la neurona final
	#se le suma el bias final
	#se aplica la activacion sigmoidal
	output= sigmoid(np.dot(act,red.pesosFinal) + red.biasFinal)

	#se crea una matriz de error de la matriz contra los valores esperados
	error = (output - valores_Y)

	#se calculo el fitness usando el error cuadratico medio basado en la matriz anterior
	red.fitness = 1/valores_Y.size * np.dot(error.transpose((1, 0)),error)


	
#input y validacion de input
crossover = input('Cual operador de cruza desea: \n1 Aritmetico \n2 One point ')
while int(crossover) > 2 or int(crossover) < 1:
	crossover = input('Operador de cruza invalido selecione uno de los siguientes: \n1 Aritmetico \n2 One point ')

mutacion = input('Cual operador de mutacion desea: \n1 Uniforme \n2 Normal  ')
while int(mutacion) > 2 or int(mutacion) < 1:
	mutacion = input('Operador de mutacion invalido selecione uno de los siguientes: \n1 Uniforme \n2 Limite ')

iteraciones_Max = input('Numero maximo de iteracioners: ')
while int(iteraciones_Max) < 2:
	iteraciones_Max = input('Porfavor ingrese un valor mayor a 1')

poblacion = input('Ingrese el tamaño de la poblacion: ')
while int(iteraciones_Max) < 4:
	poblacion = input('Porfavor ingrese un tamaño de poblacion mayor a 3')

funcion_Activacion = input('Cual funcion de activacion desea: \n1 sigmoidal \n2 lineal \n3 Tanh ')
while int(funcion_Activacion) > 3 or int(funcion_Activacion) < 1:
	funcion_Activacion = input('Funcion de activacion invalida selecione una de las siguientes: \n1 sigmoidal \n2 lineal \n3 Tanh ')


data = loadmat('blood.mat')
X = data['X']
Y = data['Y']

contador_individuos = 0

#colecion que contiene la poblacion que paso a la siguiente poblacion
generacion = queue.Queue()

#colecion que ordena a los individuos con el fitness mas bajo para pasar a la siguiente generacio
poblacionEvol = queue.PriorityQueue()

#creacion de la generacion inicial
while i < int(poblacion) :
	#se crea un individuo para ser agregado a la siguiente generacion
	primogenito = Ann(X[0].size)
	#se entrena al individuo
	trainANN(primogenito,X,Y,int(funcion_Activacion))
	#se agrega a la poblacion que paso a la siguiente generacion
	generacion.put(primogenito)
	#agrega 1 a la cuenta de individuos
	contador_individuos+=1

#iniciamos variable ce control de ciclo
iteracionActual = 0
while iteracionActual < int(iteraciones_Max):
	#mientras no se haya procesado cada individuo de la generacion
	while not generacion.empty():
		#sacamos 2 individuos de la generacion
		padre = generacion.get()
		madre = generacion.get()

		#en base al input de usuario cruzamos los individuos
		#y creamos un individuo hijo
		if int(crossover) == 1:
			hijo = crossAritmetico(padre,madre)
		elif int(crossover) == 2:
			hijo = crossAritmetico(padre,madre)

		#en base al input de usurtio el hijo muta
		if int(mutacion) == 1:
			mutacion_Uniforme(hijo)
		elif int(mutacion) == 2:
			mutacion_limite(hijo)

		#se enetrena al hijo
		trainANN(hijo,X,Y,int(funcion_Activacion))

		#se agregar los 3 individuos a la colecion donde se realizara la proportional selection
		poblacionEvol.put(hijo)
		poblacionEvol.put(padre)
		poblacionEvol.put(madre)
		
	#Agregamos la cantidad necesaria de poblacion a la siguiente generacion
	while generacion.qsize() < int(poblacion):
		generacion.put(poblacionEvol.get())

	#dejamos morir a los individuos que no fueron selecionados
	while not poblacionEvol.empty():
		poblacionEvol.get()

	#aumentamos la generacion
	iteracionActual+=1
	print("Generacion",iteracionActual,"fue realizada")


#pruebas
while not generacion.empty():
	mejor = generacion.get()
	print("fitness de individuo ",int(poblacion)-generacion.qsize(), "es ", mejor.fitness)




