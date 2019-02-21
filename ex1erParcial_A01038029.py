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

def trainANN(red, valores_X, valores_Y):
	""" 
	“Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
	me comprometo a que mi actuacion en este examen este regida por la honestidad
	academica.”
	"""
	act = lineal(np.dot(valores_X,red.pesosOculta))
	#print("Activacion\n",act)
	act+=red.biasOculta
	#print("bias\n",act)

	output= sigmoid(np.dot(act,red.pesosFinal))
	#print("final\n",output)

	output+=red.biasFinal
	#print("final\n",output)

	error = (output - valores_Y)
	red.fitness = 1/valores_Y.size * np.dot(error.transpose((1, 0)),error)


	"""-
#input y validacion de input
crossover = input('Cual operador de cruza desea: \n1 Aritmetico \n2 One point ')
while int(crossover) > 2 or int(crossover) < 1:
	crossover = input('Operador de cruza invalido selecione uno de los siguientes: \n1 Aritmetico \n2 One point ')

mutacion = input('Cual operador de mutacion desea: \n1 Uniforme \n2 Normal  ')
while int(mutacion) > 2 or int(mutacion) < 1:
	mutacion = input('Operador de mutacion invalido selecione uno de los siguientes: \n1 Uniforme \n2 Normal ')

iteraciones_Max = input('Numero maximo de iteracioners: ')

poblacion = input('Ingrese el tamaño de la poblacion: ')

funcion_Activacion = input('Cual funcion de activacion desea: \n1 sigmoidal \n2 lineal \n3 Tanh')
while int(funcion_Activacion) > 3 or int(funcion_Activacion) < 1:
	funcion_Activacion = input('Funcion de activacion invalida selecione una de las siguientes: \n1 sigmoidal \n2 lineal \n3 Tanh')

	"""



data = loadmat('blood.mat')
X = data['X']
Y = data['Y']


primogenito = Ann(X[0].size)
primogenita = Ann(X[0].size)
primogenito.imprime()


trainANN(primogenito,X,Y)
primogenito.imprime()

mutacion_Uniforme(primogenito)

trainANN(primogenito,X,Y)
primogenito.imprime()

poblacionEvol = queue.PriorityQueue()

poblacionEvol.put(primogenita)
poblacionEvol.put(primogenito)

poblacionEvol.get(0).imprime()
poblacionEvol.get(1).imprime()



primogenita.imprime()

