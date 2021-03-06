import numpy as np

#individuo utilizado para el algoritmo genetico, contiene 2 matrices qu representa los pesos
# y 2 arreglos que representan los bias, ademas contiene el fitness de cada individuo

""" 
  “Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
  me comprometo a que mi actuacion en este examen este regida por la honestidad
  academica.”
  """
class Ann:
  def __init__(self, k):
    """ 
    “Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
    me comprometo a que mi actuacion en este examen este regida por la honestidad
    academica.”
    """

  	#k se refiere a las neuonas de entrada
  	#creamos matriz de k por 5 neuronas en la capa oculta
    self.pesosOculta = np.random.rand(k,5)
    #necesitamos un vector con el bias de cada neurona
    self.biasOculta = np.random.rand(5)
    #pesos que van de la cpa oculta a la neurona final
    self.pesosFinal = np.random.rand(5,1)
   	#bias final de la red
    self.biasFinal = np.random.rand(1)
    #fitness de la red
    self.fitness = 1000

    self.size = k

  def imprime(self):
    """ 
    “Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
    me comprometo a que mi actuacion en este examen este regida por la honestidad
    academica.”
    """

    print("Pesos de capa oculta\n",self.pesosOculta)
    print("Bias de capa oculta\n",self.biasOculta)
    print("Pesos de neurona Output\n",self.pesosFinal)
    print("Bias de neurona Output\n",self.biasFinal)
    print("Fitness\n",self.fitness)

  def __lt__(self, other):
    """ 
    “Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey,
    me comprometo a que mi actuacion en este examen este regida por la honestidad
    academica.”
    """
    return self.fitness < other.fitness