from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from copy import deepcopy
from _superclases import ClasificadorArbol, Arbol

class ArbolDecisionID3(Arbol, ClasificadorArbol):
    def __init__(self, max_prof: int = -1, min_obs_nodo: int = -1) -> None:
        super().__init__()
        ClasificadorArbol.__init__(self, max_prof, min_obs_nodo)
            
    def _traer_hiperparametros(self, arbol_previo):
        self.max_prof = arbol_previo.max_prof
        self.min_obs_nodo = arbol_previo.min_obs_nodo
        
    def _traer_hiperparametros(self, arbol_previo):
        self.max_prof = arbol_previo.max_prof
        self.min_obs_nodo = arbol_previo.min_obs_nodo
    
    def _mejor_split(self) -> str: 
        mejor_ig = -1
        mejor_atributo = None
        atributos = self.data.columns

        for atributo in atributos:
            ig = self._information_gain(atributo)
            if ig > mejor_ig:
                mejor_ig = ig
                mejor_atributo = atributo
        
        return mejor_atributo

    def _split(self, atributo: str) -> None:
        self.atributo = atributo # guardo el atributo por el cual spliteo
        for categoria in self.data[atributo].unique():
            nueva_data = self.data[self.data[atributo] == categoria]
            nueva_data = nueva_data.drop(atributo, axis = 1) # la data del nuevo nodo sin el atributo por el cual ya se filtró
            nuevo_target = self.target[self.data[atributo] == categoria]
            nuevo_arbol = ArbolDecisionID3()
            nuevo_arbol.data = nueva_data
            nuevo_arbol.target = nuevo_target
            nuevo_arbol.categoria = categoria
            nuevo_arbol.clase = nuevo_target.value_counts().idxmax()
            nuevo_arbol._traer_hiperparametros(self) # hice un metodo porque van a ser muchos de hiperparametros
            self.subs.append(nuevo_arbol)
    
    def entropia(self) -> float:
        entropia = 0
        proporciones = self.target.value_counts(normalize= True)
        target_categorias = self.target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            entropia += proporcion * np.log2(proporcion)
        return -entropia if entropia != 0 else 0
    
    def _information_gain(self, atributo: str) -> float:
        entropia_actual = self.entropia()
        len_actual = len(self.data)

        nuevo = deepcopy(self)
        nuevo._split(atributo)

        entropias_subarboles = 0 

        for subarbol in nuevo.subs:
            entropia = subarbol.entropia()
            len_subarbol = len(subarbol.data)
            entropias_subarboles += ((len_subarbol/len_actual)*entropia)

        information_gain = entropia_actual - entropias_subarboles
        return information_gain       


    def fit(self, X: pd.DataFrame, y: pd.Series):
        '''
        Condicion de split
              - Unico valor para target (nodo puro)
              - No hay mas atributos
              - max_profundidaself.data = X
        '''
        self.target = y
        self.data = X
        self.clase = self.target.value_counts().idxmax()
        
        def _interna(arbol: ArbolDecisionID3, prof_acum: int = 0):
            arbol.target_categorias = y.unique()
            
            if prof_acum == 0:
                prof_acum = 1
            
            if not ( len(arbol.target.unique()) == 1 or len(arbol.data.columns) == 0 
                    or (arbol.max_prof != -1 and arbol.max_prof <= prof_acum) 
                    or (arbol.min_obs_nodo != -1 and arbol.min_obs_nodo > arbol._total_samples() ) ):
                
                mejor_atributo = arbol._mejor_split()
                arbol._split(mejor_atributo)
                for sub_arbol in arbol.subs:
                    _interna(sub_arbol, prof_acum+1)

        _interna(self)
    
    def predict(self, X:pd.DataFrame) -> list[str]:
        predicciones = []

        def _recorrer(arbol, fila: pd.Series) -> None:
            if arbol.es_hoja():
                predicciones.append(arbol.clase)
            else:
                direccion = fila[arbol.atributo]
                for subarbol in arbol.subs:
                    if direccion == subarbol.categoria: #subarbol.valor
                        _recorrer(subarbol, fila)
        
        for _, fila in X.iterrows():
            _recorrer(self, fila)
        
        return predicciones
    
    def imprimir(self, prefijo: str = '  ', es_ultimo: bool = True) -> None:
        simbolo_rama = '└─── ' if es_ultimo else '├─── '
        split = "Split: " + str(self.atributo)
        rta = "Valor: " + str(self.categoria)
        entropia = f"Entropia: {round(self.entropia(), 2)}"
        samples = f"Samples: {str (self._total_samples())}"
        values = f"Values: {str(self._values())}"
        clase = 'Clase: ' + str(self.clase)
        if self.es_raiz():
            print(entropia)
            print(samples)
            print(values)
            print(clase)
            print(split)

            for i, sub_arbol in enumerate(self.subs):
                ultimo: bool = i == len(self.subs) - 1
                sub_arbol.imprimir(prefijo, ultimo)

        elif not self.es_hoja():
            print(prefijo + "│")
            print(prefijo + simbolo_rama + rta)
            prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo +"│" + " " * (len(simbolo_rama) - 1)
            print(prefijo2 + entropia)
            print(prefijo2 + samples)
            print(prefijo2 + values)
            print(prefijo2 + clase)
            print(prefijo2 + split)
            
            prefijo += ' '*10 if es_ultimo else '│' + ' '*9
            for i, sub_arbol in enumerate(self.subs):
                ultimo: bool = i == len(self.subs) - 1
                sub_arbol.imprimir(prefijo, ultimo)
        else:
            prefijo_hoja = prefijo + " "*len(simbolo_rama) if es_ultimo else prefijo + "│" + " "*(len(simbolo_rama) -1)
            print(prefijo + "│")
            print(prefijo + simbolo_rama + rta)
            print(prefijo_hoja + entropia)
            print(prefijo_hoja + samples)
            print(prefijo_hoja + values)
            print(prefijo_hoja + clase)


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError()
        correctas = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        precision = correctas / len(y_true)
        return precision


def probar(df, target:str):
    X = df.drop(target, axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    arbol = ArbolDecisionID3(min_obs_nodo=1)
    arbol.fit(x_train, y_train)
    arbol.imprimir()
    y_pred = arbol.predict(x_test)
    print(f"\naccuracy: {accuracy_score(y_test.tolist(), y_pred)}")
    print(f"cantidad de nodos: {len(arbol)}")
    print(f"altura: {arbol.altura()}\n")


if __name__ == "__main__":
    #https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link
    patients = pd.read_csv("cancer_patients.csv", index_col=0)
    patients = patients.drop("Patient Id", axis = 1)
    bins = [0, 15, 20, 30, 40, 50, 60, 70, float('inf')]
    labels = ['0-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
    patients['Age'] = pd.cut(patients['Age'], bins=bins, labels=labels, right=False)

    tennis = pd.read_csv("PlayTennis.csv")

    print("Pruebo con patients\n")
    probar(patients, "Level")
    print("Pruebo con Play Tennis\n")
    probar(tennis, "Play Tennis")
