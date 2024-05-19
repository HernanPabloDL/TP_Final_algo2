
from copy import deepcopy
from ArbolDecisionID3 import ArbolDecisionID3
import pandas as pd
import numpy as np
from _superclases import ClasificadorArbol, Arbol
from typing import Any, Optional

class ArbolDecisionC45(ArbolDecisionID3):
    def __init__(self, max_prof: int = -1, min_obs_nodo: int = -1) -> None:
        super().__init__(max_prof, min_obs_nodo)
        
    def _split_continuo(self, atributo: str) -> None:
        valores_ordenados = sorted(self.data[atributo].unique())
        umbrales = [(valores_ordenados[i] + valores_ordenados[i+1]) / 2 for i in range(len(valores_ordenados) - 1)]
        mejor_ig_ratio = -1
        mejor_umbral = None
        for umbral in umbrales:
            ig_ratio = self._information_gain_ratio_continuo(atributo, umbral)
            if ig_ratio > mejor_ig_ratio:
                mejor_ig_ratio = ig_ratio
                mejor_umbral = umbral
        self._split(atributo, mejor_umbral)    
    
    def _information_gain_ratio_continuo(self, atributo: str, umbral: float) -> float:
        grupo_izquierdo = self.data[self.data[atributo] <= umbral]
        grupo_derecho = self.data[self.data[atributo] > umbral]
        entropia_grupo_izquierdo = self._entropia(grupo_izquierdo)
        entropia_grupo_derecho = self._entropia(grupo_derecho)
        entropia_atributo = (len(grupo_izquierdo) / len(self.data)) * entropia_grupo_izquierdo \
                            + (len(grupo_derecho) / len(self.data)) * entropia_grupo_derecho
        entropia_distribucion_atributo = self._entropia(self.data[atributo])
        split_info = self._entropia(grupo_izquierdo[atributo]) + self._entropia(grupo_derecho[atributo])
        if split_info != 0:
            gain_ratio = (entropia_distribucion_atributo - entropia_atributo) / split_info
        else:
            gain_ratio = 0
        return gain_ratio      
    
    def _entropia(self, data: Optional[pd.Series] = None) -> float:
        if data is None:
            data = self.data
        proporciones = data.value_counts(normalize=True)
        entropia = -(proporciones * np.log2(proporciones)).sum()
        return entropia if not np.isnan(entropia) else 0   
    
    def _mejor_umbral_split(self, atributo: str) -> float:
        self.data = self.data.sort_values(by=atributo)
        self.target = self.target.loc[self.data.index]
        mejor_ig = -1
        mejor_umbral = None
        for umbral in self.data[atributo].values:
            ig_ratio = self._information_gain_ratio_continuo(atributo, umbral) 
            if ig_ratio > mejor_ig:
                mejor_ig = ig_ratio
                mejor_umbral = umbral
        return float(mejor_umbral)
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.target = y
        self.data = X
        self.clase = self.target.value_counts().idxmax()
        def _interna(arbol: ArbolDecisionC45, prof_acum: int = 0):
            arbol.target_categorias = y.unique()
            if prof_acum == 0:
                prof_acum = 1
            if not ( len(arbol.target.unique()) == 1 or len(arbol.data.columns) == 0 
                    or (arbol.max_prof != -1 and arbol.max_prof <= prof_acum) 
                    or (arbol.min_obs_nodo != -1 and arbol.min_obs_nodo > arbol._total_samples() ) ):
                mejor_atributo = arbol._mejor_atributo_split()
                if pd.api.types.is_numeric_dtype(self.data[mejor_atributo]): 
                    mejor_umbral = arbol._mejor_umbral_split(mejor_atributo)
                    arbol._split(mejor_atributo, mejor_umbral)
                else:
                    arbol._split(mejor_atributo)
                for sub_arbol in arbol.subs:
                    _interna(sub_arbol, prof_acum+1)
        _interna(arbol) 
    
    def imprimir(self, prefijo: str = '  ', es_ultimo: bool = True) -> None:
        simbolo_rama = '└─── ' if es_ultimo else '├─── '
        split = f"Split: {str(self.atributo)}"
        rta = f"Valor: > {str(self.valor)}" if es_ultimo else f"Valor: <= {str(self.valor)}"
        entropia = f"Entropia: {round(self._entropia(), 2)}"
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
                
#----------------------------------------------------------------------------------                
if __name__ == "__main__":
    import sklearn.datasets
    iris = sklearn.datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    X = df.drop("target", axis = 1)
    y = df["target"]

    arbol = ArbolDecisionC45(max_prof=3)
    arbol.fit(X, y)
    arbol.imprimir()