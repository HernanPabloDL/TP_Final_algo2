from ArbolDecisionID3 import *
from ArbolDecisionC45 import *

if __name__ == "__main__":
    #https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link
    
    patients = pd.read_csv("cancer_patients.csv", index_col=0)

    patients = patients.drop("Patient Id", axis = 1)
    bins = [0, 15, 20, 30, 40, 50, 60, 70, float('inf')]
    labels = ['0-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
    patients['Age'] = pd.cut(patients['Age'], bins=bins, labels=labels, right=False)

    tennis = pd.read_csv("PlayTennis.csv")

    print("Pruebo Algoritmo ID3 con patients\n")
    probar_ID3(patients, "Level")
    print("Pruebo Algoritmo ID3 con Play Tennis\n")
    probar_ID3(tennis, "Play Tennis")

    print("Pruebo Algoritmo C4.5 con patients\n")
    probar_C45(patients, "Level")
    print("Pruebo Algoritmo C4.5 con Play Tennis\n")
    probar_C45(tennis, "Play Tennis")