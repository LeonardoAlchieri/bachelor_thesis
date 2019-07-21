# BACHELOR'S THESIS
Code to clean Planck/HFI and the neural network to analyse it.

## NOTE GENERALI ##
  – Dire nella tesi che, per mancanza di tempo (abbiamo scoperto troppo tardi), non sono riuscito a discriminare 
    tra sorgenti puntiformi e glitch (che a occhio appaiono uguali). Dire che comunque in un esperimento reale questa cosa
    sarebbe facilmente eseguibile: se un "glitch" appare in due giorni diversi nello stesso posto, significa che è una
    sorgente puntiforme.

## COSE FATTE ##
  – Ho aggiunto al codice per il test ```main_network_test.py``` il calcolo di una _confusion matrix_ e della curva _ROC_ (fatta su un numero grandissimo di possibili thresholds, 300000). 

## DOMANDE ##  
  – Devo andare anche a calcolare l'area sotto la _ROC_ (nota come _AUC_)? Ritengo che semplicemente mostrare come la curva 
    "molto bella" (ovvero quasi tutta schiacciata a sinistra) sia già abbastanza per far vedere che il mio network funziona 
    molto bene.

©Leonardo Alchieri, 2019
