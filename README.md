# BACHELOR'S THESIS
Code to clean Planck/HFI and the neural network to analyse it.

## NOTE GENERALI ##
  – Dire nella tesi che, per mancanza di tempo (abbiamo scoperto troppo tardi), non sono riuscito a discriminare 
    tra sorgenti puntiformi e glitch (che a occhio appaiono uguali). Dire che comunque in un esperimento reale questa cosa
    sarebbe facilmente eseguibile: se un "glitch" appare in due giorni diversi nello stesso posto, significa che è una
    sorgente puntiforme.

## COSE DA FARE ##
  - Calcola area sotto curva ROC.
  - Aggiustare le figure più rappresentative per i _glitches_ e i _non-glitches_ (asse y). 

    
## COSE FATTE ##
  – Ho aggiunto al codice per il test ```main_network_test.py``` il calcolo di una _confusion matrix_ e della curva _ROC_ (fatta su un numero grandissimo di possibili thresholds, 300000). 
  - (12 Agosto) Aggiunto tutti i titoli dei paragrafi. Penso che siano abbastanza in ordine logico. Ho sistemato gli assi y delle figure presenti. 

## DOMANDE ##  


## INFO ##
  – Sono andato a provare a fare una seconda ROC utilizzando dei nuovi dati (sempre però dallo stesso pool di quegli altri). In questo caso, ho ottenuto una precisione inferiore (98%) sulla carta.
    Andando però a visualizzare i dati che sono stati classificati in maniera "sbagliata", ho notato che sono delle situazioni in cui, molto probabilmente, ho sbagliato io quando ho eseguito la 
    la classificazione manuale. 
    Per vedere di che cosa parlo, vedere in locale il notebook ```TESTING.ipynb```.

©Leonardo Alchieri, 2019
