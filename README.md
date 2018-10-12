# DSLR42
Repo for the DSLR project of 42, Simon Pagezy and Dimitri Iordanovitch group.

Le fichier describe.py prend un dataset en parametre, et retourne des informations sur les features numeriques.

Les fichiers .py pair_plot, scatter_plot et histogram prennent un dataset en parametre et retournent des 
graphiques sur les donnees.


Le fichier logreg_train.py prend un dataset en parametre et permet d'entrainer une regression logistique sur le 
dataset afin de predire la bonne maison pour chaque eleve.

Le fichier logreg_predict.py prend un dataset en parametre et retourne une prediction des maisons pour chaque 
eleve grace aux poids calcules via le script logreg_train.
