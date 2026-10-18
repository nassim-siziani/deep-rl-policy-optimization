# Apprentissage par renforcement profond (Deep RL)

## Présentation du projet

Ce projet consiste à entraîner un agent autonome pour maintenir un pendule inversé en équilibre dans l'environnement `CartPole-v1` de Gymnasium.

L'objectif est d'explorer et de comparer deux méthodes d'apprentissage par renforcement : une méthode basée sur l'estimation de la valeur (Double Dueling DQN) et une méthode basée sur l'optimisation directe de la politique (REINFORCE avec Baseline).

## Travail réalisé

Dans ce projet, j'ai travaillé sur plusieurs étapes :

- **Modélisation du problème :** prise en main de l'environnement `CartPole-v1`, gestion des états, des actions et des récompenses.

- **Implémentation de Double Dueling DQN :** séparation de l'estimation de la valeur de l'état et de l'avantage de chaque action, et découplage entre le choix et l'évaluation des actions pour éviter de surestimer les scores.

- **Implémentation de REINFORCE avec Baseline :** apprentissage direct des actions avec un réseau de neurones et ajout d'une ligne de base pour réduire les variations et rendre l'apprentissage plus stable.

- **Benchmark et comparaison :** entraînement des deux méthodes sur 200 épisodes et tracé de la courbe pour comparer la vitesse d'apprentissage et la stabilité.

- **Tests unitaires :** vérification du bon fonctionnement des modèles et des calculs avec `pytest`.

## Résultats

L'agent **Double Dueling DQN** apprend rapidement et parvient à maintenir le pendule en équilibre sur la durée maximale (score de 500 points). La méthode **REINFORCE avec Baseline** progresse de manière plus régulière et stabilise ses choix au fil des épisodes.

![Comparaison de convergence](convergence_comparison.png)

## Technologies utilisées

- Python
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- Pytest

## Lancer le projet

Clonez le dépôt :

    git clone git@github.com:nassim-siziani/deep-rl-policy-optimization.git
    cd deep-rl-policy-optimization

Créez l'environnement virtuel et installez les dépendances :

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Lancez les tests unitaires :

    python3 -m pytest tests/

Lancez l'entraînement et générez le graphique :

    python3 -m src.train_benchmark --episodes 200

## Ce que j'ai appris

Ce projet m'a permis de comprendre le fonctionnement pratique du Deep Reinforcement Learning. J'ai appris à créer des modèles de décision avec PyTorch, à gérer une mémoire d'expériences (*replay buffer*) et à stabiliser l'apprentissage grâce aux versions améliorées de DQN et aux gradients de politique.

Il m'a également permis d'appliquer des bonnes pratiques de développement en organisant le code proprement et en écrivant des tests automatisés.
