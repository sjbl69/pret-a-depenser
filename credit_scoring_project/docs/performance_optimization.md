# Étape 4 — Analyse et optimisation des performances du modèle

## 1. Introduction

L’objectif de cette étape est d’évaluer les performances du modèle en conditions proches de la production via l’API déployée, puis d’identifier et d’implémenter des optimisations afin de réduire la latence tout en conservant des performances prédictives satisfaisantes.

---

## 2. Analyse des performances initiales

Les performances ont été mesurées à partir de requêtes envoyées à l’API déployée, en simulant un usage réel.

### Résultats avant optimisation

| Métrique         | Valeur         |
|------------------|----------------|
| Latence moyenne  | ~0.04 s        |
| Latence maximale | ~0.07 s        |

### Méthodologie

- Envoi de plusieurs requêtes via l’endpoint `/predict`
- Mesure du temps de réponse avec logs API
- Conditions proches de la production (API déployée via Docker + CI/CD)

### Interprétation

Les performances initiales sont déjà bonnes, avec une latence inférieure à 100 ms, ce qui est compatible avec un usage en production.

---

## 3. Identification des goulots d’étranglement

L’analyse des logs et du code a permis d’identifier plusieurs sources de latence :

- création d’un DataFrame à chaque requête
- application du scaler à chaque appel
- absence de pipeline optimisé (traitements répétés)
- chargement non optimisé des objets nécessaires à la prédiction

---

## 4. Optimisations implémentées

Les optimisations suivantes ont été effectivement mises en place dans le code de l’API :

- chargement du modèle et du scaler une seule fois au démarrage de l’API
- suppression des traitements redondants
- simplification du pipeline de prédiction
- réduction de la transformation des données au strict nécessaire

Ces modifications ont été intégrées directement dans le code source de l’API (`app.py`).

---

## 5. Redéploiement du modèle optimisé

Après implémentation des optimisations :

- le code a été versionné et poussé sur la branche principale
- le pipeline CI/CD s’est exécuté automatiquement
- une nouvelle image Docker a été construite et déployée

Cela garantit que les optimisations sont bien actives en production.

---

## 6. Résultats après optimisation

### Mesures après optimisation

| Métrique         | Avant  | Après  |
|------------------|--------|--------|
| Latence moyenne  | ~0.04s | ~0.03s |
| Latence maximale | ~0.07s | ~0.05s |

### Analyse

- amélioration mesurable de la latence (~25%)
- gain surtout visible sur la latence maximale
- comportement plus stable de l’API

---

## 7. Validation des performances du modèle

Les performances prédictives ont été vérifiées après optimisation :

- AUC ≈ 0.74
- aucune dégradation des prédictions observée
- cohérence des résultats entre version avant et après optimisation

---

## 8. Configuration finale retenue

- Modèle : Logistic Regression
- Exécution : CPU
- Pipeline : simplifié et optimisé
- Chargement des objets : au démarrage de l’API

### Justification

Le choix de cette configuration repose sur :

- une latence très faible compatible production
- une complexité technique limitée
- une maintenance facilitée
- un bon compromis performance / simplicité

Des solutions plus avancées (ONNX, GPU) ont été étudiées mais non retenues car :

- gain marginal dans ce contexte
- complexité supplémentaire inutile

---

## 9. Conclusion

Le modèle est désormais optimisé et prêt pour un usage en production.

Les optimisations mises en place ont permis :

- de réduire la latence
- d’améliorer la stabilité de l’API
- de conserver les performances prédictives

Le pipeline CI/CD garantit que toute amélioration future pourra être automatiquement déployée.