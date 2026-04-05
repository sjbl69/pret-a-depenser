#  Étape 4 — Analyse et optimisation des performances du modèle

## 1. Introduction

L’objectif de cette étape est d’analyser les performances du modèle en conditions proches de la production, puis d’identifier des pistes d’optimisation afin d’améliorer le temps d’inférence et la latence.

---

## 2. Analyse des performances initiales

Une simulation d’inférence a été mise en place afin d’évaluer le comportement du modèle.

### Méthodologie

* Simulation de 100 prédictions
* Mesure du temps d’inférence pour chaque requête
* Calcul des métriques suivantes :

  * latence moyenne
  * latence maximale
  * latence minimale

### Résultats

| Métrique         | Valeur         |
| ---------------- | -------------- |
| Latence moyenne  | ~0.10 - 0.15 s |
| Latence maximale | ~0.20 s        |
| Latence minimale | ~0.05 s        |

### Interprétation

Les performances sont correctes mais peuvent être améliorées pour un usage en production à grande échelle.

---

## 3. Identification des goulots d’étranglement

Les principaux points d’amélioration identifiés sont :

* Temps d’inférence relativement élevé
* Modèle non optimisé pour la production
* Absence de format optimisé (ONNX)

---

## 4. Stratégie d’optimisation

Une optimisation via ONNX est envisagée afin de :

* réduire la latence
* améliorer les performances d’exécution
* faciliter le déploiement

---

## 5. Résultats attendus après optimisation

| Métrique         | Avant   | Après ONNX |
| ---------------- | ------- | ---------- |
| Latence moyenne  | ~0.12 s | ~0.04 s    |
| Latence maximale | ~0.20 s | ~0.08 s    |
| Latence minimale | ~0.05 s | ~0.02 s    |

---

## 6. Validation

L’optimisation devra être validée en vérifiant :

* la cohérence des prédictions
* l’absence de dégradation des performances (AUC, accuracy)

---

## 7. Intégration CI/CD

La version optimisée du modèle sera intégrée au pipeline CI/CD afin d’être automatiquement déployée.

---

## 8. Conclusion

L’analyse des performances a permis d’identifier des axes d’amélioration.
L’utilisation d’ONNX permettra de réduire significativement le temps d’inférence et d’améliorer la robustesse du modèle en production.

---

## 9. Perspectives

* Implémentation réelle d’ONNX
* Ajout de profiling (cProfile)
* Monitoring avancé
