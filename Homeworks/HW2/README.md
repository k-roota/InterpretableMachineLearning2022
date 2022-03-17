Add your second homework to this folder.

Deadline 2021-03-25 EOD


Task:
For a selected data set (you can use data from your project or data from Homework 1) prepare a knitr/jupiter notebook with the following points.
Submit your results on GitHub to the directory Homeworks/H2.

TODO:
1. For the selected data set, train at least one tree-based ensemble model (random forest, gbm, catboost or any other boosting)
2. for some selected observations (two or three) from this dataset, calculate predictions for model (1)
3. for observations selected in (2), calculate the decomposition of model prediction using SHAP, Break Down or both (packages for R: DALEX, iml, packages for python: dalex, shap).
4. find two observations in the data set, such that they have different variables of the highest importance (e.g. age and gender are the most important for observation A, but race and class for observation B)
5. (if possible) select one variable and find two observations in the data set such that for one observation this variable has a positive effect and for the other a negative effect
6. train a second model (of any class, neural nets, linear, other boosting) and find an observation for which BD/shap attributions are different between the models
7. Comment on the results for points (4), (5) and (6)


**Important note:**

The submitted homework should consist of two parts (try to render html file out of your jupiter notebook). 

The first part is the key results and comments from points 3,4,7. In this part **PLESE DO NOT SHOW ANY R/PYTHON CODES, RESULTS (IMAGES, COMMENTS) ONLY.**

The second part should start with the word Appendix or Załącznik and should include the reproducible R/PYTHON code used to implement points 1-6.

Such division 1. will make these homework more readable, 2. will create good habits related to reporting.
