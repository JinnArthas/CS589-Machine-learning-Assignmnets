

def GridSearch(clf, parameter, folds):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.metrics import make_scorer
    scoring = {'accuracy': make_scorer(accuracy_score)} # scoring = Accuracy
    grid = GridSearchCV(clf, param_grid= parameter, cv = folds, scoring= "accuracy") # performing gridsearch
    grid.fit(train_x, train_y) # fitting the data on gridsearch
    best_clf = grid.best_estimator_ #taking the best estimator
    best_clf.fit(train_x, train_y) # fitting the training data in the best estimator
    predicted_y = best_clf.predict(test_x)  # predicting the test data
    print("The best parameter: ", grid.best_params_)
    print("grid Score for the best parameter: ", grid.grid_scores_)
    return grid.best_params_, grid.grid_scores_, predicted_y
