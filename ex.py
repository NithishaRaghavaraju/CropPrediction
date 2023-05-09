#Random Forest
RFmodel = RandomForestClassifier()
RFmodel.fit(X_train, y_train)
y_pred = RFmodel.predict(X_test)
RFaccuracy = accuracy_score(y_test, y_pred)
Accuracy.append(RFaccuracy)
FalseRate.append(100-RFaccuracy)


#BaggingClassifier
BCmodel = BaggingClassifier()
BCmodel.fit(X_train, y_train)
y_pred = BCmodel.predict(X_test)
ETaccuracy = accuracy_score(y_test, y_pred)
Accuracy.append(BCaccuracy)
FalseRate.append(100-BCaccuracy)


#DecisionTreeClassifier
DTmodel = DecisionTreeClassifier()
DTmodel.fit(X_train, y_train)
y_pred = ETmodel.predict(X_test)
DTaccuracy = accuracy_score(y_test, y_pred)
Accuracy.append(DTaccuracy)
FalseRate.append(100-DTaccuracy)



#ExtraTreesClassifier
ETmodel = ExtraTreesClassifier()
ETmodel.fit(X_train, y_train)
y_pred = ETmodel.predict(X_test)
ETaccuracy = accuracy_score(y_test, y_pred)
Accuracy.append(ETaccuracy)
FalseRate.append(100-ETaccuracy)


#KNeighborsClassifier
KNmodel = KNeighborsClassifier()
KNmodel.fit(X_train, y_train)
y_pred = KNmodel.predict(X_test)
KNaccuracy = accuracy_score(y_test, y_pred)
Accuracy.append(accuracy)
FalseRate.append(100-KNaccuracy)