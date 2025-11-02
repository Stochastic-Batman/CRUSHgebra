from ucimlrepo import fetch_ucirepo


student_performance = fetch_ucirepo(name='Student Performance')
print(student_performance.metadata.additional_info.summary)

X = student_performance.data.features
y = student_performance.data.targets
print(X.columns)
print(y.columns)