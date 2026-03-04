import optuna
study = optuna.load_study(study_name='yolov7_hpo', storage='sqlite:///optuna_yolov7.db')

optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_optimization_history(study).show()
