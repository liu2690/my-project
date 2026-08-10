from src.experiment import experiment


result = experiment(

    save_path=
    "result/k8_unknown",

    use_unknown=True,

    num_cluster=8,

    epochs=30,

    experiment_repeats=1,

    seed=42

)