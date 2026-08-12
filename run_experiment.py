from src.experiment import experiment


experiment(

    save_path=
        "result/k8_unknown_cnn_top100_cv5_test",

    use_unknown=True,

    num_cluster=8,

    top_k_features=100,

    epochs=100,

    batch_size=16,

    cv_folds=5,

    experiment_repeats=10,

    seed=42
)