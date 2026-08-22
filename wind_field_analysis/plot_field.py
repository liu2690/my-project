import matplotlib.pyplot as plt


def plot_field(
        data,
        ylabel,
        title,
        save
):

    plt.figure(
        figsize=(12,5)
    )


    plt.imshow(
        data,
        aspect="auto",
        origin="lower"
    )


    plt.colorbar(
        label="Velocity"
    )


    plt.xlabel(
        "Time sample"
    )


    plt.ylabel(
        ylabel
    )


    plt.title(
        title
    )


    plt.tight_layout()


    plt.savefig(
        save,
        dpi=300
    )


    plt.show()