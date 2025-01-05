import matplotlib.pyplot as plt
import pandas as pd
import os

# Dosya yolu ve isimleri ile ilgili ayar
def create_results_folder(folder_name="model_results"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def save_results_to_csv(models, folder_name):
    results_df = pd.DataFrame(models, columns=["Algorithm", "Train Accuracy", "Test Accuracy", "Accuracy Score", "Performance Time"])
    result_file_path = os.path.join(folder_name, "model_performance.csv")
    results_df.to_csv(result_file_path, index=False)
    print(f'Results saved to {result_file_path}')


# Grafiklerini PNG formatında kaydetmek
def save_plot_to_png(fig, algorithm_name, folder_name="model_results"):
    fig_filename = f"{algorithm_name}_performance_plot.png"
    filepath = os.path.join(folder_name, fig_filename)
    fig.savefig(filepath)
    print(f"Plot saved to {filepath}")

# Plotları PDF formatında kaydetmek
def save_plot_to_pdf(fig, algorithm_name, folder_name="model_results"):
    fig_filename = f"{algorithm_name}_performance_plot.pdf"
    filepath = os.path.join(folder_name, fig_filename)
    fig.savefig(filepath)
    print(f"Plot saved to {filepath}")
