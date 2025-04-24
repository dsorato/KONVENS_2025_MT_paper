import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_data(folder_path):
    """Load all batch files from a folder into a single DataFrame. """
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".tsv")]
    print(all_files)
    df_list = [pd.read_csv(os.path.join(folder_path, file), sep="\t") for file in all_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # Ensure sentence_number is integer
    df["sentence_number"] = pd.to_numeric(df["sentence_number"], errors='coerce').astype('Int64')
    df = df.dropna(subset=["sentence_number"])  # Drop any NaN values in sentence_number
    df["sentence_number"] = df["sentence_number"].astype(int)
    
    # Clean metric values
    metrics = ["bertscore_f1", "CHRF", "COMET"]
    for metric in metrics:
        df[metric] = df[metric].astype(str).str.replace(r'\[|\]', '', regex=True).astype(float)

    df["CHRF_normalized"] = df["CHRF"] / 100.0
    
    return df



def plot_metrics(df, output_folder):
    """Scatter plots with metric on Y-axis and prompt-temperature pair on X-axis"""
    metrics = ["bertscore_f1", "CHRF", "COMET"]

    # Create a new column for prompt-temperature pair
    df["prompt_temp"] = df["prompt"] + "_" + df["temperature"].astype(str)
    df['prompt_temp'] = df['prompt_temp'].str.replace('prompt','p')

    # Ensure sorting order: first by temperature, then by prompt
    df["temperature"] = df["temperature"].astype(float)
    df["prompt_temp"] = pd.Categorical(
        df["prompt_temp"],
        categories=sorted(df["prompt_temp"].unique(), key=lambda x: (float(x.split("_")[1]), x.split("_")[0])),
        ordered=True
    )


    df["CHRF_normalized"] = df["CHRF"] / 100
    df_variance = df.groupby(["sentence_number", "prompt_temp", "prompt", "temperature"])[["bertscore_f1", "CHRF_normalized", "COMET"]].agg(["mean", "var"]).reset_index()

    print('******** MEAN AND VARIANCE BETWEEN BATCHES FOR EACH SENTENCE ID')

    df_variance = df_variance.dropna()
    df_variance.to_csv('mean_and_variance_sentence_ID.tsv', sep='\t', encoding='utf-8', index=False)
    print(df_variance)


    # Group by sentence, prompt, and temperature, then take the mean
    df_avg = df.groupby(["sentence_number", "prompt_temp", "prompt", "temperature"])[metrics].mean().reset_index()


    os.makedirs(output_folder, exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(14, 6))

        # Scatter plot with different colors for prompts
        a = sns.stripplot(data=df_avg, x="prompt_temp", y=metric, hue="prompt", dodge=False, jitter=True, alpha=0.7)

        # Mean values and confidence intervals 
        sns.pointplot(
            data=df_avg, x="prompt_temp", y=metric, dodge=False, linestyle='none', 
            errorbar=('ci', 95), err_kws={'linewidth': 1.5}, markers="D", color="black", zorder=3
        )


        plt.xticks(rotation=20)
        if metric=='CHRF':
            metric='Chrf++'
        elif metric=='bertscore_f1':
            metric='BERTScore F1'
        elif metric=='COMET':
            metric='COMET-22'

        # plt.title(f"{metric} by Prompt-Temperature Pair")
        plt.xlabel("Prompt-Temperature Pair")
        plt.ylabel(metric)
     
        handles, labels = plt.gca().get_legend_handles_labels()

        plt.legend(handles, labels, title="Prompt", bbox_to_anchor=(1, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{metric}_scatter.png"))
        plt.close()



def plot_metrics_by_item_type(df, output_folder, item_type):
    """Scatter plots with metric on Y-axis and prompt-temperature pair on X-axis"""
    metrics = ["bertscore_f1", "CHRF", "COMET"]

    # Create a new column for prompt-temperature pair
    df["prompt_temp"] = df["prompt"] + "_" + df["temperature"].astype(str)
    df['prompt_temp'] = df['prompt_temp'].str.replace('prompt','p')

    # Ensure sorting order: first by temperature, then by prompt
    df["temperature"] = df["temperature"].astype(float)
    df["prompt_temp"] = pd.Categorical(
        df["prompt_temp"],
        categories=sorted(df["prompt_temp"].unique(), key=lambda x: (float(x.split("_")[1]), x.split("_")[0])),
        ordered=True
    )


    df["CHRF_normalized"] = df["CHRF"] / 100
    df_variance = df.groupby(["sentence_number", "prompt_temp", "prompt", "temperature"])[["bertscore_f1", "CHRF_normalized", "COMET"]].agg(["mean", "var"]).reset_index()

    print('******** MEAN AND VARIANCE BETWEEN BATCHES FOR EACH SENTENCE ID')

    df_variance = df_variance.dropna()
    df_variance.to_csv('mean_and_variance_sentence_ID.tsv', sep='\t', encoding='utf-8', index=False)
    print(df_variance)


    # Group by sentence, prompt, and temperature, then take the mean
    df_avg = df.groupby(["sentence_number", "prompt_temp", "prompt", "temperature"])[metrics].mean().reset_index()


    os.makedirs(output_folder, exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(14, 6))

        # Scatter plot with different colors for prompts
        a = sns.stripplot(data=df_avg, x="prompt_temp", y=metric, hue="prompt", dodge=False, jitter=True, alpha=0.7)

        # Mean values and confidence intervals 
        sns.pointplot(
            data=df_avg, x="prompt_temp", y=metric, dodge=False, linestyle='none', 
            errorbar=('ci', 95), err_kws={'linewidth': 1.5}, markers="D", color="black", zorder=3
        )


        plt.xticks(rotation=20)
        if metric=='CHRF':
            metric='Chrf++'
        elif metric=='bertscore_f1':
            metric='BERTScore F1'
        elif metric=='COMET':
            metric='COMET-22'

        # plt.title(f"{metric} by Prompt-Temperature Pair")
        plt.xlabel("Prompt-Temperature Pair")
        plt.ylabel(metric)
     

        # Remove black markers from the legend
        handles, labels = plt.gca().get_legend_handles_labels()

        plt.legend(handles, labels, title="Prompt", bbox_to_anchor=(1, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{metric}_scatter_"+item_type+".png"))
        plt.close()


def plot_comparative_graphs(df, output_folder):
    """Generate comparative line graphs for each metric with X-axis as prompts and temperature."""
    metrics = ["bertscore_f1", "CHRF", "COMET"]
    df_avg = df.groupby(["prompt", "temperature"])[metrics].mean().reset_index()
    
    os.makedirs(output_folder, exist_ok=True)
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_avg, x="prompt", y=metric, hue="temperature", marker="o")

        if metric=='CHRF':
            metric='Chrf++'
        elif metric=='bertscore_f1':
            metric='BERTScore F1'
        elif metric=='COMET':
            metric='COMET-22'

        plt.title(f"Comparative Performance by Prompt and Temperature - {metric}")
        plt.xlabel("Prompt")
        plt.ylabel(metric)
        plt.savefig(os.path.join(output_folder, f"comparative_performance_{metric}.png"))
        plt.close()


def find_best_prompt_temperature(df):
    """Determine the best prompt and temperature combination."""
    metric_means = df.groupby(["prompt", "temperature"])[["bertscore_f1", "CHRF", "COMET"]].mean()
    best_combo = metric_means.mean(axis=1).idxmax()
    return f"Best prompt and temperature: {best_combo}\n\n{metric_means}"

def plot_metrics_filtered(df, aux_df, output_folder, item_type):
    """Generate plots for each metric using only selected sentence IDs from the auxiliary file."""
    sentence_ids = aux_df[aux_df["item_type"] == item_type]["sentence_number"].tolist()    
    df_filtered = df[df["sentence_number"].isin(sentence_ids)]

    plot_metrics_by_item_type(df_filtered, output_folder, item_type)

def plot_sentence_level_heatmaps(df, output_folder, item_type):
    metrics = ["bertscore_f1", "CHRF", "COMET"]
    df_avg = df.groupby(["sentence_number", "prompt_temp", "prompt", "temperature"])[metrics].mean().reset_index()
    df_avg=df_avg.dropna()

    df_avg["prompt_temp"] = df_avg["prompt"] + "_" + df_avg["temperature"].astype(str)
    df_avg['prompt_temp'] = df_avg['prompt_temp'].str.replace('prompt','p')

    for metric in metrics:
        df_pivot = df_avg.pivot(index="sentence_number", columns="prompt_temp", values=metric)

        if metric=='CHRF':
            metric='Chrf++'
        elif metric=='bertscore_f1':
            metric='BERTScore F1'
        elif metric=='COMET':
            metric='COMET-22'

        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_pivot, annot=False, cmap="coolwarm", cbar=True, linewidths=0.5, linecolor="black")
        # plt.title(f"Sentence-Level {metric} Heatmap for "+item_type.lower()+" sentences")
        plt.xlabel("Prompt-temperature pair")
        plt.ylabel("Sentence ID")
        plt.tick_params(axis='both', labelsize=13)
        # plt.xticks(rotation=50)
        
        plt.savefig(f"{output_folder}/sentence_level_{metric}_heatmap_"+item_type+".png", bbox_inches="tight", dpi=300)
        plt.close()


def find_best_prompt_temperature_filtered(df, aux_df, item_type, output_folder):
    """Determine the best prompt and temperature combination using only selected sentence IDs."""
    if item_type=="RESPONSE":
        response_filtered = aux_df[(aux_df["item_type"] == item_type) & (aux_df["response_has_qualifier_intensifier"] == 1)]
        sentence_ids = response_filtered["sentence_number"].tolist() 
    else:
        sentence_ids = aux_df[aux_df["item_type"] == item_type]["sentence_number"].tolist()
    
    df_filtered = df[df["sentence_number"].isin(sentence_ids)]
    plot_sentence_level_heatmaps(df_filtered, output_folder, item_type)

    return find_best_prompt_temperature(df_filtered)



def get_best_performance_per_sentence_ID(df_metrics):
    """
    Identify the best batch, prompt, and temperature for each sentence_number based on the highest 
    BERTScore, CHRF, and COMET scores
    """

    df_best_bertscore = df_metrics.loc[df_metrics.groupby("sentence_number")["bertscore_f1"].idxmax(), ["sentence_number", "batch", "prompt", "temperature", "bertscore_f1"]]
    df_best_chrf = df_metrics.loc[df_metrics.groupby("sentence_number")["CHRF"].idxmax(), ["sentence_number", "batch", "prompt", "temperature", "CHRF"]]
    df_best_comet = df_metrics.loc[df_metrics.groupby("sentence_number")["COMET"].idxmax(), ["sentence_number", "batch", "prompt", "temperature", "COMET"]]

    df_best_bertscore.rename(columns={"batch": "batch_bertscore", "prompt": "prompt_bertscore", "temperature": "temp_bertscore"}, inplace=True)
    df_best_chrf.rename(columns={"batch": "batch_chrf", "prompt": "prompt_chrf", "temperature": "temp_chrf"}, inplace=True)
    df_best_comet.rename(columns={"batch": "batch_comet", "prompt": "prompt_comet", "temperature": "temp_comet"}, inplace=True)

    df_best = df_best_bertscore.merge(df_best_chrf, on="sentence_number").merge(df_best_comet, on="sentence_number")

    df_best.to_csv('prompts_by_sentence.tsv', sep='\t', encoding='utf-8', index=False)

    return df_best

def main(folder_path, structure_mapping_file, output_folder):
    df = load_data(folder_path)
    structure_mapping_file_df = pd.read_csv(structure_mapping_file, sep="\t")

    df_best = get_best_performance_per_sentence_ID(df)
    print(df_best)
    plot_metrics(df, output_folder)
    plot_comparative_graphs(df, output_folder)
    best_prompt_temperature = find_best_prompt_temperature(df)
    print('***************')
    print(best_prompt_temperature)
    with open(os.path.join(output_folder, "best_performance.txt"), "w") as f:
        f.write(best_prompt_temperature)
    
    item_types = ["INSTRUCTION", "RESPONSE"]

    for item_type in item_types:
        plot_metrics_filtered(df, structure_mapping_file_df, output_folder, item_type)
        best_prompt_temperature_filtered = find_best_prompt_temperature_filtered(df, structure_mapping_file_df, item_type, output_folder)
        print('***************')
        print(best_prompt_temperature_filtered)
        with open(os.path.join(output_folder, "best_performance_filtered_"+item_type+".txt"), "w") as f:
            f.write(best_prompt_temperature_filtered)


if __name__ == "__main__":
    current_dir = os.getcwd()
    folder_path = current_dir+"/mt_metrics_and_plots/" 
    output_folder = current_dir

    structure_mapping_file = "dataset_structure_mapped.tsv"  

    main(folder_path, structure_mapping_file, output_folder)
