# KONVENS_2025_MT_paper

Code and graphs used in the ''Evaluating the Feasibility of Using ChatGPT for Cross-cultural Survey Translation'' KONVENS 2025 submission 

questionnaire.txt -> Laboratory questionnaire

reference_translations.txt -> Human reference translations produced through the TRAPD methodology

gpt_outputs -> folder containing the outputs of the GPT‑4o mini using the 5 different prompts mentioned in the paper, generated using the prompt_chatgpt.py file

biterms.csv -> list of biterms automatically extracted from the MCSQ-aligned data (available in folder MCSQ_aligned_data) through the biterm_extractor.py file

mt_metrics_and_plots -> Folder containing the plots generated using the plot_graphs.py file. Metrics were computed using compute_evaluation_metrics.py file

human_questionnaire_answers -> Questionnaire and participant answers used in the human evaluation step
