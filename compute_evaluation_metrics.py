import sys
import re
import os
import string
import evaluate 
from evaluate import load
import comet
from comet import download_model, load_from_checkpoint
import difflib
import pandas as pd
from huggingface_hub import login
from sacrebleu import CHRF


punctuation_to_remove = string.punctuation.replace("/", "")

def remove_punctuation(sentence: str) -> str:
    cleaned = sentence.translate(str.maketrans('', '', punctuation_to_remove))

    # cleaned = cleaned.lower()
    return cleaned


def highlight_differences_html(ref_translation: str, mt_translation: str) -> str:
    ref_words = ref_translation.split()
    mt_words = mt_translation.split()

    diff = list(difflib.ndiff(ref_words, mt_words))
    highlighted_diff = []

    for word in diff:
        if word.startswith("- "):  # Word removed
            highlighted_diff.append(f'<span style="color:red; text-decoration:line-through;">{word[2:]}</span>')
        elif word.startswith("+ "):  # Word added
            highlighted_diff.append(f'<span style="color:green; text-decoration:underline;">{word[2:]}</span>')
        else:
            highlighted_diff.append(word[2:])  # Unchanged word
    
    return " ".join(highlighted_diff)



def compute_metrics_without_source(clean_ref, clean_mt, bertscore, chrf):
    # CHRF = chrf.corpus_score([clean_mt], [clean_ref])
    CHRF = chrf.compute(predictions=[clean_mt], references=[[clean_ref]], word_order=2, lowercase=True)
    bscore= bertscore.compute(predictions=[clean_mt], references=[clean_ref], lang="de", verbose=True)


    return bscore['precision'], bscore['recall'], bscore['f1'], CHRF['score']

def compute_comet(clean_ref, clean_mt, clean_source, comet_metric):
    results = comet_metric.compute(predictions=[clean_mt], references=[clean_ref], sources=[clean_source])

    return results["scores"]



def build_html_structure():
    html_header = """
    <html>
    <head>
        <title>Text Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f4f4f4; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            span { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Text Comparison Report</h1>
    """

    html_body = ""

    return html_header, html_body

def main(mt_translation_folder, batch_number):
    df_results = pd.DataFrame(columns=['sentence_number', 'batch', 'model', 'prompt', 'temperature', 'bertscore_precision', 
        'bertscore_recall', 'bertscore_f1', 'CHRF','COMET'])

    source_translations = 'questionnaire.txt'
    reference_translations = 'reference_translations.txt'
    
    chrf = evaluate.load("chrf")
    bertscore = load("bertscore")

    comet_metric = load('comet')

    if "mini" in mt_translation_folder:
        model="ChatGPT 4 mini"
    else:
        model="ChatGPT 4"

    print(model)

    html_header, html_body = build_html_structure()
    with open('html_report_model_'+model+'.html', 'w+', encoding='utf-8') as html_report:
        html_report.write(html_header)

    #iterates through all text files in a folder to compare them against the reference
    for filename in os.listdir(mt_translation_folder):
        if filename.endswith(".txt"):  
            automatic_translations = os.path.join(mt_translation_folder, filename)
            split_filename = filename.split("_")
            prompt=split_filename[3]
            print(f"prompt: {prompt}")
            temperature=split_filename[6].split('.txt')[0]
            print(f"temperature: {temperature}")

            

            with open(reference_translations, 'r', encoding='utf-8') as ref, open(automatic_translations, 'r', encoding='utf-8') as mt, open(source_translations, 'r', encoding='utf-8') as source:
                ref_lines = ref.readlines()
                mt_lines = mt.readlines()
                source_lines = source.readlines()


                html_body += f"<h2>Comparing: {filename}</h2>"
                html_body += """
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Prompt</th>
                            <th>Temperature</th>
                            <th>Sentence #</th>
                            <th>Source</th>
                            <th>Reference translation</th>
                            <th>MT translation</th>
                            <th>Differences</th>
                            <th>bertscore_precision</th>
                            <th>bertscore_recall</th>
                            <th>bertscore_f1</th>
                            <th>CHRF</th>
                            <th>COMET</th>
                        </tr>
                        """
                max_lines = len(ref_lines)
                results = []

                for i in range(max_lines):
                    clean_source = remove_punctuation(source_lines[i].strip())
                    clean_ref = remove_punctuation(ref_lines[i].strip())
                    clean_mt = remove_punctuation(mt_lines[i].strip())

                    print(f"Source: {clean_source}")
                    print(f"Reference Translation: {clean_ref}")
                    print(f"Automatic Translation: {clean_mt}")

                    if clean_mt=="MISSING":
                        bertscore_precision=bertscore_recall=bertscore_f1=CHRF = 0
                        comet_score = 0 
                    else:
                        bertscore_precision, bertscore_recall, bertscore_f1, CHRF = compute_metrics_without_source(clean_ref, clean_mt, bertscore, chrf)
                        comet_score = compute_comet(clean_ref, clean_mt, clean_source, comet_metric)

                    data = {'sentence_number': str(i), 'batch': batch_number, 'model': model, 'prompt': prompt, 'temperature': temperature, 'bertscore_precision':bertscore_precision[0], 
		              'bertscore_recall': bertscore_recall[0], 'bertscore_f1':bertscore_f1[0], 'CHRF':CHRF, 'COMET':comet_score[0]}

                    df_results = pd.concat([df_results, pd.DataFrame([data])], ignore_index=True)

                    highlighted_diff = highlight_differences_html(clean_ref, clean_mt)
                    results.append(f"""
                        <tr>
                            <td>{model}</td>
                            <td>{prompt}</td>
                            <td>{temperature}</td>
                            <td>MTE_GER_{i}</td>
                            <td>{clean_source}</td>
                            <td>{clean_ref}</td>
                            <td>{clean_mt}</td>
                            <td>{highlighted_diff}</td>
                            <td>{bertscore_precision}</td>
                            <td>{bertscore_recall}</td>
                            <td>{bertscore_f1}</td>
                            <td>{CHRF}</td>
                            <td>{comet_score}</td>
                        </tr>""")

     
                html_body += "".join(results)  


                html_body += "</table>"
                html_footer = """
                    </body>
                    </html>"""
                

                with open('html_report_model_'+model+'.html', 'a+', encoding='utf-8') as html_report:
                    html_report.write(html_body)

    with open('html_report_model_'+model+'.html', 'a+', encoding='utf-8') as html_report:
        html_report.write(html_footer)

    df_results.to_csv('mt_similarity_with_baseline_'+model+'.tsv', sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    mt_translation_folder = str(sys.argv[1])
    batch_number = str(sys.argv[2])
    main(mt_translation_folder, batch_number)