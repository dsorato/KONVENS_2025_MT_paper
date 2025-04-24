import openai
import pandas as pd
import json
import os
import sys
from openai import OpenAI



def load_files(questionnaire_path, biterms_path):
    """Load questionnaire and biterms."""
    with open(questionnaire_path, 'r', encoding='utf-8') as f:
        questionnaire = f.read()
    biterms = pd.read_csv(biterms_path, sep='\t')


    return questionnaire, biterms


def create_prompt_without_biterms(context, questionnaire):
    """Combine context and questionnaire into a single prompt."""
    prompt = f"""{context}

Questionnaire:
{questionnaire}

"""

    print(prompt)
    return prompt

def create_prompt(context, questionnaire, biterms):
    """Combine context, questionnaire and biterms into a single prompt."""
    biterms_list = "\n".join(f"{row['source_term']} -> {row['target_term']}" for _, row in biterms.iterrows())

    prompt = f"""{context}

Questionnaire:
{questionnaire}

Useful examples:
{biterms_list}

"""

    print(prompt)
    return prompt



def translate_questionnaire(persona, prompt, model_temperature):
    """Send the prompt to OpenAI API and return the response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        #model="gpt-4o",  
        messages=[
            {"role": "system", "content": persona},
            {"role": "user", "content": prompt}
        ],
        temperature=model_temperature # Adjust creativity level
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content

def save_output(output, output_path):
    """Save the output to a file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

def main(batch_number):
    questionnaire_path = 'questionnaire.txt'
    biterms_path = 'biterms.csv'
    output_filename = 'translated_questionnaire_'

    #Persona
    persona = "You are a professional translator specialized in translating survey questionnaires from English (from Great Britain) to German (from Germany) who works for social survey projects like the European Social Survey and the European Values Study."

    questionnaire, biterms = load_files(questionnaire_path, biterms_path)

    #context0 = no stimuli
    context0 = """Please translate the following questionnaire written in English (from Great Britain) to German (from Germany). Your translated text should read as if it were written by a native German speaker from Germany. Use a direct and clear communication style. Use a professional, formal, and neutral tone suitable for survey questions. Replace any instances of the token "[country]" with the appropriate country in your output. Your translation output must have the same number of lines as the source (English) questionnaire. Include only the translated questionnaire in your output."""

    #context1 = biterms stimuli
    context1 = """Please translate the following questionnaire written in English (from Great Britain) to German (from Germany). After the questionnaire, you can find a list of English terms and their translations to German extracted from the European Social Survey and European Values Study questionnaires. This list may contain useful survey-specific terminology translation examples for you.
Your translated text should read as if it were written by a native German speaker from Germany. Use a direct and clear communication style. Use a professional, formal, and neutral tone suitable for survey questions. Replace any instances of the token "[country]" with the appropriate country in your output. Your translation output must have the same number of lines as the source (English) questionnaire. Include only the translated questionnaire in your output."""
    
    #context2 = biterms + country-specific expressions and terminology stimuli
    context2 = """Please translate the following questionnaire written in English (from Great Britain) to German (from Germany). After the questionnaire, you can find a list of English terms and their translations to German extracted from the European Social Survey and European Values Study questionnaires. This list may contain useful survey-specific terminology translation examples for you. 
    Use expressions and terminology that are specific from Germany to ensure proper cultural adaptation when applicable, especially when translating terms related to education levels and job descriptions/titles. For instance, the "Qualification from vocational ISCED 2C programmes of 2 years or longer duration, no access to ISCED 3" education level would be equivalent to "Volks-/Hauptschulabschluss bzw. Polytechnische Oberschule mit Abschluss 8. oder 9. Klasse" in Germany; the job description "in education, (not paid for by employer) even if on vacation" would be equivalent to "Schule/Ausbildung (nicht vom Arbeitgeber bezahlt; auch während der Ferien oder im Urlaub)" in Germany.
Your translated text should read as if it were written by a native German speaker from Germany. Use a direct and clear communication style. Use a professional, formal, and neutral tone suitable for survey questions. Replace any instances of the token "[country]" with the appropriate country in your output. Your translation output must have the same number of lines as the source (English) questionnaire. Include only the translated questionnaire in your output."""
    
    #context3 = biterms + country-specific expressions and terminology + scales stimuli
    context3 = """Please translate the following questionnaire written in English (from Great Britain) to German (from Germany). After the questionnaire, you can find a list of English terms and their translations to German extracted from the European Social Survey and European Values Study questionnaires. This list may contain useful survey-specific terminology translation examples for you. 
    Use expressions and terminology that are specific from Germany to ensure proper cultural adaptation when applicable, especially when translating terms related to education levels and job descriptions/titles. For instance, the "Qualification from vocational ISCED 2C programmes of 2 years or longer duration, no access to ISCED 3" education level would be equivalent to "Volks-/Hauptschulabschluss bzw. Polytechnische Oberschule mit Abschluss 8. oder 9. Klasse" in Germany; the job description "in education, (not paid for by employer) even if on vacation" would be equivalent to "Schule/Ausbildung (nicht vom Arbeitgeber bezahlt; auch während der Ferien oder im Urlaub)" in Germany. When translating response scales, focus on keeping the concepts of interest and intensity of qualitifies/intensifiers the same across languages (e.g., "Agree strongly" -> "Stimme voll und ganz zu", "Very dissatisfied" -> "Sehr unzufrieden"). Do not forget to correctly translate and add to your output important survey-specific elements such as instructions to the respondent/interviewer like "READ OUT" and "Code all mentioned".
Your translated text should read as if it were written by a native German speaker from Germany. Use a direct and clear communication style. Use a professional, formal, and neutral tone suitable for survey questions. Replace any instances of the token "[country]" with the appropriate country in your output. Your translation output must have the same number of lines as the source (English) questionnaire. Include only the translated questionnaire in your output."""


    #context4 = biterms + country-specific expressions and terminology + scales stimuli + examples
    context4 = """Please translate the following questionnaire written in English (from Great Britain) to German (from Germany). After the questionnaire, you can find a list of English terms and their translations to German extracted from the European Social Survey and European Values Study questionnaires. This list may contain useful survey-specific terminology translation examples for you. 
    Use expressions and terminology that are specific from Germany to ensure proper cultural adaptation when applicable, especially when translating terms related to education levels and job descriptions/titles. For instance, the "Qualification from vocational ISCED 2C programmes of 2 years or longer duration, no access to ISCED 3" education level would be equivalent to "Volks-/Hauptschulabschluss bzw. Polytechnische Oberschule mit Abschluss 8. oder 9. Klasse" in Germany; the job description "in education, (not paid for by employer) even if on vacation" would be equivalent to "Schule/Ausbildung (nicht vom Arbeitgeber bezahlt; auch während der Ferien oder im Urlaub)" in Germany. When translating response scales, focus on keeping the concepts of interest and intensity of qualitifies/intensifiers the same across languages (e.g., "Agree strongly" -> "Stimme voll und ganz zu", "Very dissatisfied" -> "Sehr unzufrieden"). Do not forget to correctly translate and add to your output important survey-specific elements such as instructions to the respondent/interviewer like "READ OUT" and "Code all mentioned". Your translated text should read as if it were written by a native German speaker from Germany. Use a direct and clear communication style. Use a professional, formal, and neutral tone suitable for survey questions. Replace any instances of the token "[country]" with the appropriate country in your output. Your translation output must have the same number of lines as the source (English) questionnaire. Include only the translated questionnaire in your output. To further help you in your task, below you can find some examples of survey items extracted from past European Values Study and European Social Survey questionnaires translated from English (United Kingdom) to German (Germany), observe these examples and apply the same text style and terminology to your translations:

Example 1)
Source:
Generally speaking, would you say that most people can be trusted or that you can't be too careful in dealing with people?
Most people can be trusted
Can't be too careful
Don't know 
No answer

Translation:
Würden Sie ganz allgemein sagen, dass man den meisten Menschen vertrauen kann, oder dass man da nicht vorsichtig genug sein kann?
Man kann den meisten vertrauen
Man kann nicht vorsichtig genug sein
Weiß nicht
Verweigert

Example 2)
Source:
SHOW CARD 41 – READ OUT AND CODE ONE ANSWER PER LINE
How much do you agree or disagree with each of the following:
Religious leaders should not influence government decisions
agree strongly
agree
neither agree nor disagree
disagree 
disagree strongly
Don't know 
No answer

Translation:
LISTE 41 VORLEGEN - VORGABEN VORLESEN UND EINE ANTWORT PRO ZEILE ANKREUZEN
Sagen Sie mir bitte zu jeder der folgenden Aussagen, ob Sie ihr voll und ganz zustimmen, zustimmen, nicht zustimmen oder überhaupt nicht zustimmen.
Die Kirchenoberhäupter sollten nicht versuchen, Entscheidungen der Regierung zu beeinflussen
Stimme voll und ganz zu
Stimme zu 
Weder noch
Stimme nicht zu
Stimme überhaupt nicht zu
Weiß nicht
Verweigert

Example 3)
Source:
In the past 2 years, did the police in [country] approach you, stop you or make contact with you for any reason?
Yes
No
Don't know 

Translation:
In den letzten 2 Jahren, hat sich die Polizei in Deutschland aus irgendeinem Grund an Sie gewendet, Sie angehalten oder kontaktiert?
Ja 
Nein
Weiß nicht

Example 4)
Source:
Now some questions about whether or not the police in [country] treat victims of crime equally. 
Please answer based on what you have heard or your own experience.
When victims report crimes, do you think the police treat rich people worse, poor people worse, or are rich and poor treated equally? 
Choose your answer from this card.
Rich people treated worse
Poor people treated worse
Rich and poor treated equally
Don't know

Translation:
Nun einige Fragen dazu, ob die Polizei in Deutschland alle Opfer von Straftaten gleich behandelt oder nicht.
Bitte denken Sie bei Ihrer Antwort an das, was Sie gehört oder selbsterlebt haben. 
Wenn Opfer zur Polizei gehen, um eine Straftat zu melden, glauben Sie, dass die Polizei reiche Leute schlechter behandelt, arme Leute schlechter behandelt oder dass beide gleich behandelt werden? 
Wählen Sie Ihre Antwort aus Liste 29 aus.
Reiche Leute werden schlechter behandelt
Arme Leute werden schlechter behandelt
Reiche und arme Leute werden gleich behandelt
Weiß nicht
"""

    context_list = [context0, context1, context2, context3, context4]


    model_temperature = [0, 0.15, 0.3, 0.45, 0.6, 0.75]
    

    for temperature in model_temperature:
        for context in context_list:
            if context==context0:
                prompt = create_prompt_without_biterms(context, questionnaire)
            else:
                prompt = create_prompt(context, questionnaire, biterms)

            print(prompt)

            # Translate questionnaire
            translated_output = translate_questionnaire(persona, prompt, temperature)

            # Save output
            if context==context0:
                out_file = output_filename+"_prompt0_"+"model_temperature_"+str(temperature)+"_batch_"+batch_number+".txt"
            elif context==context1:
                out_file = output_filename+"_prompt1_"+"model_temperature_"+str(temperature)+"_batch_"+batch_number+".txt"
            elif context==context2:
                out_file = output_filename+"_prompt2_"+"model_temperature_"+str(temperature)+"_batch_"+batch_number+".txt"
            elif context==context3:
                out_file = output_filename+"_prompt3_"+"model_temperature_"+str(temperature)+"_batch_"+batch_number+".txt"
            elif context==context4:
                out_file = output_filename+"_prompt4_"+"model_temperature_"+str(temperature)+"_batch_"+batch_number+".txt"
            
            save_output(translated_output, out_file)
            print(f"Translated questionnaire saved to {out_file}")
           

if __name__ == "__main__":
    batch_number = str(sys.argv[1])
    main(batch_number)
