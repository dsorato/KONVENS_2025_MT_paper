import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def get_color(row):
    if row['is_human']:
        return '#0072B2'  
    elif row['is_mt']:
        return '#CC79A7'  

df = pd.read_csv("mapping.tsv", sep="\t")

# Prepare data for plotting
plot_data = []

for _, row in df.iterrows():
    question = str(row['question_number'])  # Ensure it's string for plotting
    answers = row[['participant_0', 'participant_1', 'participant_2', 'participant_3', 'participant_4']]
    count_A = (answers == 'A').sum()
    count_B = (answers == 'B').sum()
    
    majority_choice = 'A' if count_A > count_B else 'B' if count_B > count_A else 'Tie'

    plot_data.append({
        'question': question,
        'option': 'A',
        'votes': count_A,
        'is_human': row['human_translation'] == 'A',
        'is_mt': row['mt_translation'] == 'A',
        'is_majority': majority_choice == 'A'
    })

    plot_data.append({
        'question': question,
        'option': 'B',
        'votes': count_B,
        'is_human': row['human_translation'] == 'B',
        'is_mt': row['mt_translation'] == 'B',
        'is_majority': majority_choice == 'B'
    })


plot_df = pd.DataFrame(plot_data)

#positions for grouped bars
plot_df['question'] = plot_df['question'].astype(int)
questions = sorted(plot_df['question'].unique())
plot_df = plot_df.sort_values(by=['question', 'option'])

x = range(len(questions))
bar_width = 0.35


votes_A = plot_df[plot_df['option'] == 'A']['votes'].values
votes_B = plot_df[plot_df['option'] == 'B']['votes'].values

colors_A = plot_df[plot_df['option'] == 'A'].apply(get_color, axis=1).values
colors_B = plot_df[plot_df['option'] == 'B'].apply(get_color, axis=1).values


fig, ax = plt.subplots(figsize=(20, 4))
bars_A = ax.bar([i - bar_width/2 for i in x], votes_A, width=bar_width, color=colors_A, edgecolor='black')
bars_B = ax.bar([i + bar_width/2 for i in x], votes_B, width=bar_width, color=colors_B, edgecolor='black')

#indicate MT option with a star
for i, (row_A, row_B) in enumerate(zip(
        plot_df[plot_df['option'] == 'A'].itertuples(),
        plot_df[plot_df['option'] == 'B'].itertuples())):
    
    if row_A.is_mt:
        ax.text(i - bar_width/2, row_A.votes + 0.2, '★', ha='center', va='bottom', fontsize=14)
    if row_B.is_mt:
        ax.text(i + bar_width/2, row_B.votes + 0.2, '★', ha='center', va='bottom', fontsize=14)

ax.set_xlabel("Question Number", fontsize=14)
ax.set_ylabel("Number of Votes", fontsize=14)
ax.set_title("Answer distributions per question (★ = GPT translation)", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(questions, rotation=90)

plt.tight_layout()
plt.show()




