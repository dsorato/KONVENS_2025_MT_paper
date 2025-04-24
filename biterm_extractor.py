import os
import sys
print(os.path.abspath("."))
import pandas as pd
import numpy as np
from tm2tb import BitermExtractor
from tm2tb import BitextReader

def main(folder_path):
	files = os.listdir(folder_path)

	df_biterms=pd.DataFrame(columns=['source_term', 'target_term'])


	for index, file in enumerate(files):
		if file.endswith(".csv"):
			print('reading file:', file)
			df= pd.read_csv(folder_path+'/'+file, sep='\t')
			df['source_text'].replace('', np.nan, inplace=True)
			df.dropna(subset=['source_text'], inplace=True)
			df['target_text'].replace('', np.nan, inplace=True)
			df.dropna(subset=['target_text'], inplace=True)

			src=''
			tgt=''
			for i, row in df.iterrows():
				src=row['source_text'] + ". " + src
				tgt=row['target_text'] + ". " + tgt

				print(src, tgt)
				
			extractor = BitermExtractor(([src], [tgt]))
			biterms = extractor.extract_terms()


			for i, row in biterms.iterrows():
				if not ((df_biterms['source_term'] == row['src_term']) & (df_biterms['target_term'] == row['tgt_term'])).any():
					data = {'source_term': row['src_term'], 'target_term': row['tgt_term']}

					df_biterms = pd.concat([df_biterms, pd.DataFrame([data])], ignore_index=True)

					df_biterms.to_csv('biterms.csv', encoding='utf-8-sig', sep='\t', index=False)

			print(biterms)
			


if __name__ == "__main__":
	folder_path = str(sys.argv[1])
	main(folder_path)

