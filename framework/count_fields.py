from os import listdir
from os import walk, path

folder='TestDatasets_lowD'
#folder='low_dim'
top_file = []
for (dirpath, dirnames, filenames) in walk(folder):
	for f in filenames:
		if f.startswith('.'):
			continue
		if not f.endswith('.csv'):
			continue
		#print(dirpath, f)
		file = path.join(dirpath, f)
		with open(file) as tfile:
			for line in tfile:
				ct = len(line.split(','))
				top_file.append({'count': ct, 'file': file})
				break # read only the first line

sorted_by_value = sorted(top_file, key=lambda kv: kv['count'])
print(sorted_by_value[-3:])

