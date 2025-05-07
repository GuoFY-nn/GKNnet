import pandas as pd


input_file = 'work/extrat.bed'
output_file = 'work/merge.bed'


columns = ['chrom', 'start', 'end', 'label']
bed_df = pd.read_csv(input_file, sep='\t', header=None, names=columns)


merged_intervals = []


for index, row in bed_df.iterrows():
    if index == 0:

        merged_intervals.append(row)
    else:

        prev = merged_intervals[-1]
        current = row


        distance = current['start'] - prev['end']
        min_len = min(prev['end'] - prev['start'], current['end'] - current['start'])
        max_len = max(prev['end'] - prev['start'], current['end'] - current['start'])
        coverage_ratio = min_len / max_len


        if distance <= alpha and coverage_ratio > 0.7:
            merged_intervals[-1]['end'] = max(prev['end'], current['end'])
        else:
            merged_intervals.append(current)


merged_df = pd.DataFrame(merged_intervals)
merged_df.to_csv(output_file, sep='\t', header=False, index=False)


