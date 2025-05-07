import pandas as pd

def normalize_svtype(svtype):
    if 'DEL' in svtype.upper():
        return 'DEL'
    elif 'INS' in svtype.upper():
        return 'INS'
    else:
        return 'UNKNOWN'

def overlap_ratio(start1, end1, start2, end2):
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = max(0, overlap_end - overlap_start + 1)
    total_length = max(end1, end2) - min(start1, start2) + 1
    return overlap_length / total_length

def calculate_metrics1(predicted_bed, true_bed, overlap_threshold=0.5):

    predicted = pd.read_csv(predicted_bed, sep='\t', header=None, names=['chrom', 'start', 'end', 'svtype'])
    true = pd.read_csv(true_bed, sep='\t', header=None, names=['chrom', 'start', 'end', 'svtype'])


    true_positive = 0
    false_positive = len(predicted)
    false_negative = len(true)


    matched_true_indices = set()

    for _, pred_row in predicted.iterrows():
        for true_index, true_row in true.iterrows():

            if pred_row['chrom'] == true_row['chrom']:

                if pred_row['svtype'] == true_row['svtype']:
                    overlap = overlap_ratio(pred_row['start'], pred_row['end'], true_row['start'], true_row['end'])
                    if overlap >= overlap_threshold:
                        true_positive += 1
                        matched_true_indices.add(true_index)
                        false_positive -= 1
                        false_negative -= 1
                        break

    false_negative += len(true) - len(matched_true_indices)


    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    #recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    recall = len(matched_true_indices) / len(true)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

def main():
    predicted_bed = '/work/0.125/svim.bed'
    true_bed = '/work/0.125/merge.bed'

    print("Calculating metrics based on  predictions...")
    calculate_metrics1(predicted_bed, true_bed)

if __name__ == "__main__":
    main()
