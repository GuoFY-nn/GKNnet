import os


def vcf_to_bed(vcf_file, bed_file):

    os.makedirs(os.path.dirname(bed_file), exist_ok=True)

    with open(vcf_file, 'r') as vcf, open(bed_file, 'w') as bed:
        for line in vcf:

            if line.startswith('#'):
                continue

            columns = line.strip().split('\t')
            chrom = columns[0]
            pos = int(columns[1]) - 1
            x = columns[7]

            pairs = x.split(';')

            result = {}

            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=')
                    result[key] = value

            svtype = result.get('SVTYPE')
            if svtype == 'DEL':
                svlen = abs(int(result.get('SVLEN')))
                start = pos
                end = start + svlen

                bed.write(f"{chrom}\t{start}\t{end}\t{svtype}\n")

            # svlen_str = result.get('SVLEN', '0')
            # if svlen_str == '.' or not svlen_str.isdigit():
            #     continue

            # svlen = abs(int(svlen_str))
            # svlen = int(columns[2])
            # svtype = columns[3]
            # if svtype == 'INS':
            #     start = pos
            #     end = start + svlen

            #     bed.write(f"{chrom}\t{start}\t{end}\t{svtype}\n")

    print(f"Converted {vcf_file} to {bed_file} (DEL only)")



vcf_to_bed('/work/0.125/SVIM_merge.vcf', '/work/0.125/svim.bed')
