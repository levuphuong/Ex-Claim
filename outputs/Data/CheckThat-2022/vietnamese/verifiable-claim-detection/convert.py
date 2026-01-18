import re

def convert_to_tsv(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            line = line.rstrip()
            # Bỏ qua dòng trống
            if not line:
                continue
            
            # Nếu là header thì thay toàn bộ khoảng trắng bằng tab
            if line.startswith("topic"):
                outfile.write(re.sub(r"\s+", "\t", line) + "\n")
                continue
            
            # Với dữ liệu: topic, tweet_id, tweet_url, tweet_text, class_label
            # Tách 3 cột đầu bằng khoảng trắng
            parts = line.split()
            topic = parts[0]
            tweet_id = parts[1]
            tweet_url = parts[2]
            
            # Phần còn lại là tweet_text + class_label
            rest = " ".join(parts[3:])
            
            # Tách class_label (0 hoặc 1) ở cuối
            if rest.endswith(" 0") or rest.endswith(" 1"):
                tweet_text = rest[:-2].strip()
                class_label = rest[-1]
            else:
                # Nếu không có khoảng trắng trước class_label
                tweet_text = rest[:-1].strip()
                class_label = rest[-1]
            
            # Ghi ra file với tab phân cách
            outfile.write("\t".join([topic, tweet_id, tweet_url, tweet_text, class_label]) + "\n")

# Ví dụ sử dụng
convert_to_tsv("dev_test.tsv", "output.tsv")

