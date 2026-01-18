from datasets import load_dataset
import csv

# load từ HF
ds = load_dataset("tranthaihoa/vifactcheck")["test"]

out_file = "vifactcheck_claim_checkthat.tsv"

with open(out_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["topic", "tweet_id", "tweet_url", "tweet_text", "class_label"])
    
    for x in ds:
        writer.writerow([
            x["Topic"],
            x["index"],
            x.get("Url", ""),
            x["Statement"],
            1
        ])

print("Done →", out_file)
