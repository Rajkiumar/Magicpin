import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load env variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Import composition functions from bot.py
from bot import compose_message

def main():
    expanded_dir = Path("dataset/expanded")
    test_pairs_path = expanded_dir / "test_pairs.json"
    
    if not test_pairs_path.exists():
        print(f"Error: test_pairs.json not found at {test_pairs_path}. Please expand the dataset first.")
        return
        
    with open(test_pairs_path, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)["pairs"]
        
    print(f"Loaded {len(test_pairs)} test pairs.")
    
    submission_records = []
    
    for i, pair in enumerate(test_pairs):
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")
        
        print(f"[{i+1}/30] Composing for Test ID {test_id} (Merchant: {merchant_id}, Trigger: {trigger_id})...")
        
        # 1. Load Trigger
        trg_path = expanded_dir / "triggers" / f"{trigger_id}.json"
        with open(trg_path, "r", encoding="utf-8") as f:
            trigger = json.load(f)
            
        # 2. Load Merchant
        m_path = expanded_dir / "merchants" / f"{merchant_id}.json"
        with open(m_path, "r", encoding="utf-8") as f:
            merchant = json.load(f)
            
        # 3. Load Category
        cat_slug = merchant["category_slug"]
        cat_path = expanded_dir / "categories" / f"{cat_slug}.json"
        with open(cat_path, "r", encoding="utf-8") as f:
            category = json.load(f)
            
        # 4. Load Customer if present
        customer = None
        if customer_id:
            c_path = expanded_dir / "customers" / f"{customer_id}.json"
            with open(c_path, "r", encoding="utf-8") as f:
                customer = json.load(f)
                
        # Compose message using the bot logic
        res = compose_message(category, merchant, trigger, customer)
        
        # Build submission record
        record = {
            "test_id": test_id,
            "body": res.get("body", ""),
            "cta": res.get("cta", "none"),
            "send_as": res.get("send_as", "vera"),
            "suppression_key": trigger.get("suppression_key", f"{trigger_id}_suppress"),
            "rationale": res.get("rationale", "Composed with Category, Merchant, Trigger, and Customer contexts")
        }
        
        submission_records.append(record)
        print(f"    Body: {record['body'][:60]}...")
        print(f"    CTA: {record['cta']}")
        print(f"    Send As: {record['send_as']}")
        
        # Space out requests to stay below the 5 requests per minute limit
        if i < len(test_pairs) - 1:
            time.sleep(13)
        
    # Write to submission.jsonl
    submission_path = Path("submission.jsonl")
    with open(submission_path, "w", encoding="utf-8") as f:
        for r in submission_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\nSuccessfully generated submission.jsonl with {len(submission_records)} lines.")

if __name__ == "__main__":
    main()
