import torch
from transformers import AutoProcessor,LlavaForConditionalGeneration,BitsAndBytesConfig
from peft import PeftModel
import json
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

BASE_MODEL_PATH="/blue/bme6938/saririans/RetinAI_results/retinai-llava-v1-idrid-test"
ADAPTER_PATH="/blue/bme6938/saririans/RetinAI_results/retinai-llava-med-v1-aptos"
DATA_JSONL="/home/saririans/RetinAI/data/processed/aptos_train_conversations.jsonl"
BATCH_SIZE=16

text_to_grade={"no diabetic retinopathy":0,"mild diabetic retinopathy":1,"moderate diabetic retinopathy":2,"severe diabetic retinopathy":3,"proliferative diabetic retinopathy":4}

class AptosEvalDataset(Dataset):
    def __init__(self,json_file,processor):
        self.data=[json.loads(line) for line in open(json_file)]
        self.processor=processor
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        item=self.data[idx]
        image_path=item['image']
        prompt="<image>\nDescribe this retinal image.\nThis is"
        try:
            image=Image.open(image_path).convert("RGB")
            return {"image":image,"text":prompt,"gt":item['conversations'][1]['value']}
        except:
            return None

def collate_fn(batch):
    batch=[b for b in batch if b is not None]
    if len(batch)==0: return None
    images=[b['image'] for b in batch]
    texts=[b['text'] for b in batch]
    gts=[b['gt'] for b in batch]
    inputs=processor(text=texts,images=images,return_tensors="pt",padding=True)
    return inputs,gts

print("Loading Model...")
quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
processor=AutoProcessor.from_pretrained(BASE_MODEL_PATH)
processor.tokenizer.padding_side="left"
model=LlavaForConditionalGeneration.from_pretrained(BASE_MODEL_PATH,quantization_config=quantization_config,device_map="auto")
model=PeftModel.from_pretrained(model,ADAPTER_PATH)
model.eval()

dataset=AptosEvalDataset(DATA_JSONL,processor)
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,collate_fn=collate_fn,num_workers=4)

y_true=[]
y_pred=[]
valid_count=0

print(f"Starting batched inference on {len(dataset)} images...")

for batch_data in tqdm(dataloader):
    if batch_data is None: continue
    inputs,gts=batch_data
    inputs={k:v.to("cuda") for k,v in inputs.items()}
    with torch.no_grad():
        generate_ids=model.generate(**inputs,max_new_tokens=32)
    generated_ids=generate_ids[:,inputs['input_ids'].shape[1]:]
    outputs=processor.batch_decode(generated_ids,skip_special_tokens=True)
    
    for pred_tail,gt_text in zip(outputs,gts):
        gt_text=gt_text.lower()
        true_label=-1
        for key,val in text_to_grade.items():
            if key in gt_text:
                true_label=val
                break
        if true_label==-1: continue
        
        response="this is "+pred_tail.lower().strip()
        pred_label=-1
        for class_name,label_idx in text_to_grade.items():
            if class_name in response:
                pred_label=label_idx
                break
        if pred_label==-1:
            if "no diabetic" in response: pred_label=0
            elif "mild" in response: pred_label=1
            elif "moderate" in response: pred_label=2
            elif "severe" in response: pred_label=3
            elif "proliferative" in response: pred_label=4
            
        if pred_label!=-1:
            valid_count+=1
            y_pred.append(pred_label)
            y_true.append(true_label)
        else:
            y_pred.append(-1)
            y_true.append(true_label)

clean_y_true=[t for t,p in zip(y_true,y_pred) if p!=-1]
clean_y_pred=[p for p in y_pred if p!=-1]

if len(clean_y_pred)>0:
    print(f"Valid Predictions: {len(clean_y_pred)}/{len(y_true)}")
    print(f"Accuracy: {accuracy_score(clean_y_true,clean_y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(clean_y_true,clean_y_pred,labels=list(text_to_grade.values()),target_names=list(text_to_grade.keys()),zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(clean_y_true,clean_y_pred))
else:
    print("No valid predictions found.")