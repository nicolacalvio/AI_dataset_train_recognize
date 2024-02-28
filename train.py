import math
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import pytorch_lightning as pl
from huggingface_hub import HfApi, HfFolder, Repository, notebook_login
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTForImageClassification
import json
import torch

class Classifier(pl.LightningModule):
    def __init__(self, model, lr: float = 2e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.model = model
        self.forward = self.model.forward
        self.val_acc = Accuracy(
            task='multiclass' if model.config.num_labels > 2 else 'binary',
            num_classes=model.config.num_labels
        )
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"train_loss", outputs.loss)
        return outputs.loss
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"val_loss", outputs.loss)
        acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
        self.log(f"val_acc", acc, prog_bar=True)
        return outputs.loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
if __name__ == '__main__':
  # Assuming your JSON data is stored in a file named 'data.json'
  with open('drive/MyDrive/ALL_AIRPLANES.json', 'r') as file:
        aircrafts = json.load(file)
  # Extract the "Model" from each entry
  models = [aircraft["Model"] for aircraft in aircrafts]
  search_terms = sorted(models)
  search_terms = [x for x in search_terms if x.strip() != '']
  print(len(search_terms))
  data_dir = Path('images')
  #get_image_urls_with_selenium(search_terms)
  ds = ImageFolder(data_dir)
  indices = torch.randperm(len(ds)).tolist()
  n_val = math.floor(len(indices) * .15)
  train_ds = torch.utils.data.Subset(ds, indices[:-n_val])
  val_ds = torch.utils.data.Subset(ds, indices[-n_val:])
  plt.figure(figsize=(20,10))
  num_examples_per_class = 5
  i = 1
  for class_idx, class_name in enumerate(ds.classes):
      folder = ds.root / class_name
      for image_idx, image_path in enumerate(sorted(folder.glob('*'))):
          if image_path.suffix in ds.extensions:
              image = Image.open(image_path)
              plt.subplot(len(ds.classes), num_examples_per_class, i)
              ax = plt.gca()
              ax.set_title(
                  class_name,
                  size='xx-large',
                  pad=5,
                  loc='left',
                  y=0,
                  backgroundcolor='white'
              )
              ax.axis('off')
              plt.imshow(image)
              i += 1
              if image_idx + 1 == num_examples_per_class:
                  break
  label2id = {}
  id2label = {}
  for i, class_name in enumerate(ds.classes):
      label2id[class_name] = str(i)
      id2label[str(i)] = class_name

  # Specificare il percorso del file
file_path = "class_mappings.json"
import json

# Preparare i dati per la scrittura in JSON
data_to_save = {
    "label2id": label2id,
    "id2label": id2label
}
# Scrivere i dati nel file JSON
with open(file_path, "w") as json_file:
    json.dump(data_to_save, json_file, indent=4)


class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)
collator = ImageClassificationCollator(feature_extractor)
train_loader = DataLoader(train_ds, batch_size=8, collate_fn=collator, num_workers=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collator, num_workers=2)

pl.seed_everything(42)
classifier = Classifier(model, lr=2e-5)
trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16, max_epochs=4)
trainer.fit(classifier, train_loader, val_loader)
val_batch = next(iter(val_loader))
outputs = model(**val_batch)
print('Preds: ', outputs.logits.softmax(1).argmax(1))
print('Labels:', val_batch['labels'])

torch.save(classifier.model.state_dict(), "model_weights.pth")
