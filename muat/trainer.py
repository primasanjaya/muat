import os
import glob
import json
import shutil
import zipfile
import logging

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from muat.util import *

logger = logging.getLogger(__name__)


class TrainerConfig:
    def __init__(
        self,
        max_epochs=10,
        batch_size=4,
        learning_rate=3e-4,
        betas=(0.9, 0.95),
        grad_norm_clip=1.0,
        weight_decay=0.001,
        lr_decay=False,
        show_loss_interval=10,
        save_ckpt_dir=None,
        string_logs=None,
        num_workers=0,
        ckpt_name="model",
        args=None,
        target_handler=None,
    ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.grad_norm_clip = grad_norm_clip
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.show_loss_interval = show_loss_interval
        self.save_ckpt_dir = save_ckpt_dir
        self.string_logs = string_logs
        self.num_workers = num_workers
        self.ckpt_name = ckpt_name
        self.args = args
        self.target_handler = target_handler if target_handler is not None else []


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.global_acc = 0
        self.pd_logits = []

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        self.complete_save_dir = self.config.save_ckpt_dir

    def batch_train(self):
        model = self.model
        model = model.to(self.device)

        if self.config.save_ckpt_dir is None:
            raise ValueError("config.save_ckpt_dir must be set before training.")

        os.makedirs(self.config.save_ckpt_dir, exist_ok=True)

        checkpoint_dir = os.path.join(self.complete_save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        evaluation_path = os.path.join(self.complete_save_dir, "evaluation.tsv")
        if not os.path.exists(evaluation_path):
            with open(evaluation_path, "w") as f:
                f.write(
                    "epoch\ttrain_loss\ttrain_accuracy\tvalidation_loss\tvalidation_accuracy\n"
                )
                f.flush()
                os.fsync(f.fileno())

        model = torch.nn.DataParallel(model).to(self.device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=0.9,
            weight_decay=self.config.weight_decay,
        )

        trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        valloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        self.global_acc = 0
        self.save_checkpoint_v3(self.config.save_ckpt_dir)

        for e in range(self.config.max_epochs):
            running_loss = 0.0
            model.train(True)

            train_corr = []
            logit_keys = []

            for batch_idx, (data, target, sample_path) in enumerate(trainloader):
                string_data = target
                numeric_data = data.to(self.device)

                class_values = []
                for key in string_data.keys():
                    values = string_data[key]
                    if not isinstance(values, list):
                        class_values.append(values)

                if len(class_values) > 1:
                    target = torch.stack(class_values, dim=0)
                elif len(class_values) == 1:
                    target = class_values[0].unsqueeze(dim=0)
                else:
                    raise ValueError("No valid target tensors found in training batch.")

                target = target.to(self.device)

                optimizer.zero_grad()
                logits, loss = model(numeric_data, target)

                if isinstance(logits, dict):
                    logit_keys = [x for x in logits.keys() if "logits" in x]
                    train_corr_inside = []

                    for nk, lk in enumerate(logit_keys):
                        logit = logits[lk]
                        pred = logit.argmax(dim=1, keepdim=True)
                        target_on_device = target[nk].to(pred.device)
                        train_corr_inside.append(
                            pred.eq(target_on_device.view_as(pred)).sum().item()
                        )

                    if len(train_corr) == 0:
                        train_corr = np.zeros(len(logit_keys))
                    train_corr += np.asarray(train_corr_inside)
                else:
                    raise ValueError("Model output logits must be a dictionary.")

                loss.backward()

                if self.config.grad_norm_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.grad_norm_clip,
                    )

                optimizer.step()
                running_loss += loss.item()
                train_acc = train_corr / len(self.train_dataset)

                if batch_idx % self.config.show_loss_interval == 0:
                    show_text = (
                        f"Epoch {e + 1} - Batch ({batch_idx}/{len(trainloader)}) "
                        f"- Mini-batch Training loss: {running_loss / (batch_idx + 1):.4f}"
                    )
                    for x in range(len(logit_keys)):
                        show_text += f" - Training Acc {x + 1}: {train_acc[x]:.4f}"
                    print(show_text)

            running_loss /= (batch_idx + 1)
            train_acc = train_corr / len(self.train_dataset)

            show_text = f"Epoch {e + 1} - Full-batch Training loss: {running_loss:.4f}"
            for x in range(len(logit_keys)):
                show_text += f" - Training Acc {x + 1}: {train_acc[x]:.4f}"
            print(show_text)

            test_loss = 0.0
            test_correct = []

            model.train(False)
            for batch_idx_val, (data, target, sample_path) in enumerate(valloader):
                string_data = target
                numeric_data = data.to(self.device)

                class_values = []
                for key in string_data.keys():
                    values = string_data[key]
                    if not isinstance(values, list):
                        class_values.append(values)

                if len(class_values) > 1:
                    target = torch.stack(class_values, dim=0)
                elif len(class_values) == 1:
                    target = class_values[0].unsqueeze(dim=0)
                else:
                    raise ValueError("No valid target tensors found in validation batch.")

                target = target.to(self.device)

                with torch.no_grad():
                    logits, loss = model(numeric_data, target)
                    test_loss += loss.item()

                    if isinstance(logits, dict):
                        logit_keys = [x for x in logits.keys() if "logits" in x]
                        test_correct_inside = []

                        for nk, lk in enumerate(logit_keys):
                            logit = logits[lk]
                            predicted = logit.argmax(dim=1)

                            logits_cpu = logit.detach().cpu().numpy()
                            logit_filename = f"val_{lk}.tsv"
                            logit_filepath = os.path.join(self.complete_save_dir, logit_filename)

                            if batch_idx_val == 0:
                                with open(logit_filepath, "w+") as f:
                                    target_handler = self.config.target_handler[nk]
                                    header_class = target_handler.classes_
                                    write_header = "\t".join(header_class)
                                    f.write(write_header)
                                    f.write("\ttarget_name\tsample")

                            with open(logit_filepath, "a+") as f:
                                for i_b in range(len(sample_path)):
                                    f.write("\n")
                                    logits_cpu_flat = logits_cpu[i_b].flatten()
                                    logits_cpu_list = logits_cpu_flat.tolist()
                                    write_logits = [f"{i:.8f}" for i in logits_cpu_list]

                                    target_handler = self.config.target_handler[nk]
                                    target_name = target_handler.inverse_transform(
                                        [target[nk].detach().cpu().numpy().tolist()[i_b]]
                                    )[0]

                                    write_logits.append(str(target_name))
                                    write_logits.append(sample_path[i_b])
                                    write_header = "\t".join(write_logits)
                                    f.write(write_header)

                            target_on_device = target[nk].to(predicted.device)
                            test_correct_inside.append(
                                predicted.eq(target_on_device.view_as(predicted)).sum().item()
                            )

                        if len(test_correct) == 0:
                            test_correct = np.zeros(len(logit_keys))
                        test_correct += np.asarray(test_correct_inside)
                    else:
                        raise ValueError("Model output logits must be a dictionary.")

            test_loss /= (batch_idx_val + 1)
            test_acc = test_correct[0] / len(self.test_dataset)

            print(
                "Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    test_correct[0],
                    len(self.test_dataset),
                    100.0 * test_acc,
                )
            )

            with open(evaluation_path, "a") as f:
                f.write(
                    f"{e + 1}\t{running_loss:.8f}\t{train_acc[0]:.8f}\t{test_loss:.8f}\t{test_acc:.8f}\n"
                )
                f.flush()
                os.fsync(f.fileno())

            self.save_checkpoint_v3(self.config.save_ckpt_dir)
            self.save_checkpoint_v3(os.path.join(checkpoint_dir, f"epoch_{e}"))

            if test_acc > self.global_acc:
                self.global_acc = test_acc
                print(self.global_acc)

                for nk, lk in enumerate(logit_keys):
                    logit_filename = f"val_{lk}.tsv"
                    src = os.path.join(self.complete_save_dir, logit_filename)
                    dst = os.path.join(self.complete_save_dir, f"best_{logit_filename}")
                    shutil.copyfile(src, dst)
                    os.remove(src)

                ckpt_path = os.path.join(
                    self.config.save_ckpt_dir,
                    self.config.ckpt_name + ".pthx"
                )
                best_ckpt_path = os.path.join(
                    self.config.save_ckpt_dir,
                    "best_ckpt.pthx"
                )
                shutil.copyfile(ckpt_path, best_ckpt_path)

    def make_json_serializable(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif hasattr(obj, "__class__") and obj.__class__.__name__ == "LabelEncoderFromCSV":
            return {
                "class_to_idx": obj.class_to_idx,
                "idx_to_class": obj.idx_to_class,
                "classes_": list(obj.classes_),
            }
        else:
            return obj

    def recursive_serialize(self, obj):
        if isinstance(obj, dict):
            return {k: self.recursive_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.recursive_serialize(v) for v in obj]
        else:
            return self.make_json_serializable(obj)

    def config_to_dict(self, config):
        result = {}
        for name in dir(config):
            if name.startswith("_"):
                continue
            value = getattr(config, name)
            if callable(value):
                continue
            result[name] = value
        return result

    def save_model_config_to_json(self, config, filepath: str):
        serialisable_dict = self.recursive_serialize(self.config_to_dict(config))

        for key in ("save_ckpt_dir", "save_ckpt_path", "target_handler", "args"):
            serialisable_dict.pop(key, None)

        with open(filepath, "w") as f:
            json.dump(serialisable_dict, f, indent=2)

    def save_dict_to_json(self, data, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.recursive_serialize(data), f, indent=2)

    def save_dataframe_to_json(self, df, filepath: str):
        data = df.to_dict(orient="records")
        self.save_dict_to_json(data, filepath)

    def save_checkpoint_v3(self, save_dir: str = None):
        if save_dir is None:
            save_dir = self.config.save_ckpt_dir
        if save_dir is None:
            raise ValueError("No save directory specified. Set config.save_ckpt_dir or provide save_dir.")

        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "weight": self.model.state_dict(),
            "target_handler": self.config.target_handler,
            "model_config": self.model.config,
            "trainer_config": self.config,
            "dataloader_config": self.train_dataset.config,
            "model_name": self.model.__class__.__name__,
            "motif_dict": self.model.config.dict_motif,
            "pos_dict": self.model.config.dict_pos,
            "ges_dict": self.model.config.dict_ges,
            "metadata": {
                "checkpoint_version": 3,
                "format": "muat_v3_zip",
                "model_name": self.model.__class__.__name__,
                "dictionary_schema_version": 1,
            },
        }

        weights_path = os.path.join(save_dir, "weight.pth")
        torch.save(checkpoint["weight"], weights_path)

        self.save_dict_to_json(
            checkpoint["metadata"],
            os.path.join(save_dir, "metadata.json")
        )

        for idx, handler in enumerate(checkpoint["target_handler"]):
            filepath = os.path.join(save_dir, f"target_handler_{idx + 1}.json")
            self.save_dict_to_json(
                {
                    "class_to_idx": handler.class_to_idx,
                    "idx_to_class": handler.idx_to_class,
                    "classes_": list(handler.classes_),
                },
                filepath,
            )

        configs = {
            "model_config": checkpoint["model_config"],
            "trainer_config": checkpoint["trainer_config"],
            "dataloader_config": checkpoint["dataloader_config"],
        }

        for name, config in configs.items():
            filepath = os.path.join(save_dir, f"{name}.json")
            self.save_model_config_to_json(config, filepath)

        self.save_dict_to_json(
            checkpoint["model_name"],
            os.path.join(save_dir, "model_name.json"),
        )

        dicts = {
            "motif_dict": checkpoint["motif_dict"],
            "pos_dict": checkpoint["pos_dict"],
            "ges_dict": checkpoint["ges_dict"],
        }

        for name, df in dicts.items():
            filepath = os.path.join(save_dir, f"{name}.json")
            self.save_dataframe_to_json(df, filepath)

        zip_name = self.config.ckpt_name + ".pthx"
        zip_path = os.path.join(save_dir, zip_name)

        files_to_zip = []
        for ext in [".json", ".pth"]:
            files_to_zip.extend(glob.glob(os.path.join(save_dir, f"*{ext}")))

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for file in files_to_zip:
                zipf.write(file, os.path.basename(file))

        for file in files_to_zip:
            os.remove(file)

        logger.info(f"Checkpoint saved to {zip_path}")
        return zip_path