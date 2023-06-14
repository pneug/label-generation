from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.engine import HookBase
from detectron2.evaluation import DatasetEvaluators, inference_on_dataset
from detectron2.utils.events import EventStorage
from detectron2.data import build_detection_test_loader

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import DatasetCatalog, MetadataCatalog


def filter_annotations(dataset_dicts):
    filtered_dicts = []
    for d in dataset_dicts:
        if len(d["annotations"]) > 0:
            filtered_dicts.append(d)
    return filtered_dicts


# Modify the build_detection_test_loader function to use the filter_annotations function
def build_detection_test_loader_with_filter(cfg, dataset_name):
    dataset = DatasetCatalog.get(dataset_name)
    dataset_dicts = filter_annotations(dataset)
    return build_detection_test_loader(
        # cfg,
        dataset_dicts,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    mapper=DatasetMapper(cfg, is_train=False),
    # num_workers=num_workers,
    # batch_size=batch_size,
    )
        # cfg, dataset_dicts, dataset_name=dataset_name, mapper=None)


class EarlyStopping(HookBase):
    def __init__(self, cfg, model, val_evaluator, patience=3, eval_period=1):
        self.cfg = cfg.clone()
        self.model = model
        self.val_evaluator = val_evaluator
        # self.val_loader = build_detection_test_loader(
        #     self.cfg, self.cfg.DATASETS.VAL[0],
        #     batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
        # )
        self.val_loader = build_detection_test_loader_with_filter(
            self.cfg, self.cfg.DATASETS.VAL[0],
        )
        self.patience = patience
        self.eval_period = eval_period
        self.best_score = -1
        self.counter = 0
        self.stop_training = False

    def after_step(self):
        if self.trainer.iter % self.eval_period != 0:
            return
        
        print("after_step at step {}".format(self.trainer.iter))
        
        storage = self.trainer.storage
        
        # with EventStorage():  # capture events in a new storage to discard them later
        evaluators = DatasetEvaluators([self.val_evaluator])
        print("starting evaluation!!!")
        results = inference_on_dataset(self.model, self.val_loader, evaluators)
        print("Results: ", results)
        score = results["bbox"]["AP"]
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            checkpointer = PeriodicCheckpointer(
                self.trainer.checkpointer, self.eval_period
            )
            model_name = "model_best"
            checkpointer.save(model_name) # {"model": self.trainer.iter, }
        storage.put_scalar("val_best_score", self.best_score)
        # |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
        print("Putting scalar for val/AP: ", score, "at iter", self.trainer.iter, "storage at iter ", storage._iter)
        storage.put_scalar("val/AP", score)
        storage.put_scalar("val/AP50", results["bbox"]["AP50"])
        storage.put_scalar("val/AP75", results["bbox"]["AP75"])
        storage.put_scalar("val/APs", results["bbox"]["APs"])
        storage.put_scalar("val/APm", results["bbox"]["APm"])
        storage.put_scalar("val/APl", results["bbox"]["APl"])

        print("Early stopping counter: ", self.counter, "/", self.patience)
        if self.counter == self.patience:
            print("Early stopping triggered")
            self.trainer.storage.put_scalar("early_stop", True)
            self.stop_training = True

            # raise an exception to end training
            raise Exception("Early stopping")
            return
        
        self.counter += 1

