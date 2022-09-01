import json

import numpy as np
import torch


def compute_depth_metrics(gt, pred, mult_a=False):
    """
    Computes error metrics between predicted and ground truth depths
    """

    thresh = torch.max((gt / pred), (pred / gt))
    a_dict = {}
    a_dict["a5"] = (thresh < 1.05     ).float().mean()
    a_dict["a10"] = (thresh < 1.10     ).float().mean()
    a_dict["a25"] = (thresh < 1.25     ).float().mean()

    a_dict["a0"] = (thresh < 1.10     ).float().mean()
    a_dict["a1"] = (thresh < 1.25     ).float().mean()
    a_dict["a2"] = (thresh < 1.25 ** 2).float().mean()
    a_dict["a3"] = (thresh < 1.25 ** 3).float().mean()


    if mult_a:
        for key in a_dict:
            a_dict[key] = a_dict[key]*100

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    abs_diff = torch.mean(torch.abs(gt - pred))

    metrics_dict = {
                    "abs_diff": abs_diff,
                    "abs_rel": abs_rel,
                    "sq_rel": sq_rel,
                    "rmse": rmse,
                    "rmse_log": rmse_log,
                }
    metrics_dict.update(a_dict)

    return metrics_dict

def compute_depth_metrics_batched(gt_bN, pred_bN, valid_masks_bN, mult_a=False):
    """
    Computes error metrics between predicted and ground truth depths, 
    batched. Abuses nan behavior in torch.
    """

    gt_bN = gt_bN.clone()
    pred_bN = pred_bN.clone()

    gt_bN[~valid_masks_bN] = torch.nan
    pred_bN[~valid_masks_bN] = torch.nan

    thresh_bN = torch.max(torch.stack([(gt_bN / pred_bN), (pred_bN / gt_bN)], 
                                                            dim=2), dim=2)[0]
    a_dict = {}
    
    a_val = (thresh_bN < (1.0+0.05)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a5"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.10)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a10"] = torch.nanmean(a_val, dim=1) 

    a_val = (thresh_bN < (1.0+0.25)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a25"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.10)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a0"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a1"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25) ** 2).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a2"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25) ** 3).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a3"] = torch.nanmean(a_val, dim=1)

    if mult_a:
        for key in a_dict:
            a_dict[key] = a_dict[key]*100

    rmse_bN = (gt_bN - pred_bN) ** 2
    rmse_b = torch.sqrt(torch.nanmean(rmse_bN, dim=1))

    rmse_log_bN = (torch.log(gt_bN) - torch.log(pred_bN)) ** 2
    rmse_log_b = torch.sqrt(torch.nanmean(rmse_log_bN, dim=1))

    abs_rel_b = torch.nanmean(torch.abs(gt_bN - pred_bN) / gt_bN, dim=1)

    sq_rel_b = torch.nanmean((gt_bN - pred_bN) ** 2 / gt_bN, dim=1)

    abs_diff_b = torch.nanmean(torch.abs(gt_bN - pred_bN), dim=1)

    metrics_dict = {
                    "abs_diff": abs_diff_b,
                    "abs_rel": abs_rel_b,
                    "sq_rel": sq_rel_b,
                    "rmse": rmse_b,
                    "rmse_log": rmse_log_b,
                }
    metrics_dict.update(a_dict)

    return metrics_dict

class ResultsAverager():
    """ 
    Helper class for stable averaging of metrics across frames and scenes. 
    """
    def __init__(self, exp_name, metrics_name):
        """
            Args:
                exp_name: name of the specific experiment. 
                metrics_name: type of metrics.
        """
        self.exp_name = exp_name
        self.metrics_name = metrics_name

        self.elem_metrics_list = []
        self.running_metrics = None
        self.running_count = 0

        self.final_computed_average = None

    def update_results(self, elem_metrics):
        """
        Adds elem_matrix to elem_metrics_list. Updates running_metrics with 
        incomming metrics to keep a running average. 

        running_metrics are cheap to compute but not totally stable.
        """

        self.elem_metrics_list.append(elem_metrics.copy())

        if self.running_metrics is None:
            self.running_metrics = elem_metrics.copy()
        else:
            for key in list(elem_metrics.keys()):
                self.running_metrics[key] = (
                                                self.running_metrics[key] * 
                                                    self.running_count 
                                                + elem_metrics[key]
                                            ) / (self.running_count + 1)

        self.running_count += 1

    def print_sheets_friendly(
                        self, print_exp_name=True, 
                        include_metrics_names=False, 
                        print_running_metrics=True,
                    ):
        """
        Print for easy sheets copy/paste.
        Args:   
            print_exp_name: should we print the experiment name?
            include_metrics_names: should we print a row for metric names?
            print_running_metrics: should we print running metrics or the 
                final average?
        """

        if print_exp_name:
            print(f"{self.exp_name}, {self.metrics_name}")

        if print_running_metrics:
            metrics_to_print = self.running_metrics
        else:
            metrics_to_print = self.final_metrics 

        if len(self.elem_metrics_list) == 0:
            print("WARNING: No valid metrics to print.")
            return

        metric_names_row = ""
        metrics_row = ""
        for k, v in metrics_to_print.items():
            metric_names_row += f"{k:8} "
            metric_string = f"{v:.4f},"
            metrics_row += f"{metric_string:8} "
        
        if include_metrics_names:
            print(metric_names_row)
        print(metrics_row)
        
    def output_json(self, filepath, print_running_metrics=False):
        """
        Outputs metrics to a json file.
        Args:   
            filepath: file path where we should save the file.
            print_running_metrics: should we print running metrics or the 
                final average?
        """
        scores_dict = {}
        scores_dict["exp_name"] = self.exp_name
        scores_dict["metrics_type"] = self.metrics_name
        
        scores_dict["scores"] = {}

        if print_running_metrics:
            metrics_to_use = self.running_metrics
        else:
            metrics_to_use = self.final_metrics 

        if len(self.elem_metrics_list) == 0:
            print("WARNING: No valid metrics will be output.")

        metric_names_row = ""
        metrics_row = ""
        for k, v in metrics_to_use.items():
            metric_names_row += f"{k:8} "
            metric_string = f"{v:.4f},"
            metrics_row += f"{metric_string:8} "
            scores_dict["scores"][k] = float(v)
        
        scores_dict["metrics_string"] = metric_names_row
        scores_dict["scores_string"] = metrics_row

        with open(filepath, "w") as file:
            json.dump(scores_dict, file, indent=4)

    def pretty_print_results(
                        self, 
                        print_exp_name=True, 
                        print_running_metrics=True
                    ):
        """
        Pretty print for easy(ier) reading
        Args:   
            print_exp_name: should we print the experiment name?
            include_metrics_names: should we print a row for metric names?
            print_running_metrics: should we print running metrics or the 
                final average?
        """
        if print_running_metrics:
            metrics_to_print = self.running_metrics
        else:
            metrics_to_print = self.final_metrics 

        if len(self.elem_metrics_list) == 0:
            print("WARNING: No valid metrics to print.")
            return

        if print_exp_name:
            print(f"{self.exp_name}, {self.metrics_name}")
        for k, v in metrics_to_print.items():
            print(f"{k:8}: {v:.4f}")

    def compute_final_average(self, ignore_nans=False):
        """
        Computes final a final average on the metrics element list using 
        numpy.
    
        This should be more accurate than running metrics as it's a single 
        average vs multiple high level multiplications and divisions.

        Args:
            ignore_nans: ignore nans in the results and run using nanmean.
        """

        self.final_metrics = {}

        if len(self.elem_metrics_list) == 0:
            print("WARNING: no valid entry to average!")
            return

        for key in list(self.running_metrics.keys()):
            values = []
            for element in self.elem_metrics_list:
                if torch.is_tensor(element[key]):
                    values.append(element[key].cpu().numpy())
                else:
                    values.append(element[key])

            if ignore_nans:
                mean_value = np.nanmean(np.array(values))
            else:
                mean_value = np.array(values).mean()
            self.final_metrics[key] = mean_value
