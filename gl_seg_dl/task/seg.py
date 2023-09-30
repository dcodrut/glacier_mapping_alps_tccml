import pytorch_lightning as pl
import torch
import torchmetrics as tm
import pandas as pd
import logging
import xarray as xr
from pathlib import Path

# local imports
from task import loss as losses


class GlSegTask(pl.LightningModule):
    def __init__(self, model, task_params, outdir=None):
        super().__init__()

        self.model = model
        self.loss = getattr(losses, task_params['loss']['name'])(**task_params['loss']['args'])
        self.val_metrics = tm.MetricCollection([
            tm.Accuracy(threshold=0.5, task='binary'),
            tm.JaccardIndex(threshold=0.5, task='binary'),
            tm.Precision(threshold=0.5, task='binary'),
            tm.Recall(threshold=0.5, task='binary'),
            tm.F1Score(threshold=0.5, task='binary')
        ])
        self.optimizer_settings = task_params['optimization']['optimizer']
        self.lr_scheduler_settings = task_params['optimization']['lr_schedule']

        # get the main logger
        self._logger = logging.getLogger('pytorch_lightning.core')

        self.outdir = outdir

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizers = [
            getattr(torch.optim, o['name'])(self.parameters(), **o['args'])
            for o in self.optimizer_settings
        ]
        schedulers = [
            getattr(torch.optim.lr_scheduler, s['name'])(optimizers[i], **s['args'])
            for i, s in enumerate(self.lr_scheduler_settings)
        ]
        return optimizers, schedulers

    def compute_masked_val_metrics(self, y_pred, y_true, mask):
        # apply the mask for each element in the batch and compute the metrics
        val_metrics_samplewise = []
        for i in range(len(y_true)):
            if mask[i].sum() > 0:
                m = self.val_metrics(preds=y_pred[i][mask[i]], target=y_true[i][mask[i]])
            else:
                m = {k: torch.nan for k in self.val_metrics}
            val_metrics_samplewise.append(m)

        # restructure the output in a single dict {metric: list}
        val_metrics_samplewise = {
            k: torch.tensor([x[k] for x in val_metrics_samplewise], device=y_true.device)
            for k in val_metrics_samplewise[0].keys()
        }

        return val_metrics_samplewise

    def aggregate_step_metrics(self, step_outputs, split_name):
        # extract the glacier name from the patch filepath
        all_fp = [fp for x in step_outputs for fp in x['filepaths']]
        df = pd.DataFrame({'fp': all_fp})
        df['gid'] = df.fp.apply(lambda s: Path(s).parent.parent.name)

        # add the metrics
        for m in step_outputs[0]['metrics']:
            df[m] = torch.stack([y for x in step_outputs for y in x['metrics'][m]]).cpu().numpy()

        # summarize the metrics per glacier
        avg_tb_logs = {}
        stats_per_g = df.groupby('gid').mean()
        for m in stats_per_g.columns:
            avg_tb_logs[f'{m}_{split_name}_epoch_avg_per_g'] = stats_per_g[m].mean()

        return avg_tb_logs, df

    def training_step(self, batch, batch_idx):
        y_pred = self(batch)
        y_true = batch['mask_all_g'].type(y_pred.dtype).unsqueeze(dim=1)
        mask = ~batch['mask_no_data'].unsqueeze(dim=1)
        loss = self.loss(preds=y_pred, targets=y_true, mask=mask, samplewise=True)

        # loss_cp = loss.clone().detach()
        tb_logs = {'train_loss': loss.mean()}
        self.log_dict(tb_logs, on_epoch=True, on_step=True, batch_size=len(y_true), sync_dist=True)

        # compute the evaluation metrics for each element in the batch
        val_metrics_samplewise = self.compute_masked_val_metrics(y_pred, y_true, mask)

        # compute the recall of the debris-covered areas
        y_true_debris = batch['mask_debris_crt_g'].type(y_pred.dtype).unsqueeze(dim=1)
        y_true_debris *= y_true  # in case the debris mask contains areas outside the current outlines
        area_debris_fraction = y_true_debris.flatten(start_dim=1).sum(dim=1) / y_true.flatten(start_dim=1).sum(dim=1)
        recall_samplewise_debris = self.compute_masked_val_metrics(y_pred, y_true_debris, mask)['BinaryRecall']
        recall_samplewise_debris[area_debris_fraction < 0.01] = torch.nan
        val_metrics_samplewise['BinaryRecall_debris'] = recall_samplewise_debris

        res = {'loss': loss.mean(), 'metrics': val_metrics_samplewise, 'filepaths': batch['fp']}

        return res

    def on_train_epoch_start(self):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2189
        print('\n')

    def training_epoch_end(self, train_step_outputs):
        avg_tb_logs, df = self.aggregate_step_metrics(train_step_outputs, split_name='train')

        # show the epoch as the x-coordinate
        avg_tb_logs['step'] = float(self.current_epoch)
        self.log_dict(avg_tb_logs, on_step=False, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        y_pred = self(batch)
        y_true = batch['mask_all_g'].type(y_pred.dtype).unsqueeze(dim=1)
        mask = ~batch['mask_no_data'].unsqueeze(dim=1)
        loss_samplewise = self.loss(preds=y_pred, targets=y_true, mask=mask, samplewise=True)

        # compute the evaluation metrics for each element in the batch
        val_metrics_samplewise = self.compute_masked_val_metrics(y_pred, y_true, mask)

        # compute the recall of the debris-covered areas (if the area is above 1%)
        y_true_debris = batch['mask_debris_crt_g'].type(y_pred.dtype).unsqueeze(dim=1)
        y_true_debris *= y_true  # in case the debris mask contains areas outside the current outlines
        area_debris_fraction = y_true_debris.flatten(start_dim=1).sum(dim=1) / y_true.flatten(start_dim=1).sum(dim=1)
        recall_samplewise_debris = self.compute_masked_val_metrics(y_pred, y_true_debris, mask)['BinaryRecall']
        recall_samplewise_debris[area_debris_fraction < 0.01] = torch.nan
        val_metrics_samplewise['BinaryRecall_debris'] = recall_samplewise_debris

        # add also the loss to the metrics
        val_metrics_samplewise.update({'loss': loss_samplewise})

        tb_logs = {'val_loss': loss_samplewise.mean()}
        self.log_dict(tb_logs, on_epoch=True, on_step=True, batch_size=len(y_true), sync_dist=True)

        res = {'metrics': val_metrics_samplewise, 'filepaths': batch['fp']}

        return res

    def validation_epoch_end(self, validation_step_outputs):
        avg_tb_logs, df = self.aggregate_step_metrics(validation_step_outputs, split_name='val')

        # show the stats
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', None):
            self._logger.info(f'validation scores stats:\n{df.describe()}')
            self._logger.info(f'validation scores stats (per glacier):\n{df.groupby("gid").mean().describe()}')

        # export the stats if needed
        if self.outdir is not None:
            self.outdir = Path(self.outdir)
            self.outdir.mkdir(parents=True, exist_ok=True)
            fp = self.outdir / 'stats.csv'
            df.to_csv(fp)
            self._logger.info(f'Stats exported to {str(fp)}')
            fp = self.outdir / 'stats_avg_per_glacier.csv'
            df.groupby('gid').mean().to_csv(fp)
            self._logger.info(f'Stats per glacier exported to {str(fp)}')

        # show the epoch as the x-coordinate
        avg_tb_logs['step'] = float(self.current_epoch)
        self.log_dict(avg_tb_logs, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        y_pred = self(batch)
        y_true = batch['mask_all_g'].type(y_pred.dtype).unsqueeze(dim=1)
        mask = ~batch['mask_no_data'].unsqueeze(dim=1)
        loss_samplewise = self.loss(preds=y_pred, targets=y_true, mask=mask, samplewise=True)

        # compute the evaluation metrics for each element in the batch
        val_metrics_samplewise = self.compute_masked_val_metrics(y_pred, y_true, mask)

        # add also the loss to the metrics
        val_metrics_samplewise.update({'loss': loss_samplewise})

        res = {'metrics': val_metrics_samplewise, 'filepaths': batch['fp']}
        res['patch_info'] = batch['patch_info']
        res['preds'] = y_pred
        return res

    def test_epoch_end(self, outputs):
        # collect all filepaths
        filepaths = [y for x in outputs for y in x['filepaths']]

        # ensure that all the predictions are for the same glacier
        if len(set(filepaths)) > 1:
            raise NotImplementedError
        cube_fp = filepaths[0]

        # read the original glacier nc and create accumulators based on its shape
        nc = xr.open_dataset(cube_fp, decode_coords='all')
        preds_acc = torch.zeros(nc.mask_crt_g.shape).to(self.device)
        preds_cnt = torch.zeros(size=preds_acc.shape).to(self.device)
        for j in range(len(outputs)):
            preds = outputs[j]['preds']
            patch_infos = outputs[j]['patch_info']

            batch_size = preds.shape[0]
            for i in range(batch_size):
                crt_pred = preds[i][0]

                # get the (pixel) coordinates of the current patch
                i_minx, i_maxx = patch_infos['bounds_px'][0][i], patch_infos['bounds_px'][2][i]
                i_miny, i_maxy = patch_infos['bounds_px'][1][i], patch_infos['bounds_px'][3][i]
                preds_acc[i_miny:i_maxy, i_minx:i_maxx] += crt_pred
                preds_cnt[i_miny:i_maxy, i_minx:i_maxx] += 1

        preds_acc /= preds_cnt

        # copy to CPU memory
        preds_acc_np = preds_acc.cpu().numpy()

        # store the predictions as xarray based on the original nc
        nc_pred = nc.copy()
        nc_pred['pred'] = (('y', 'x'), preds_acc_np)
        nc_pred['pred'].rio.write_crs(nc_pred.rio.crs, inplace=True)
        nc_pred['pred_b'] = (('y', 'x'), preds_acc_np >= 0.5)
        nc_pred['pred_b'].rio.write_crs(nc_pred.rio.crs, inplace=True)

        rgi_id = Path(cube_fp).parent.parent.name
        gl_num = Path(cube_fp).parent.name
        cube_pred_fp = Path(self.outdir) / rgi_id / f'{gl_num}.nc'
        cube_pred_fp.parent.mkdir(parents=True, exist_ok=True)
        cube_pred_fp.unlink(missing_ok=True)
        nc_pred.to_netcdf(cube_pred_fp)
        self._logger.info(f'Cube with predictions exported to {cube_pred_fp}')
