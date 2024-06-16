_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    'configdota.py',
    'configfcos.py'
]

train_dataloader = dict(batch_size=8, num_workers=8)
optim_wrapper = dict(optimizer=dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=0.0001))
